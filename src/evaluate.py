import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.part2_utils import load_jsonl
from src.retrieval import PaperRetriever


def get_relevant_ids(query: dict) -> List[str]:
    if query.get("relevant_arxiv_ids"):
        return query["relevant_arxiv_ids"]
    if query.get("relevant_paper_ids"):
        return query["relevant_paper_ids"]
    return []


def precision_at_k(predictions: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top_k = predictions[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in top_k if item in relevant_set) / len(top_k)


def recall_at_k(predictions: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in predictions[:k] if item in relevant_set) / len(relevant_set)


def average_precision(predictions: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for idx, item in enumerate(predictions, start=1):
        if item in relevant_set:
            hit_count += 1
            precision_sum += hit_count / idx
    return precision_sum / len(relevant_set)


def ndcg_at_k(predictions: Sequence[str], relevant: Sequence[str], k: int) -> float:
    relevant_set = set(relevant)
    gains = np.array([1.0 if item in relevant_set else 0.0 for item in predictions[:k]], dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_hits = min(len(relevant_set), k)
    if ideal_hits == 0:
        return 0.0
    ideal_gains = np.ones(ideal_hits, dtype=np.float32)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
    idcg = float(np.sum(ideal_gains * ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def run_method(
    retriever: PaperRetriever,
    queries: List[dict],
    mode: str,
    top_k: int,
    alpha: float,
    dense_candidates: int,
) -> Dict[str, float]:
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    map_scores = []

    for query in queries:
        if mode == "tfidf":
            results = retriever.retrieve_tfidf(query["query_text"], k=top_k)
        elif mode == "dense":
            results = retriever.retrieve_dense(query["query_text"], k=top_k)
        else:
            results = retriever.retrieve_hybrid(
                query["query_text"],
                k=top_k,
                alpha=alpha,
                dense_candidates=dense_candidates,
            )
        if query.get("relevant_arxiv_ids"):
            predicted_ids = [result["arxiv_id"] for result in results]
        else:
            predicted_ids = [result["paper_id"] for result in results]
        relevant_ids = get_relevant_ids(query)

        precision_scores.append(precision_at_k(predicted_ids, relevant_ids, top_k))
        recall_scores.append(recall_at_k(predicted_ids, relevant_ids, top_k))
        ndcg_scores.append(ndcg_at_k(predicted_ids, relevant_ids, min(10, top_k)))
        map_scores.append(average_precision(predicted_ids, relevant_ids))

    return {
        "Precision@K": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "Recall@K": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "NDCG@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "MAP": float(np.mean(map_scores)) if map_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate paper-level retrieval methods")
    parser.add_argument("--queries", default="data/queries_val.jsonl")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--dense-model", default=None)
    args = parser.parse_args()

    query_rows = load_jsonl(Path(args.queries))
    retriever = PaperRetriever(config_path=args.config, dense_model_name=args.dense_model)

    print(f"Evaluating {len(query_rows)} labeled queries from {args.queries}\n")
    for mode in ["tfidf", "dense", "hybrid"]:
        print(mode.upper())
        try:
            metrics = run_method(
                retriever,
                query_rows,
                mode=mode,
                top_k=args.top_k,
                alpha=args.alpha,
                dense_candidates=args.dense_candidates,
            )
            for metric_name, value in metrics.items():
                print(f"  {metric_name:<12} {value:.4f}")
        except RuntimeError as exc:
            print(f"  Skipped: {exc}")
        print()


if __name__ == "__main__":
    main()
