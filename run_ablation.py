import argparse
import collections
import csv
from pathlib import Path

from src.feedback.feedback_logic import apply_facet_weights, apply_rocchio
from src.feedback.llm_refinement import LLMRefinement
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.evaluate import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from src.retrieval.retrieval import PaperRetriever
from src.utils.helpers import load_jsonl


def normalize_id(paper_id: str) -> str:
    pid = str(paper_id).lower().strip()
    if pid.startswith("arxiv:"):
        pid = pid.replace("arxiv:", "")
    if "v" in pid:
        pid = pid.split("v")[0]
    return pid


def compute_metrics(predicted_ids, relevant_ids, k):
    return {
        "precision": precision_at_k(predicted_ids, relevant_ids, k),
        "recall": recall_at_k(predicted_ids, relevant_ids, k),
        "ndcg": ndcg_at_k(predicted_ids, relevant_ids, min(10, k)),
        "map": average_precision(predicted_ids, relevant_ids),
    }


def run_retrieval_methods(retriever, query_rows, top_k, alpha, dense_candidates):
    records = []
    for mode in ["tfidf", "dense", "hybrid"]:
        print(f"\n  Running {mode.upper()}...")
        for row in query_rows:
            if mode == "tfidf":
                results = retriever.retrieve_tfidf(row["query_text"], k=top_k)
            elif mode == "dense":
                results = retriever.retrieve_dense(row["query_text"], k=top_k)
            else:
                results = retriever.retrieve_hybrid(
                    row["query_text"], k=top_k, alpha=alpha, dense_candidates=dense_candidates
                )
            predicted_ids = [normalize_id(r["arxiv_id"]) for r in results]
            relevant_ids = [normalize_id(i) for i in row.get("relevant_arxiv_ids", [])]
            metrics = compute_metrics(predicted_ids, relevant_ids, top_k)
            records.append({"method": mode, "round": 0, "query_id": row.get("query_id", ""), **metrics})
    return records


def run_rocchio_feedback(retriever, query_rows, top_k, rounds):
    records = []
    for row in query_rows:
        relevant_ids_norm = {normalize_id(i) for i in row.get("relevant_arxiv_ids", [])}
        relevant_ids_list = list(relevant_ids_norm)
        current_vector = retriever.encode_query(row["query_text"])

        for round_idx in range(rounds + 1):
            results = retriever.retrieve_by_vector(current_vector, k=top_k)
            predicted_ids = [normalize_id(r["arxiv_id"]) for r in results]
            metrics = compute_metrics(predicted_ids, relevant_ids_list, top_k)
            records.append({
                "method": "rocchio",
                "round": round_idx,
                "query_id": row.get("query_id", ""),
                **metrics,
            })

            if round_idx >= rounds:
                break

            hits = [r for r in results if normalize_id(r["arxiv_id"]) in relevant_ids_norm]
            if not hits:
                break
            positive_vectors = [retriever.get_embedding(r["paper_id"]) for r in hits]
            current_vector = apply_rocchio(current_vector, positive_vectors)

    return records


def run_llm_feedback(retriever, query_rows, top_k, rounds, feedback_method):
    llm_refiner = LLMRefinement()
    records = []
    for row in query_rows:
        relevant_ids_norm = {normalize_id(i) for i in row.get("relevant_arxiv_ids", [])}
        relevant_ids_list = list(relevant_ids_norm)
        original_query = row["query_text"]
        current_query = original_query
        current_vector = retriever.encode_query(current_query)

        for round_idx in range(rounds + 1):
            results = retriever.retrieve_by_vector(current_vector, k=top_k)
            predicted_ids = [normalize_id(r["arxiv_id"]) for r in results]
            metrics = compute_metrics(predicted_ids, relevant_ids_list, top_k)
            records.append({
                "method": feedback_method,
                "round": round_idx,
                "query_id": row.get("query_id", ""),
                **metrics,
            })

            if round_idx >= rounds:
                break

            hits = [r for r in results if normalize_id(r["arxiv_id"]) in relevant_ids_norm]
            if not hits:
                hits = [results[0]] if results else []
            if not hits:
                break

            positive_vectors = [retriever.get_embedding(r["paper_id"]) for r in hits]
            feedback_text = "These papers are relevant. Find more like: " + "; ".join(r["title"] for r in hits[:3])

            if feedback_method == "combined":
                rocchio_vector = apply_rocchio(current_vector, positive_vectors)
                seed_results = retriever.retrieve_by_vector(rocchio_vector, k=top_k)
            else:
                seed_results = results

            refinement = llm_refiner.refine_query(
                original_query=original_query,
                current_query=current_query,
                retrieved_titles=[r["title"] for r in seed_results],
                user_feedback_text=feedback_text,
            )
            current_query = refinement.rewritten_query
            current_vector = apply_facet_weights(retriever.encode_query(current_query), refinement.facet_weights)

    return records


def run_chunk_retrieval(retriever, query_rows, top_k):
    records = []
    for row in query_rows:
        query_vector = retriever.encode_query(row["query_text"])
        paper_results = retriever.retrieve_by_vector(query_vector, k=top_k)
        paper_ids = [r["paper_id"] for r in paper_results]
        chunk_results = retriever.retrieve_chunks(query_vector, paper_ids, top_m=top_k)
        predicted_ids = [normalize_id(r["arxiv_id"]) for r in chunk_results]
        relevant_ids = [normalize_id(i) for i in row.get("relevant_arxiv_ids", [])]
        metrics = compute_metrics(predicted_ids, relevant_ids, top_k)
        records.append({"method": "chunk", "round": 0, "query_id": row.get("query_id", ""), **metrics})
    return records


def summarize(records):
    by_method_round = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in records:
        key = (rec["method"], rec["round"])
        for m in ("precision", "recall", "ndcg", "map"):
            by_method_round[key][m].append(rec[m])

    print(f"\n{'=' * 75}")
    print(f"  {'Method':<18} {'Round':>5} {'P@K':>8} {'R@K':>8} {'NDCG@10':>10} {'MAP':>8}")
    print(f"  {'-' * 61}")
    for (method, rnd) in sorted(by_method_round):
        vals = by_method_round[(method, rnd)]
        n = len(vals["precision"])
        print(
            f"  {method:<18} {rnd:>5} "
            f"{sum(vals['precision'])/n:>8.3f} "
            f"{sum(vals['recall'])/n:>8.3f} "
            f"{sum(vals['ndcg'])/n:>10.3f} "
            f"{sum(vals['map'])/n:>8.3f}"
        )
    print(f"{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(description="Full ablation: retrieval methods + Rocchio feedback")
    parser.add_argument("--queries", default="data/queries_test.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--rounds", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-feedback", action="store_true")
    parser.add_argument("--include-llm", action="store_true", help="Also run LLM and Combined feedback (requires API key)")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    query_rows = load_jsonl(Path(args.queries))
    if args.max_queries is not None:
        query_rows = query_rows[: args.max_queries]
    print(f"Loaded {len(query_rows)} queries from {args.queries}")

    records = []

    if not args.skip_retrieval:
        print("\n[Retrieval] TF-IDF / Dense / Hybrid")
        paper_retriever = PaperRetriever(dense_model_name=args.dense_model)
        records += run_retrieval_methods(paper_retriever, query_rows, args.top_k, args.alpha, args.dense_candidates)

    if not args.skip_feedback:
        print("\n[Feedback] Rocchio relevance feedback")
        dense_model = args.dense_model or "sentence-transformers/all-MiniLM-L6-v2"
        dense_retriever = DenseRetriever(dense_model_name=dense_model)
        records += run_rocchio_feedback(dense_retriever, query_rows, args.top_k, args.rounds)
        print("\n[Chunk] Chunk-level evidence retrieval")
        records += run_chunk_retrieval(dense_retriever, query_rows, args.top_k)

    if args.include_llm:
        for method in ["llm", "combined"]:
            print(f"\n[Feedback] {method.upper()} relevance feedback")
            records += run_llm_feedback(dense_retriever, query_rows, args.top_k, args.rounds, method)

    summarize(records)

    if args.output_csv and records:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "round", "query_id", "precision", "recall", "ndcg", "map"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
