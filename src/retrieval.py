import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.dense_encoder import DenseEncoder
from src.part2_utils import (
    build_paper_lookup,
    load_config,
    load_dense_embeddings,
    load_dense_embedding_metadata,
    load_papers,
    load_tfidf_artifacts,
    normalize_scores,
)


class PaperRetriever:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        data_dir = Path(cfg["data_collection"]["output_path"]).parent
        corpus_path = data_dir / "arxiv_corpus_cleaned.jsonl"

        self.cfg = cfg
        self.data_dir = data_dir
        self.papers = load_papers(str(corpus_path))
        self.paper_lookup = build_paper_lookup(self.papers)
        self.paper_ids = [paper["paper_id"] for paper in self.papers]
        self.vectorizer, self.tfidf_matrix = load_tfidf_artifacts(data_dir)
        self.dense_embeddings = load_dense_embeddings(data_dir)
        self.dense_metadata = load_dense_embedding_metadata(data_dir)
        self.dense_model_name = cfg["embeddings"]["dense_model"]
        self._dense_encoder = None

    def _load_dense_encoder(self):
        self._validate_dense_setup()
        if self._dense_encoder is not None:
            return self._dense_encoder
        self._dense_encoder = DenseEncoder(
            model_name=self.dense_model_name,
            max_length=self.cfg["embeddings"]["dense_max_length"],
            document_adapter=self.dense_metadata.get("document_adapter") if self.dense_metadata else None,
            query_adapter=self.dense_metadata.get("query_adapter") if self.dense_metadata else None,
        )
        self._dense_encoder.load()
        return self._dense_encoder

    def _validate_dense_setup(self):
        if self.dense_metadata is None:
            raise RuntimeError(
                "Missing data/dense_embeddings_meta.json. "
                "Regenerate dense embeddings with `python3 run_part1.py` so retrieval can verify compatibility."
            )
        if self.dense_metadata.get("simulated"):
            raise RuntimeError(
                "Stored dense embeddings were generated in simulated mode, "
                "but Part 2 encodes queries with the transformer model. "
                "Regenerate dense embeddings without `--simulated` before running dense or hybrid retrieval."
            )
        stored_model = self.dense_metadata.get("model_name")
        if stored_model != self.dense_model_name:
            raise RuntimeError(
                f"Dense embedding model mismatch: stored embeddings use `{stored_model}`, "
                f"but retrieval is configured for `{self.dense_model_name}`. "
                "Regenerate embeddings or update the config so they match."
            )
        stored_dim = self.dense_metadata.get("embedding_dim")
        if stored_dim is not None and int(stored_dim) != int(self.dense_embeddings.shape[1]):
            raise RuntimeError(
                f"Dense embedding metadata mismatch: metadata dim={stored_dim}, "
                f"array dim={self.dense_embeddings.shape[1]}."
            )

    def _format_results(self, indices: np.ndarray, scores: np.ndarray) -> List[dict]:
        results = []
        for idx, score in zip(indices.tolist(), scores.tolist()):
            paper = self.papers[idx]
            results.append(
                {
                    "paper_id": paper["paper_id"],
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "categories": paper["categories"],
                    "score": float(score),
                }
            )
        return results

    def retrieve_tfidf(self, query: str, k: int = 10) -> List[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
        top_idx = np.argsort(scores)[::-1][:k]
        return self._format_results(top_idx, scores[top_idx])

    def retrieve_dense(self, query: str, k: int = 10) -> List[dict]:
        encoder = self._load_dense_encoder()
        query_vec = encoder.encode_queries([query])
        scores = self.dense_embeddings @ query_vec[0]
        top_idx = np.argsort(scores)[::-1][:k]
        return self._format_results(top_idx, scores[top_idx])

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5,
        dense_candidates: int = 100,
    ) -> List[dict]:
        encoder = self._load_dense_encoder()
        query_vec = encoder.encode_queries([query])
        dense_scores = self.dense_embeddings @ query_vec[0]
        candidate_idx = np.argsort(dense_scores)[::-1][:dense_candidates]

        sparse_query = self.vectorizer.transform([query])
        sparse_scores = (self.tfidf_matrix[candidate_idx] @ sparse_query.T).toarray().ravel()

        dense_norm = normalize_scores(dense_scores[candidate_idx])
        sparse_norm = normalize_scores(sparse_scores)
        hybrid_scores = alpha * dense_norm + (1.0 - alpha) * sparse_norm
        reranked = candidate_idx[np.argsort(hybrid_scores)[::-1][:k]]
        final_scores = hybrid_scores[np.argsort(hybrid_scores)[::-1][:k]]
        return self._format_results(reranked, final_scores)


def main():
    parser = argparse.ArgumentParser(description="Paper-level retrieval over the arXiv corpus")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--mode", choices=["tfidf", "dense", "hybrid"], default="tfidf")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    retriever = PaperRetriever(config_path=args.config)
    if args.mode == "tfidf":
        results = retriever.retrieve_tfidf(args.query, k=args.k)
    elif args.mode == "dense":
        results = retriever.retrieve_dense(args.query, k=args.k)
    else:
        results = retriever.retrieve_hybrid(
            args.query,
            k=args.k,
            alpha=args.alpha,
            dense_candidates=args.dense_candidates,
        )

    print(f"\nMode: {args.mode}")
    print(f"Query: {args.query}\n")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>2}. [{result['paper_id']}] {result['title']} "
            f"(score={result['score']:.4f})"
        )


if __name__ == "__main__":
    main()
