from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.part2_utils import load_config, load_jsonl
from src.retrieval import PaperRetriever


class PaperRetrieverExtended(PaperRetriever):
    def retrieve_by_vector(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search the paper embedding space with a precomputed query vector."""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores = self.dense_embeddings @ query_vector.T
        scores = scores.ravel()
        top_k_idx = np.argsort(scores)[::-1][:k]
        return self._format_results(top_k_idx, scores[top_k_idx])

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a query with the same dense model used for paper retrieval."""
        encoder = self._load_dense_encoder()
        return encoder.encode_queries([query_text])[0]


class ChunkRetriever:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        chunk_path: str | None = None,
        embeddings_path: str | None = None,
        paper_retriever: PaperRetriever | None = None,
    ):
        cfg = load_config(config_path)
        data_dir = Path(cfg["data_collection"]["output_path"]).parent

        self.cfg = cfg
        self.data_dir = data_dir
        self.chunk_path = Path(chunk_path) if chunk_path else data_dir / "arxiv_chunks.jsonl"
        self.embeddings_path = Path(embeddings_path) if embeddings_path else data_dir / "chunk_embeddings.npy"
        self.paper_retriever = paper_retriever or PaperRetriever(config_path=config_path)
        self.chunks = load_jsonl(self.chunk_path)

        self.chunk_texts = [chunk["chunk_text"] for chunk in self.chunks]
        self.chunk_by_paper: Dict[str, List[int]] = {}
        for idx, chunk in enumerate(self.chunks):
            self.chunk_by_paper.setdefault(chunk["paper_id"], []).append(idx)

        self._chunk_embeddings: np.ndarray | None = None
        self._chunk_tfidf_matrix = None

    def encode_query(self, query_text: str) -> np.ndarray:
        return self.paper_retriever.encode_query(query_text)

    def _load_chunk_embeddings(self) -> np.ndarray:
        if self._chunk_embeddings is not None:
            return self._chunk_embeddings

        if self.embeddings_path.exists():
            self._chunk_embeddings = np.load(self.embeddings_path)
            return self._chunk_embeddings

        encoder = self.paper_retriever._load_dense_encoder()
        batch_size = int(self.cfg["embeddings"].get("dense_batch_size", 64))
        chunk_embeddings = encoder.encode_documents(
            self.chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, chunk_embeddings.astype(np.float32))
        self._chunk_embeddings = chunk_embeddings.astype(np.float32)
        return self._chunk_embeddings

    def _load_chunk_tfidf_matrix(self):
        if self._chunk_tfidf_matrix is not None:
            return self._chunk_tfidf_matrix
        self._chunk_tfidf_matrix = self.paper_retriever.vectorizer.transform(self.chunk_texts)
        return self._chunk_tfidf_matrix

    def _score_chunk_subset_dense(
        self,
        query_vector: np.ndarray,
        candidate_indices: Sequence[int],
    ) -> np.ndarray:
        if query_vector.ndim > 1:
            query_vector = query_vector[0]
        embeddings = self._load_chunk_embeddings()
        subset = embeddings[np.asarray(candidate_indices)]
        return subset @ query_vector

    def _score_chunk_subset_tfidf(
        self,
        query_text: str,
        candidate_indices: Sequence[int],
    ) -> np.ndarray:
        chunk_matrix = self._load_chunk_tfidf_matrix()
        query_vec = self.paper_retriever.vectorizer.transform([query_text])
        return (chunk_matrix[np.asarray(candidate_indices)] @ query_vec.T).toarray().ravel()

    def select_chunks_for_papers(
        self,
        paper_ids: Sequence[str],
        query_text: str,
        query_vector: np.ndarray | None = None,
        token_budget: int = 3000,
        max_chunks: int | None = None,
        score_mode: str = "dense",
    ) -> List[Dict]:
        candidate_indices: List[int] = []
        for paper_id in paper_ids:
            candidate_indices.extend(self.chunk_by_paper.get(paper_id, []))

        if not candidate_indices:
            return []

        if score_mode == "dense":
            if query_vector is None:
                query_vector = self.encode_query(query_text)
            scores = self._score_chunk_subset_dense(query_vector, candidate_indices)
        elif score_mode == "tfidf":
            scores = self._score_chunk_subset_tfidf(query_text, candidate_indices)
        else:
            raise ValueError(f"Unsupported chunk score mode: {score_mode}")

        ranked_positions = np.argsort(scores)[::-1]
        selected: List[Dict] = []
        used_tokens = 0

        for pos in ranked_positions:
            chunk_idx = candidate_indices[int(pos)]
            score = float(scores[int(pos)])
            chunk = self.chunks[chunk_idx]
            token_count = int(chunk.get("tokens_est", 0))

            if selected and used_tokens + token_count > token_budget:
                continue

            paper = self.paper_retriever.paper_lookup.get(chunk["paper_id"], {})
            selected.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "arxiv_id": paper.get("arxiv_id", chunk["paper_id"]),
                    "title": paper.get("title", ""),
                    "chunk_text": chunk["chunk_text"],
                    "chunk_index": chunk.get("chunk_index"),
                    "chunk_type": chunk.get("chunk_type", "abstract"),
                    "tokens_est": token_count,
                    "score": score,
                }
            )
            used_tokens += token_count

            if max_chunks is not None and len(selected) >= max_chunks:
                break
            if used_tokens >= token_budget:
                break

        return selected
