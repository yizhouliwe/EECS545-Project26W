from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from src.features.dense_encoder import DenseEncoder
from src.utils.helpers import (
    build_paper_lookup,
    chunk_embedding_filename,
    dense_embedding_filename,
    dense_embedding_metadata_filename,
    l2_normalize,
    load_config,
    load_dense_embedding_metadata,
    load_dense_embeddings,
    load_jsonl,
    load_papers,
)


class ChunkRetriever:
    def __init__(
        self,
        data_dir: Path,
        papers: List[dict],
        dense_model_name: str,
        dense_encoder_loader,
    ):
        self.data_dir = data_dir
        self.papers = papers
        self.paper_lookup = build_paper_lookup(papers)
        self.dense_model_name = dense_model_name
        self._dense_encoder_loader = dense_encoder_loader

        self.chunks = load_jsonl(data_dir / "arxiv_chunks.jsonl")
        self.paper_to_chunks: dict[str, list[tuple[int, dict]]] = {}
        for idx, chunk in enumerate(self.chunks):
            self.paper_to_chunks.setdefault(chunk["paper_id"], []).append((idx, chunk))

        self.chunk_embeddings_path = data_dir / chunk_embedding_filename(dense_model_name)
        self.chunk_embeddings = (
            np.load(self.chunk_embeddings_path)
            if self.chunk_embeddings_path.exists()
            else None
        )

    def _chunk_text_for_embedding(self, chunk: dict) -> str:
        paper = self.paper_lookup.get(chunk["paper_id"], {})
        title = paper.get("title", "").strip()
        text = chunk["chunk_text"].strip()
        if self.dense_model_name == "allenai/specter2_base":
            return f"{title} [SEP] {text}" if title else text
        return f"{title}. {text}" if title else text

    def _get_candidate_chunks(self, paper_ids: List[str]) -> list[tuple[int, dict]]:
        candidates: list[tuple[int, dict]] = []
        for paper_id in paper_ids:
            candidates.extend(self.paper_to_chunks.get(paper_id, []))
        return candidates

    def _get_chunk_embeddings(self, candidates: list[tuple[int, dict]]) -> np.ndarray:
        if not candidates:
            return np.zeros((0, 0), dtype=np.float32)

        if self.chunk_embeddings is not None:
            indices = [idx for idx, _ in candidates]
            return self.chunk_embeddings[indices]

        encoder = self._dense_encoder_loader()
        texts = [self._chunk_text_for_embedding(chunk) for _, chunk in candidates]
        return encoder.encode_documents(texts)

    def retrieve(
        self,
        query_vector: np.ndarray,
        paper_ids: List[str],
        top_m: int = 8,
        token_budget: int = 3000,
        max_chunks_per_paper: int = 1,
    ) -> List[dict]:
        candidates = self._get_candidate_chunks(paper_ids)
        if not candidates:
            return []

        query_vector = l2_normalize(query_vector)
        chunk_embeddings = l2_normalize(self._get_chunk_embeddings(candidates))
        scores = chunk_embeddings @ query_vector
        ranked_indices = np.argsort(scores)[::-1]

        selected: list[dict] = []
        used_tokens = 0
        chunks_per_paper: dict[str, int] = {}
        for ranked_idx in ranked_indices:
            _, chunk = candidates[int(ranked_idx)]
            paper_id = chunk["paper_id"]
            if chunks_per_paper.get(paper_id, 0) >= max_chunks_per_paper:
                continue
            chunk_tokens = int(chunk.get("tokens_est", len(chunk["chunk_text"].split())))
            if selected and used_tokens + chunk_tokens > token_budget:
                continue

            paper = self.paper_lookup.get(paper_id, {})
            selected.append(
                {
                    "context_type": "chunk",
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": paper_id,
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "title": paper.get("title", ""),
                    "evidence_text": chunk["chunk_text"],
                    "chunk_index": chunk.get("chunk_index"),
                    "tokens_est": chunk_tokens,
                    "score": float(scores[int(ranked_idx)]),
                }
            )
            used_tokens += chunk_tokens
            chunks_per_paper[paper_id] = chunks_per_paper.get(paper_id, 0) + 1
            if len(selected) >= top_m or used_tokens >= token_budget:
                break
        return selected


class PaperRetrieverExtended:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        dense_model_name: Optional[str] = None,
    ):
        cfg = load_config(config_path)
        data_dir = Path(cfg["data_collection"]["output_path"]).parent
        corpus_path = data_dir / "arxiv_corpus_cleaned.jsonl"

        self.cfg = cfg
        self.data_dir = data_dir
        self.papers = load_papers(str(corpus_path))
        self.paper_lookup = build_paper_lookup(self.papers)
        self.paper_ids = [paper["paper_id"] for paper in self.papers]
        self.paper_id_to_index = {
            paper_id: idx for idx, paper_id in enumerate(self.paper_ids)
        }

        # Stage 3 defaults to the scientific-domain SPECTER2 encoder.
        self.dense_model_name = dense_model_name or "allenai/specter2_base"
        self.dense_embeddings_path = data_dir / dense_embedding_filename(
            self.dense_model_name
        )
        self.dense_metadata_path = data_dir / dense_embedding_metadata_filename(
            self.dense_model_name
        )
        self.dense_embeddings = load_dense_embeddings(data_dir, self.dense_model_name)
        self.dense_metadata = load_dense_embedding_metadata(
            data_dir, self.dense_model_name
        )
        self._dense_encoder = None
        self.chunk_retriever = ChunkRetriever(
            data_dir=data_dir,
            papers=self.papers,
            dense_model_name=self.dense_model_name,
            dense_encoder_loader=self._load_dense_encoder,
        )

    def _load_dense_encoder(self):
        self._validate_dense_setup()
        if self._dense_encoder is not None:
            return self._dense_encoder
        self._dense_encoder = DenseEncoder(
            model_name=self.dense_model_name,
            max_length=self.cfg["embeddings"]["dense_max_length"],
            document_adapter=self.dense_metadata.get("document_adapter")
            if self.dense_metadata
            else None,
            query_adapter=self.dense_metadata.get("query_adapter")
            if self.dense_metadata
            else None,
        )
        self._dense_encoder.load()
        return self._dense_encoder

    def _validate_dense_setup(self):
        if self.dense_metadata is None:
            raise RuntimeError(
                f"Missing {self.dense_metadata_path}. Regenerate dense embeddings for "
                f"`{self.dense_model_name}` before running Part 3."
            )
        stored_model = self.dense_metadata.get("model_name")
        if stored_model != self.dense_model_name:
            raise RuntimeError(
                f"Dense embedding model mismatch: metadata uses `{stored_model}`, "
                f"but Stage 3 expects `{self.dense_model_name}`."
            )
        stored_dim = self.dense_metadata.get("embedding_dim")
        if stored_dim is not None and int(stored_dim) != int(
            self.dense_embeddings.shape[1]
        ):
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
                    "context_type": "paper",
                    "paper_id": paper["paper_id"],
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "evidence_text": paper["abstract"],
                    "categories": paper["categories"],
                    "tokens_est": len(paper["abstract"].split()),
                    "score": float(score),
                }
            )
        return results

    def encode_query(self, query: str) -> np.ndarray:
        encoder = self._load_dense_encoder()
        return encoder.encode_queries([query])[0]

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        encoder = self._load_dense_encoder()
        return encoder.encode_documents(texts)

    def get_embedding(self, paper_id: str) -> np.ndarray:
        return self.dense_embeddings[self.paper_id_to_index[paper_id]]

    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        return self.retrieve_by_vector(self.encode_query(query), k=k)

    def retrieve_by_vector(self, query_vector: np.ndarray, k: int = 10) -> List[dict]:
        query_vector = l2_normalize(query_vector)
        scores = self.dense_embeddings @ query_vector
        top_idx = np.argsort(scores)[::-1][:k]
        return self._format_results(top_idx, scores[top_idx])

    def retrieve_chunks(
        self,
        query_vector: np.ndarray,
        paper_ids: List[str],
        top_m: int = 8,
        token_budget: int = 3000,
    ) -> List[dict]:
        return self.chunk_retriever.retrieve(
            query_vector=query_vector,
            paper_ids=paper_ids,
            top_m=top_m,
            token_budget=token_budget,
        )
