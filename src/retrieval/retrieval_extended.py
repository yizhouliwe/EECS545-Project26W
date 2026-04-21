from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from src.features.dense_encoder import DenseEncoder
from src.utils.helpers import (
    build_paper_lookup,
    dense_embedding_filename,
    dense_embedding_metadata_filename,
    l2_normalize,
    load_config,
    load_dense_embedding_metadata,
    load_dense_embeddings,
    load_papers,
)


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
                    "paper_id": paper["paper_id"],
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "categories": paper["categories"],
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
