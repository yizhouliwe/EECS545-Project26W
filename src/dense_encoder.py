from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (embeddings / norms).astype(np.float32)


class DenseEncoder:
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        device: Optional[str] = None,
        document_adapter: Optional[str] = None,
        query_adapter: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.document_adapter = document_adapter
        self.query_adapter = query_adapter
        self.local_files_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}

        self.backend = "sentence_transformers"
        self.model = None
        self.tokenizer = None
        self._loaded_adapters: set[str] = set()
        self._transformers_model = None

        if model_name == "allenai/specter2_base":
            self.backend = "specter2_adapters"
            self.document_adapter = document_adapter or "allenai/specter2"
            self.query_adapter = query_adapter or "allenai/specter2_adhoc_query"
        elif model_name == "allenai/scibert_scivocab_uncased":
            self.backend = "hf_mean_pooling"

    def load(self):
        if self.model is not None:
            return

        if self.backend == "specter2_adapters":
            try:
                import torch
                from adapters import AutoAdapterModel
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "SPECTER2 retrieval requires the `adapters` package. "
                    "Install it with `pip install adapters` and then regenerate dense embeddings."
                ) from exc

            self._torch = torch
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            logger.info(
                "Loading SPECTER2 base tokenizer from %s (local_files_only=%s)",
                self.model_name,
                self.local_files_only,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            logger.info(
                "Loading SPECTER2 base model from %s (local_files_only=%s)",
                self.model_name,
                self.local_files_only,
            )
            self.model = AutoAdapterModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            if self.device:
                self.model.to(self.device)
            self.model.eval()
        elif self.backend == "hf_mean_pooling":
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "SciBERT retrieval requires transformers and torch."
                ) from exc

            self._torch = torch
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            logger.info(
                "Loading HF tokenizer from %s (local_files_only=%s)",
                self.model_name,
                self.local_files_only,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            logger.info(
                "Tokenizer loaded. Loading HF model from %s (local_files_only=%s)",
                self.model_name,
                self.local_files_only,
            )
            self._transformers_model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True,
            )
            logger.info("HF model loaded.")
            if self.device:
                self._transformers_model.to(self.device)
            self._transformers_model.eval()
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Dense retrieval requires sentence-transformers. "
                    "Install it with `pip install sentence-transformers`."
                ) from exc
            logger.info(
                "Loading sentence-transformers model from %s (local_files_only=%s)",
                self.model_name,
                self.local_files_only,
            )
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )

    def _ensure_adapter(self, adapter_name: str):
        if adapter_name in self._loaded_adapters:
            self.model.set_active_adapters(adapter_name)
            return
        logger.info(
            "Loading adapter %s (local_files_only=%s)",
            adapter_name,
            self.local_files_only,
        )
        self.model.load_adapter(
            adapter_name,
            source="hf",
            load_as=adapter_name,
            set_active=True,
            local_files_only=self.local_files_only,
        )
        if self.device:
            # adapters are loaded after the base model is moved, so keep all weights on the same device.
            self.model.to(self.device)
        self._loaded_adapters.add(adapter_name)

    def _encode_with_adapter(
        self,
        texts: List[str],
        adapter_name: str,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self._ensure_adapter(adapter_name)

        batches = []
        batch_starts = range(0, len(texts), batch_size)
        if show_progress_bar:
            try:
                from tqdm.auto import tqdm
                batch_starts = tqdm(batch_starts, total=(len(texts) + batch_size - 1) // batch_size)
            except ImportError:
                pass

        for start in batch_starts:
            text_batch = texts[start:start + batch_size]
            if start == 0:
                logger.info("Encoding first batch with adapter-backed transformer backend.")
            inputs = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=self.max_length,
            )
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self.model(**inputs)
                # AllenAI's SPECTER2 usage pools the first token representation.
                cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                batches.append(cls_embeddings)

        return _normalize(np.vstack(batches))

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.load()
        if self.backend == "specter2_adapters":
            return self._encode_with_adapter(
                texts,
                adapter_name=self.document_adapter,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
        if self.backend == "hf_mean_pooling":
            return self._encode_with_mean_pooling(texts, batch_size=batch_size)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def encode_queries(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self.load()
        if self.backend == "specter2_adapters":
            return self._encode_with_adapter(
                texts,
                adapter_name=self.query_adapter,
                batch_size=batch_size,
            )
        if self.backend == "hf_mean_pooling":
            return self._encode_with_mean_pooling(texts, batch_size=batch_size)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def _encode_with_mean_pooling(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), batch_size):
            text_batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self._transformers_model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked = token_embeddings * attention_mask
                pooled = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                batches.append(pooled.detach().cpu().numpy())

        return _normalize(np.vstack(batches))
