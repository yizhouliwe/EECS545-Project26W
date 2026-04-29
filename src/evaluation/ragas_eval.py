"""
RAGAS-based evaluation for RAG answer quality.

Uses the same UM GPT-oss-120B endpoint as QAGenerator so scores are
produced by the same LLM family that generated the answers.

Metrics (reference-free):
  - Faithfulness: answer claims are grounded in retrieved context
  - AnswerRelevancy: answer addresses the question
"""

import os
from dataclasses import dataclass, field
from typing import List


DEFAULT_BASE_URL = "http://promaxgb10-d473.eecs.umich.edu:8000/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass
class RagasSample:
    query: str
    answer: str
    contexts: List[str]


@dataclass
class RagasResult:
    faithfulness: float | None
    answer_relevancy: float | None
    per_sample: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
        }


class RagasEvaluator:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or os.environ.get("UM_GPTOSS_BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get("UM_GPTOSS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("UM_GPTOSS_MODEL") or DEFAULT_MODEL

    def _build_ragas_llm(self):
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
        except ImportError as exc:
            raise ImportError(
                "Install langchain-openai and ragas: "
                "uv pip install ragas langchain-openai datasets"
            ) from exc

        lc_llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "none",
            model=self.model,
            temperature=0.0,
            timeout=300,
            max_retries=1,
        )
        return LangchainLLMWrapper(lc_llm)

    def _build_ragas_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Subclass BaseRagasEmbeddings using our already-installed sentence_transformers
        # so RAGAS doesn't try to auto-detect OpenAI embeddings.
        from sentence_transformers import SentenceTransformer
        from ragas.embeddings.base import BaseRagasEmbeddings

        st_model = SentenceTransformer(model_name)

        class _STEmbeddings(BaseRagasEmbeddings):
            def embed_query(self, text: str) -> list[float]:
                return st_model.encode(text, convert_to_numpy=True).tolist()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return st_model.encode(texts, convert_to_numpy=True).tolist()

            async def aembed_query(self, text: str) -> list[float]:
                return self.embed_query(text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.embed_documents(texts)

        return _STEmbeddings()

    def evaluate(self, samples: list[RagasSample]) -> RagasResult:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy

        ragas_llm = self._build_ragas_llm()
        ragas_emb = self._build_ragas_embeddings()

        metrics = []

        faithfulness_m = Faithfulness(llm=ragas_llm)
        metrics.append(faithfulness_m)

        answer_rel_m = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
        metrics.append(answer_rel_m)

        dataset = Dataset.from_dict({
            "user_input":         [s.query for s in samples],
            "response":           [s.answer for s in samples],
            "retrieved_contexts": [s.contexts for s in samples],
        })

        from ragas import RunConfig
        run_config = RunConfig(timeout=300, max_retries=1, max_wait=30, max_workers=1)
        result = evaluate(dataset=dataset, metrics=metrics, embeddings=ragas_emb, run_config=run_config)
        df = result.to_pandas()

        def _mean(col: str) -> float | None:
            if col not in df.columns:
                return None
            vals = df[col].dropna()
            return float(vals.mean()) if len(vals) > 0 else None

        return RagasResult(
            faithfulness=_mean("faithfulness"),
            answer_relevancy=_mean("answer_relevancy"),
            per_sample=df.to_dict(orient="records"),
        )
