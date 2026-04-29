from __future__ import annotations

import os

from openai import OpenAI


DEFAULT_BASE_URL = "http://promaxgb10-d473.eecs.umich.edu:8000/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class QAGenerator:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or os.environ.get("UM_GPTOSS_BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get("UM_GPTOSS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("UM_GPTOSS_MODEL") or DEFAULT_MODEL
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key) if self.api_key else None

    def format_context(self, results: list[dict]) -> str:
        sections = []
        for idx, item in enumerate(results, start=1):
            evidence_text = item.get("evidence_text") or item.get("abstract", "")
            evidence_text = evidence_text.replace("\n", " ").strip()
            context_type = item.get("context_type", "paper")
            label = "Snippet" if context_type == "chunk" else "Abstract"
            sections.append(
                f"[{idx}] {item['title']}\n"
                f"ArXiv ID: {item['arxiv_id']}\n"
                f"{label}: {evidence_text}\n"
            )
        return "\n".join(sections)

    def generate_answer(self, question: str, context: str) -> str:
        if self._client is None:
            raise RuntimeError(
                "Missing UM_GPTOSS_API_KEY / OPENAI_API_KEY. "
                "Set the API key before running grounded QA."
            )

        prompt = (
            "Answer the user's research question using only the provided paper evidence.\n"
            "When making a claim, cite the supporting snippet index like [1] or [2].\n"
            "If the context is insufficient, say so plainly.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()
