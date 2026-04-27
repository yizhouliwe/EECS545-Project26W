from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Sequence

from openai import OpenAI


DEFAULT_BASE_URL = "http://promaxgb10-d473.eecs.umich.edu:8000/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_FACETS = [
    "background_survey",
    "concrete_system",
    "method_algorithm",
    "dataset_benchmark",
    "evaluation_results",
    "efficiency_scaling",
]


@dataclass
class RefinementResult:
    rewritten_query: str
    facet_weights: List[float]
    explanation: str
    raw_response: str


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"LLM response did not contain JSON: {text[:200]}")
    return json.loads(text[start:end + 1])


class LLMRefinement:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        facets: Sequence[str] | None = None,
    ):
        self.base_url = base_url or os.environ.get("UM_GPTOSS_BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get("UM_GPTOSS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("UM_GPTOSS_MODEL") or DEFAULT_MODEL
        self.facets = list(facets or DEFAULT_FACETS)
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key) if self.api_key else None

    def refine_query(
        self,
        original_query: str,
        current_query: str,
        retrieved_titles: Sequence[str],
        user_feedback_text: str,
    ) -> RefinementResult:
        if self._client is None:
            raise RuntimeError(
                "Missing UM_GPTOSS_API_KEY / OPENAI_API_KEY. "
                "Set the API key before using LLM-based refinement."
            )

        titles_text = "\n".join(f"- {title}" for title in retrieved_titles[:8]) or "- No retrieved titles available"
        facets_text = ", ".join(self.facets)
        prompt = (
            "You are refining a scientific-literature search query after relevance feedback.\n"
            "Return valid JSON only with this schema:\n"
            "{\n"
            '  "rewritten_query": "short improved search query",\n'
            f'  "facet_weights": [{", ".join(["1.0"] * len(self.facets))}],\n'
            '  "explanation": "one short sentence"\n'
            "}\n"
            "Rules:\n"
            "- Keep the rewritten query concise and targeted.\n"
            "- Incorporate the user feedback directly.\n"
            f"- facet_weights must contain exactly {len(self.facets)} floats in [0.5, 1.5].\n"
            f"- Facet order: {facets_text}.\n"
            "- Prefer terms useful for scientific-paper retrieval, not conversational prose.\n\n"
            f"Original query: {original_query}\n"
            f"Current query: {current_query}\n"
            f"Retrieved titles:\n{titles_text}\n"
            f"User feedback: {user_feedback_text}\n"
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        payload = _extract_json_object(content)

        rewritten_query = str(payload.get("rewritten_query", current_query)).strip() or current_query
        facet_weights = payload.get("facet_weights", [1.0] * len(self.facets))
        if not isinstance(facet_weights, list) or len(facet_weights) != len(self.facets):
            facet_weights = [1.0] * len(self.facets)
        clipped_weights = [min(1.5, max(0.5, float(weight))) for weight in facet_weights]

        return RefinementResult(
            rewritten_query=rewritten_query,
            facet_weights=clipped_weights,
            explanation=str(payload.get("explanation", "")).strip(),
            raw_response=content,
        )
