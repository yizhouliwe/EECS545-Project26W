"""
Pool candidates from TF-IDF, Dense, and Hybrid retrieval for each query,
then use GPT-oss-120B to judge relevance and update labeled_queries.jsonl.
Requires UM VPN.
"""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from src.utils import load_config, load_papers
from src.retrieval import PaperRetriever


UM_CLIENT = OpenAI(
    base_url="http://promaxgb10-d668.eecs.umich.edu:8000/v1",
    api_key="api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy",
)
MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def judge_relevance(query: str, title: str, abstract: str) -> bool:
    prompt = (
        f"Query: {query[:300]}\n\n"
        f"Paper title: {title[:150]}\n"
        f"Abstract: {abstract[:400]}\n\n"
        "Is this paper relevant to the query? Reply with only YES or NO."
    )
    try:
        response = UM_CLIENT.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            print(
                f"    LLM returned None. Finish reason: {response.choices[0].finish_reason}"
            )
            return False
        answer = content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"    LLM error: {e}")
        return False


def pool_candidates(retriever: PaperRetriever, query: str, top_k: int = 20) -> list:
    candidates = {}
    methods = [
        ("tfidf", lambda q, k: retriever.retrieve_tfidf(q, k)),
        ("dense", lambda q, k: retriever.retrieve_dense(q, k)),
        ("hybrid", lambda q, k: retriever.retrieve_hybrid(q, k=k)),
    ]
    for mode, fn in methods:
        try:
            results = fn(query, top_k)
            for r in results:
                aid = r.get("arxiv_id")
                if aid and aid not in candidates:
                    candidates[aid] = r
        except Exception as e:
            print(f"  [{mode}] retrieval failed: {e}")
    return list(candidates.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/labeled_queries.jsonl")
    parser.add_argument("--output", default="data/labeled_queries.jsonl")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Seconds between LLM calls"
    )
    args = parser.parse_args()

    retriever = PaperRetriever()

    rows = []
    with open(args.queries) as f:
        for line in f:
            rows.append(json.loads(line))

    for row in rows:
        qid = row["query_id"]
        query = row["query_text"]

        if (
            row.get("relevant_arxiv_ids")
            and row.get("notes", "unlabeled") != "unlabeled"
        ):
            print(f"\n[{qid}] Skipping (already labeled: {row['notes']})")
            continue

        print(f"\n[{qid}] {query[:80]}...")

        candidates = pool_candidates(retriever, query, top_k=args.top_k)
        print(f"  Pooled {len(candidates)} unique candidates")

        relevant_ids = []
        for paper in candidates:
            aid = paper.get("arxiv_id", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            is_relevant = judge_relevance(query, title, abstract)
            print(f"  {'YES' if is_relevant else 'NO '} {aid}: {title[:60]}")
            if is_relevant:
                relevant_ids.append(aid)
            time.sleep(args.delay)

        row["relevant_arxiv_ids"] = relevant_ids
        row["notes"] = (
            f"LLM-judged: {len(relevant_ids)}/{len(candidates)} candidates relevant."
        )
        print(f"  -> {len(relevant_ids)} relevant papers found")

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. Updated {args.output}")


if __name__ == "__main__":
    main()
