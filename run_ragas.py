"""
RAGAS evaluation of RAG answer quality on the test query set.

For each configuration (context mode x feedback method), retrieves context,
generates an answer via QAGenerator, then scores with RAGAS reference-free
metrics: faithfulness, answer_relevancy, context_precision.

Usage:
    # Paper context, no feedback
    python run_ragas.py --context-mode paper --feedback-method none

    # Chunk context, Rocchio feedback (1 round)
    python run_ragas.py --context-mode chunk --feedback-method rocchio --rounds 1

    # Run all four key configurations and write combined CSV
    python run_ragas.py --all-configs --output-csv outputs/ragas_results.csv
"""

import argparse
import csv
from pathlib import Path

from src.evaluation.ragas_eval import RagasEvaluator, RagasSample
from src.feedback.feedback_logic import apply_facet_weights, apply_rocchio
from src.feedback.llm_refinement import LLMRefinement
from src.rag.qa_engine import QAGenerator
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.helpers import load_jsonl


def normalize_id(paper_id: str) -> str:
    pid = str(paper_id).lower().strip()
    if pid.startswith("arxiv:"):
        pid = pid.replace("arxiv:", "")
    if "v" in pid:
        pid = pid.split("v")[0]
    return pid


def build_sample(
    retriever: DenseRetriever,
    qa_gen: QAGenerator,
    llm_refiner: LLMRefinement | None,
    row: dict,
    feedback_method: str,
    rounds: int,
    top_k: int,
    context_mode: str,
    chunk_top_m: int,
    chunk_token_budget: int,
) -> RagasSample | None:
    query = row["query_text"]
    relevant_ids = {normalize_id(i) for i in row.get("relevant_arxiv_ids", [])}

    current_query = query
    current_vector = retriever.encode_query(current_query)

    for round_idx in range(rounds):
        results = retriever.retrieve_by_vector(current_vector, k=top_k)
        hits = [r for r in results if normalize_id(r["arxiv_id"]) in relevant_ids]
        if not hits:
            hits = [results[0]] if results else []
        if not hits:
            break

        positive_vectors = [retriever.get_embedding(r["paper_id"]) for r in hits]
        feedback_text = "These papers are relevant. Find more like: " + "; ".join(
            r["title"] for r in hits[:3]
        )

        if feedback_method == "rocchio":
            current_vector = apply_rocchio(current_vector, positive_vectors)
        elif feedback_method in ("llm", "combined") and llm_refiner is not None:
            if feedback_method == "combined":
                rocchio_vector = apply_rocchio(current_vector, positive_vectors)
                seed_results = retriever.retrieve_by_vector(rocchio_vector, k=top_k)
            else:
                seed_results = results
            refinement = llm_refiner.refine_query(
                original_query=query,
                current_query=current_query,
                retrieved_titles=[r["title"] for r in seed_results],
                user_feedback_text=feedback_text,
            )
            current_query = refinement.rewritten_query
            current_vector = apply_facet_weights(
                retriever.encode_query(current_query), refinement.facet_weights
            )

    final_results = retriever.retrieve_by_vector(current_vector, k=top_k)

    if context_mode == "chunk":
        rag_results = retriever.retrieve_chunks(
            query_vector=current_vector,
            paper_ids=[r["paper_id"] for r in final_results],
            top_m=chunk_top_m,
            token_budget=chunk_token_budget,
        )
    else:
        rag_results = final_results

    if not rag_results:
        return None

    context_str = qa_gen.format_context(rag_results)
    try:
        answer = qa_gen.generate_answer(current_query, context_str)
    except Exception as exc:
        print(f"  QA generation failed for '{query[:50]}': {exc}")
        return None

    contexts = [
        (r.get("evidence_text") or r.get("abstract", "")).replace("\n", " ").strip()
        for r in rag_results
    ]
    return RagasSample(query=query, answer=answer, contexts=contexts)


def run_config(
    retriever: DenseRetriever,
    qa_gen: QAGenerator,
    evaluator: RagasEvaluator,
    query_rows: list[dict],
    feedback_method: str,
    rounds: int,
    context_mode: str,
    top_k: int,
    chunk_top_m: int,
    chunk_token_budget: int,
    llm_refiner: LLMRefinement | None,
) -> dict:
    label = f"{context_mode}+{feedback_method}+r{rounds}"
    print(f"\n{'=' * 60}")
    print(f"  Config: {label}")
    print(f"{'=' * 60}")

    samples = []
    for row in query_rows:
        print(f"  [{row['query_id']}] {row['query_text'][:60]}...")
        sample = build_sample(
            retriever=retriever,
            qa_gen=qa_gen,
            llm_refiner=llm_refiner,
            row=row,
            feedback_method=feedback_method,
            rounds=rounds,
            top_k=top_k,
            context_mode=context_mode,
            chunk_top_m=chunk_top_m,
            chunk_token_budget=chunk_token_budget,
        )
        if sample is not None:
            samples.append(sample)

    if not samples:
        print("  No samples generated — skipping RAGAS.")
        return {
            "config": label,
            "n": 0,
            "faithfulness": None,
            "answer_relevancy": None,
        }

    print(f"\n  Running RAGAS on {len(samples)} samples...")
    result = evaluator.evaluate(samples)
    scores = result.as_dict()

    print(
        f"  Faithfulness:        {scores['faithfulness']:.3f}"
        if scores["faithfulness"] is not None
        else "  Faithfulness:        N/A"
    )
    print(
        f"  Answer Relevancy:    {scores['answer_relevancy']:.3f}"
        if scores["answer_relevancy"] is not None
        else "  Answer Relevancy:    N/A"
    )

    return {"config": label, "n": len(samples), **scores}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation of RAG answer quality"
    )
    parser.add_argument("--queries", default="data/queries_test.jsonl")
    parser.add_argument("--context-mode", choices=["paper", "chunk"], default="chunk")
    parser.add_argument(
        "--feedback-method",
        choices=["none", "rocchio", "llm", "combined"],
        default="none",
    )
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-top-m", type=int, default=8)
    parser.add_argument("--chunk-token-budget", type=int, default=3000)
    parser.add_argument(
        "--dense-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-csv", default="outputs/ragas_results.csv")
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all four key configurations: paper/chunk × none/rocchio",
    )
    args = parser.parse_args()

    print("Loading retriever...")
    retriever = DenseRetriever(dense_model_name=args.dense_model)
    qa_gen = QAGenerator()
    evaluator = RagasEvaluator()

    query_rows = load_jsonl(Path(args.queries))
    if args.max_queries is not None:
        query_rows = query_rows[: args.max_queries]
    print(f"Loaded {len(query_rows)} queries from {args.queries}")

    if args.all_configs:
        configs = [
            ("paper", "none", 0),
            ("chunk", "none", 0),
            ("chunk", "rocchio", 1),
            ("chunk", "llm", 1),
        ]
    else:
        configs = [(args.context_mode, args.feedback_method, args.rounds)]

    needs_llm = any(m in ("llm", "combined") for _, m, _ in configs)
    llm_refiner = LLMRefinement() if needs_llm else None

    all_results = []
    for context_mode, feedback_method, rounds in configs:
        row_llm_refiner = (
            llm_refiner if feedback_method in ("llm", "combined") else None
        )
        row_result = run_config(
            retriever=retriever,
            qa_gen=qa_gen,
            evaluator=evaluator,
            query_rows=query_rows,
            feedback_method=feedback_method,
            rounds=rounds,
            context_mode=context_mode,
            top_k=args.top_k,
            chunk_top_m=args.chunk_top_m,
            chunk_token_budget=args.chunk_token_budget,
            llm_refiner=row_llm_refiner,
        )
        all_results.append(row_result)

    print(f"\n{'=' * 60}")
    print(f"  {'Config':<30} {'N':>4}  {'Faith':>7}  {'AnsRel':>7}")
    print(f"  {'-' * 50}")
    for r in all_results:
        faith = f"{r['faithfulness']:.3f}" if r["faithfulness"] is not None else "  N/A"
        arel = f"{r['answer_relevancy']:.3f}" if r["answer_relevancy"] is not None else "  N/A"
        print(f"  {r['config']:<30} {r['n']:>4}  {faith:>7}  {arel:>7}")
    print(f"{'=' * 60}")

    if args.output_csv and all_results:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["config", "n", "faithfulness", "answer_relevancy"],
            )
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
