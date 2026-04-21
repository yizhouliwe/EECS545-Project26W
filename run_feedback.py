import argparse
from pathlib import Path

from src.feedback.feedback_logic import apply_facet_weights, apply_rocchio
from src.feedback.llm_refinement import LLMRefinement
from src.utils.helpers import load_jsonl
from src.rag.qa_engine import QAGenerator
from src.retrieval.retrieval_extended import PaperRetrieverExtended


def normalize_id(paper_id: str) -> str:
    pid = str(paper_id).lower().strip()
    if pid.startswith("arxiv:"):
        pid = pid.replace("arxiv:", "")
    if "v" in pid:
        pid = pid.split("v")[0]
    return pid


def build_feedback(
    row: dict,
    results: list[dict],
    hits: list[dict],
    use_pseudo: bool,
) -> tuple[list[dict], str, bool]:
    if hits:
        explicit_feedback = row.get("user_feedback_text") or row.get("feedback_text")
        if explicit_feedback:
            return hits, str(explicit_feedback), False

        hit_titles = "; ".join(result["title"] for result in hits[:3])
        feedback = (
            "The user marked these retrieved papers as relevant. "
            f"Retrieve more work like: {hit_titles}."
        )
        return hits, feedback, False

    if use_pseudo and results:
        return [results[0]], "this is relevant, find more like it", True

    return [], "", False


def apply_feedback_method(
    feedback_method: str,
    retriever: PaperRetrieverExtended,
    llm_refiner: LLMRefinement | None,
    original_query: str,
    current_query: str,
    current_vector,
    results: list[dict],
    feedback_results: list[dict],
    feedback_text: str,
):
    positive_vectors = [
        retriever.get_embedding(result["paper_id"]) for result in feedback_results
    ]

    if feedback_method == "rocchio":
        return current_query, apply_rocchio(current_vector, positive_vectors), None

    if llm_refiner is None:
        raise RuntimeError(
            "LLM-based feedback requires an initialized LLMRefinement instance."
        )

    if feedback_method == "combined":
        rocchio_vector = apply_rocchio(current_vector, positive_vectors)
        llm_seed_results = retriever.retrieve_by_vector(rocchio_vector, k=len(results))
    else:
        llm_seed_results = results

    refinement = llm_refiner.refine_query(
        original_query=original_query,
        current_query=current_query,
        retrieved_titles=[result["title"] for result in llm_seed_results],
        user_feedback_text=feedback_text,
    )
    refined_vector = retriever.encode_query(refinement.rewritten_query)
    refined_vector = apply_facet_weights(refined_vector, refinement.facet_weights)
    return refinement.rewritten_query, refined_vector, refinement


def main():
    parser = argparse.ArgumentParser(description="Part 3: Relevance Feedback & RAG")
    parser.add_argument("--queries", default="data/queries_val.jsonl")
    parser.add_argument("--rounds", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--use_pseudo",
        action="store_true",
        help="Enable PRF if no ground truth hit is found",
    )
    parser.add_argument(
        "--feedback-method",
        choices=["rocchio", "llm", "combined"],
        default="rocchio",
    )
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    retriever = PaperRetrieverExtended(dense_model_name=args.dense_model)
    llm_refiner = (
        LLMRefinement() if args.feedback_method in {"llm", "combined"} else None
    )
    qa_gen = None if args.skip_rag else QAGenerator()

    query_rows = load_jsonl(Path(args.queries))
    if args.max_queries is not None:
        query_rows = query_rows[: args.max_queries]

    print(
        f"Executing Part 3 evaluation on {len(query_rows)} queries "
        f"(feedback_method={args.feedback_method}, rounds={args.rounds})..."
    )

    for row in query_rows:
        query_text = row["query_text"]
        relevant_ids = {normalize_id(idx) for idx in row.get("relevant_arxiv_ids", [])}
        current_query = query_text
        current_vector = retriever.encode_query(current_query)
        final_results = []

        print(f"\n{'=' * 80}")
        print(f"QUERY: {query_text}")
        print(f"{'=' * 80}")

        for round_idx in range(args.rounds + 1):
            results = retriever.retrieve_by_vector(current_vector, k=args.top_k)
            final_results = results
            hits = [
                result
                for result in results
                if normalize_id(result["arxiv_id"]) in relevant_ids
            ]
            precision = len(hits) / max(args.top_k, 1)
            print(
                f"Round {round_idx} | Precision@{args.top_k}: {precision:.2f} | Hits: {len(hits)}"
            )

            if round_idx >= args.rounds:
                continue

            feedback_results, feedback_text, pseudo_used = build_feedback(
                row=row,
                results=results,
                hits=hits,
                use_pseudo=args.use_pseudo,
            )
            if not feedback_results:
                print(
                    "  --> No positive feedback available; stopping refinement early."
                )
                break

            if pseudo_used:
                print("  --> Applying pseudo-feedback using the top-ranked document.")

            try:
                current_query, current_vector, refinement = apply_feedback_method(
                    feedback_method=args.feedback_method,
                    retriever=retriever,
                    llm_refiner=llm_refiner,
                    original_query=query_text,
                    current_query=current_query,
                    current_vector=current_vector,
                    results=results,
                    feedback_results=feedback_results,
                    feedback_text=feedback_text,
                )
            except Exception as exc:
                print(f"  --> Feedback refinement failed: {exc}")
                break

            if refinement is not None:
                print(f"  --> Rewritten query: {refinement.rewritten_query}")
                print(f"  --> Facet weights: {refinement.facet_weights}")
                if refinement.explanation:
                    print(f"  --> LLM rationale: {refinement.explanation}")
            else:
                print(
                    f"  --> Rocchio updated the query vector using {len(feedback_results)} positives."
                )

        if qa_gen is None:
            continue

        print("\n[RAG] Generating answer via UM GPT-oss-120B...")
        context = qa_gen.format_context(final_results)
        try:
            answer = qa_gen.generate_answer(current_query, context)
            print(f"\n[AI Answer]:\n{answer}\n")
        except Exception as exc:
            print(f"\n[RAG] Skipped: {exc}\n")


if __name__ == "__main__":
    main()
