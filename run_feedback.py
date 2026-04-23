import argparse
import collections
import csv
from pathlib import Path

from src.feedback.feedback_logic import apply_facet_weights, apply_rocchio
from src.feedback.llm_refinement import LLMRefinement
from src.utils.helpers import load_jsonl
from src.rag.qa_engine import QAGenerator
from src.retrieval.evaluate import average_precision, ndcg_at_k, precision_at_k, recall_at_k
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


def select_papers_for_budget(results: list[dict], token_budget: int) -> list[dict]:
    selected = []
    used_tokens = 0
    for result in results:
        paper_tokens = int(result.get("tokens_est", len(result.get("abstract", "").split())))
        if selected and used_tokens + paper_tokens > token_budget:
            continue
        selected.append(result)
        used_tokens += paper_tokens
        if used_tokens >= token_budget:
            break
    return selected


def build_rag_results(
    args,
    retriever: PaperRetrieverExtended,
    current_vector,
    final_results: list[dict],
) -> list[dict]:
    if args.context_mode == "chunk":
        return retriever.retrieve_chunks(
            query_vector=current_vector,
            paper_ids=[result["paper_id"] for result in final_results],
            top_m=args.chunk_top_m,
            token_budget=args.chunk_token_budget,
        )
    return select_papers_for_budget(final_results, token_budget=args.chunk_token_budget)


def evaluate_context_results(
    rag_results: list[dict],
    relevant_ids_norm: set[str],
) -> dict:
    if not rag_results:
        return {
            "context_items": 0,
            "context_unique_papers": 0,
            "context_tokens": 0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    evidence_hits = 0
    unique_paper_ids = []
    seen = set()
    for item in rag_results:
        norm_id = normalize_id(item["arxiv_id"])
        if norm_id in relevant_ids_norm:
            evidence_hits += 1
        if norm_id not in seen:
            seen.add(norm_id)
            unique_paper_ids.append(norm_id)

    unique_relevant_hits = sum(1 for paper_id in unique_paper_ids if paper_id in relevant_ids_norm)
    total_tokens = sum(int(item.get("tokens_est", 0)) for item in rag_results)
    return {
        "context_items": len(rag_results),
        "context_unique_papers": len(unique_paper_ids),
        "context_tokens": total_tokens,
        "context_precision": evidence_hits / len(rag_results),
        "context_recall": (
            unique_relevant_hits / len(relevant_ids_norm) if relevant_ids_norm else 0.0
        ),
    }


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
    parser.add_argument(
        "--context-mode",
        choices=["paper", "chunk"],
        default="paper",
        help="Use full paper abstracts or selected chunks as the final RAG context.",
    )
    parser.add_argument(
        "--chunk-top-m",
        type=int,
        default=8,
        help="Maximum number of chunks to pass to RAG when --context-mode chunk.",
    )
    parser.add_argument(
        "--chunk-token-budget",
        type=int,
        default=3000,
        help="Approximate total token budget for the final RAG context.",
    )
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-csv", default=None, help="Path to write per-round metrics CSV")
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

    records = []

    for row in query_rows:
        query_text = row["query_text"]
        relevant_ids_norm = {normalize_id(idx) for idx in row.get("relevant_arxiv_ids", [])}
        current_query = query_text
        current_vector = retriever.encode_query(current_query)
        final_results = []

        print(f"\n{'=' * 80}")
        print(f"QUERY: {query_text}")
        print(f"{'=' * 80}")

        for round_idx in range(args.rounds + 1):
            results = retriever.retrieve_by_vector(current_vector, k=args.top_k)
            final_results = results
            predicted_ids = [normalize_id(r["arxiv_id"]) for r in results]
            relevant_ids_list = list(relevant_ids_norm)
            hits = [r for r in results if normalize_id(r["arxiv_id"]) in relevant_ids_norm]

            p = precision_at_k(predicted_ids, relevant_ids_list, args.top_k)
            r = recall_at_k(predicted_ids, relevant_ids_list, args.top_k)
            n = ndcg_at_k(predicted_ids, relevant_ids_list, min(10, args.top_k))
            m = average_precision(predicted_ids, relevant_ids_list)

            print(
                f"Round {round_idx} | "
                f"P@{args.top_k}: {p:.3f} | "
                f"R@{args.top_k}: {r:.3f} | "
                f"NDCG@10: {n:.3f} | "
                f"MAP: {m:.3f} | "
                f"Hits: {len(hits)}"
            )
            records.append({
                "query_id": row.get("query_id", query_text[:40]),
                "round": round_idx,
                "precision": p,
                "recall": r,
                "ndcg": n,
                "map": m,
                "context_items": None,
                "context_unique_papers": None,
                "context_tokens": None,
            })

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

        rag_results = build_rag_results(
            args=args,
            retriever=retriever,
            current_vector=current_vector,
            final_results=final_results,
        )
        context_metrics = evaluate_context_results(rag_results, relevant_ids_norm)
        records.append({
            "query_id": row.get("query_id", query_text[:40]),
            "round": "context",
            "precision": context_metrics["context_precision"],
            "recall": context_metrics["context_recall"],
            "ndcg": None,
            "map": None,
            "context_items": context_metrics["context_items"],
            "context_unique_papers": context_metrics["context_unique_papers"],
            "context_tokens": context_metrics["context_tokens"],
        })
        print(
            "[Context] "
            f"items={context_metrics['context_items']} | "
            f"unique_papers={context_metrics['context_unique_papers']} | "
            f"tokens~{context_metrics['context_tokens']} | "
            f"precision={context_metrics['context_precision']:.3f} | "
            f"recall={context_metrics['context_recall']:.3f}"
        )

        if qa_gen is None:
            continue

        print("\n[RAG] Generating answer via UM GPT-oss-120B...")
        if args.context_mode == "chunk":
            print(
                f"[RAG] Selected {len(rag_results)} chunks "
                f"(budget={args.chunk_token_budget}, top_m={args.chunk_top_m})."
            )
        else:
            print(
                f"[RAG] Selected {len(rag_results)} papers "
                f"(budget={args.chunk_token_budget})."
            )
        context = qa_gen.format_context(rag_results)
        try:
            answer = qa_gen.generate_answer(current_query, context)
            print(f"\n[AI Answer]:\n{answer}\n")
        except Exception as exc:
            print(f"\n[RAG] Skipped: {exc}\n")

    by_round = collections.defaultdict(lambda: collections.defaultdict(list))
    context_records = [rec for rec in records if rec["round"] == "context"]
    metric_records = [rec for rec in records if rec["round"] != "context"]
    for rec in metric_records:
        for metric in ("precision", "recall", "ndcg", "map"):
            by_round[rec["round"]][metric].append(rec[metric])

    if by_round:
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY (feedback_method={args.feedback_method})")
        print(f"  {'Round':<8} {'P@K':>8} {'R@K':>8} {'NDCG@10':>10} {'MAP':>8}")
        print(f"  {'-'*44}")
        for rnd in sorted(by_round):
            vals = by_round[rnd]
            print(
                f"  {rnd:<8} "
                f"{sum(vals['precision'])/len(vals['precision']):>8.3f} "
                f"{sum(vals['recall'])/len(vals['recall']):>8.3f} "
                f"{sum(vals['ndcg'])/len(vals['ndcg']):>10.3f} "
                f"{sum(vals['map'])/len(vals['map']):>8.3f}"
            )
        print(f"{'=' * 70}\n")

    if context_records:
        print(f"\n{'=' * 70}")
        print(f"  CONTEXT SUMMARY ({args.context_mode})")
        print(f"  {'Items':>8} {'Unique':>8} {'Tokens':>10} {'Prec':>8} {'Recall':>8}")
        print(f"  {'-'*46}")
        print(
            f"  {sum(rec['context_items'] for rec in context_records)/len(context_records):>8.2f} "
            f"{sum(rec['context_unique_papers'] for rec in context_records)/len(context_records):>8.2f} "
            f"{sum(rec['context_tokens'] for rec in context_records)/len(context_records):>10.1f} "
            f"{sum(rec['precision'] for rec in context_records)/len(context_records):>8.3f} "
            f"{sum(rec['recall'] for rec in context_records)/len(context_records):>8.3f}"
        )
        print(f"{'=' * 70}\n")

    if args.output_csv and records:
        csv_path = Path(args.output_csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_id",
                    "round",
                    "precision",
                    "recall",
                    "ndcg",
                    "map",
                    "context_items",
                    "context_unique_papers",
                    "context_tokens",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
        print(f"Metrics written to {csv_path}")


if __name__ == "__main__":
    main()
