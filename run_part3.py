import argparse
from pathlib import Path

from src.feedback_logic import apply_rocchio
from src.part2_utils import load_jsonl
from src.qa_engine import QAGenerator
from src.retrieval_extended import ChunkRetriever, PaperRetrieverExtended


def normalize_id(paper_id: str) -> str:
    """Standardize ArXiv IDs by removing prefix and version suffixes."""
    pid = str(paper_id).lower().strip()
    if pid.startswith("arxiv:"):
        pid = pid.replace("arxiv:", "")
    if "v" in pid:
        pid = pid.split("v")[0]
    return pid


def run_feedback_rounds(
    retriever: PaperRetrieverExtended,
    query_text: str,
    relevant_ids: set[str],
    top_k: int,
    rounds: int,
    use_pseudo: bool,
    retrieval_mode: str,
    hybrid_alpha: float,
    dense_candidates: int,
):
    current_vec = retriever.encode_query(query_text) if retrieval_mode in {"dense", "hybrid"} else None
    final_results = []

    for r in range(rounds + 1):
        if retrieval_mode == "tfidf":
            results = retriever.retrieve_tfidf(query_text, k=top_k)
        elif retrieval_mode == "dense":
            results = retriever.retrieve_by_vector(current_vec, k=top_k)
        elif retrieval_mode == "hybrid":
            results = retriever.retrieve_hybrid(
                query_text,
                k=top_k,
                alpha=hybrid_alpha,
                dense_candidates=dense_candidates,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
        final_results = results

        hits = [res for res in results if normalize_id(res["arxiv_id"]) in relevant_ids]
        precision = len(hits) / top_k
        print(f"Round {r} | {retrieval_mode} Precision@{top_k}: {precision:.2f}")

        if r < rounds:
            if retrieval_mode != "dense":
                continue
            if hits:
                pos_vecs = [
                    retriever.dense_embeddings[retriever.paper_ids.index(hit["paper_id"])]
                    for hit in hits
                ]
                current_vec = apply_rocchio(current_vec, pos_vecs)
                print(f"  --> Vector updated using {len(hits)} relevant papers.")
            elif use_pseudo and results:
                print("  --> Applying Pseudo-Feedback using the top-ranked document.")
                top_1_id = results[0]["paper_id"]
                top_1_vec = retriever.dense_embeddings[retriever.paper_ids.index(top_1_id)]
                current_vec = apply_rocchio(current_vec, [top_1_vec])

    if current_vec is None:
        current_vec = retriever.encode_query(query_text)

    return current_vec, final_results


def build_context_results(
    context_mode: str,
    paper_results,
    chunk_retriever: ChunkRetriever,
    query_text: str,
    query_vector,
    token_budget: int,
    max_chunks: int | None,
    chunk_score_mode: str,
):
    if context_mode == "paper":
        return paper_results

    paper_ids = [result["paper_id"] for result in paper_results]
    return chunk_retriever.select_chunks_for_papers(
        paper_ids=paper_ids,
        query_text=query_text,
        query_vector=query_vector,
        token_budget=token_budget,
        max_chunks=max_chunks,
        score_mode=chunk_score_mode,
    )


def describe_context(context_mode: str, context_results) -> str:
    if context_mode == "paper":
        return f"{len(context_results)} papers"

    total_tokens = sum(int(item.get("tokens_est", 0)) for item in context_results)
    return f"{len(context_results)} chunks / ~{total_tokens} tokens"


def main():
    parser = argparse.ArgumentParser(description="Part 3: Relevance Feedback & RAG")
    parser.add_argument("--queries", default="data/queries_val.jsonl")
    parser.add_argument("--rounds", type=int, default=1, help="Number of feedback iterations")
    parser.add_argument("--top_k", type=int, default=5, help="Number of papers to retrieve")
    parser.add_argument("--use_pseudo", action="store_true", help="Enable PRF if no ground truth is found")
    parser.add_argument(
        "--retrieval-mode",
        choices=["tfidf", "dense", "hybrid"],
        default="tfidf",
        help="Paper-level retrieval mode used before chunk selection",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Interpolation weight for hybrid retrieval",
    )
    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=100,
        help="Dense candidate pool size for hybrid retrieval",
    )
    parser.add_argument(
        "--context-mode",
        choices=["paper", "chunk"],
        default="paper",
        help="Use paper abstracts or chunk-level evidence for the final RAG context",
    )
    parser.add_argument(
        "--compare-context-modes",
        action="store_true",
        help="Run both paper and chunk context generation for each query",
    )
    parser.add_argument(
        "--chunk-token-budget",
        type=int,
        default=3000,
        help="Maximum estimated tokens to include in chunk context mode",
    )
    parser.add_argument(
        "--chunk-max-results",
        type=int,
        default=None,
        help="Optional cap on the number of selected chunks",
    )
    parser.add_argument(
        "--chunk-score-mode",
        choices=["dense", "tfidf"],
        default="dense",
        help="How to score candidate chunks within the retrieved papers",
    )
    args = parser.parse_args()

    retriever = PaperRetrieverExtended()
    chunk_retriever = ChunkRetriever(paper_retriever=retriever)
    qa_gen = QAGenerator()
    query_rows = load_jsonl(Path(args.queries))
    context_modes = ["paper", "chunk"] if args.compare_context_modes else [args.context_mode]

    print(f"Executing Part 3 evaluation on {len(query_rows)} queries...")

    for row in query_rows:
        query_text = row["query_text"]
        relevant_ids = {normalize_id(idx) for idx in row.get("relevant_arxiv_ids", [])}

        print(f"\n{'=' * 80}")
        print(f"QUERY: {query_text}")
        print(f"{'=' * 80}")

        final_query_vector, final_results = run_feedback_rounds(
            retriever=retriever,
            query_text=query_text,
            relevant_ids=relevant_ids,
            top_k=args.top_k,
            rounds=args.rounds,
            use_pseudo=args.use_pseudo,
            retrieval_mode=args.retrieval_mode,
            hybrid_alpha=args.hybrid_alpha,
            dense_candidates=args.dense_candidates,
        )

        for context_mode in context_modes:
            context_results = build_context_results(
                context_mode=context_mode,
                paper_results=final_results,
                chunk_retriever=chunk_retriever,
                query_text=query_text,
                query_vector=final_query_vector,
                token_budget=args.chunk_token_budget,
                max_chunks=args.chunk_max_results,
                chunk_score_mode=args.chunk_score_mode,
            )

            print(
                f"\n[RAG] Generating answer via UM GPT-oss-120B "
                f"with {context_mode}-level context ({describe_context(context_mode, context_results)})..."
            )
            context = qa_gen.format_context(context_results)
            answer = qa_gen.generate_answer(query_text, context)
            print(f"\n[AI Answer | {context_mode}]:\n{answer}\n")


if __name__ == "__main__":
    main()
