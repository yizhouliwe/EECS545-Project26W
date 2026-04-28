"""
Interactive feedback loop CLI.

Retrieves papers for a query, lets the user mark relevant results,
applies a feedback method, and repeats for the requested number of rounds.

Usage:
    python run_interactive.py
    python run_interactive.py --method rocchio --top-k 10 --rounds 3
    python run_interactive.py --method llm
    python run_interactive.py --method combined
"""

import argparse
import sys
from pathlib import Path

from src.feedback.feedback_logic import apply_facet_weights, apply_rocchio
from src.feedback.llm_refinement import LLMRefinement
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.helpers import dense_embedding_filename


def check_artifacts(dense_model: str) -> None:
    data_dir = Path("data")
    # Only the .npy embeddings are not tracked in git; all .jsonl and meta .json
    # files are committed and present after a fresh clone.
    required = [
        (data_dir / dense_embedding_filename(dense_model), "dense embeddings (.npy)"),
    ]

    missing = []
    for path, label in required:
        if not path.exists():
            missing.append(f"  • {path}  ({label})")
    if not missing:
        return

    print("Missing required artifacts:")
    print("\n".join(missing))
    print()
    print(
        "The corpus is already included in the repo, only embeddings need to be generated."
    )
    choice = input("Generate embeddings now? [y/n]: ").strip().lower()

    if choice != "y":
        print("Exiting.")
        sys.exit(0)

    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "src.features.feature_representation",
        "--config",
        "configs/config.yaml",
        "--dense-model",
        dense_model,
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Embedding generation failed. Fix the errors above and try again.")
        sys.exit(1)


def display_results(results: list[dict], round_idx: int) -> None:
    print(f"\n{'─' * 70}")
    print(f"  Round {round_idx} — Top {len(results)} results")
    print(f"{'─' * 70}")
    for i, r in enumerate(results, 1):
        print(f"  [{i:2d}]  {r['title']}")
        arxiv_id = r.get("arxiv_id", "")
        if arxiv_id:
            print(f"        https://arxiv.org/abs/{arxiv_id}")
        abstract = r.get("abstract", "").replace("\n", " ").strip()
        if abstract:
            snippet = abstract[:160] + ("…" if len(abstract) > 160 else "")
            print(f"        {snippet}")
        print()
    print(f"{'─' * 70}")


def parse_selections(raw: str, max_idx: int) -> list[int]:
    selections = []
    for token in raw.strip().split():
        try:
            n = int(token)
            if 1 <= n <= max_idx:
                selections.append(n - 1)
        except ValueError:
            pass
    return selections


def run_interactive(
    retriever: DenseRetriever,
    query: str,
    method: str,
    rounds: int,
    top_k: int,
    llm_refiner: LLMRefinement | None,
) -> None:
    current_query = query
    current_vector = retriever.encode_query(current_query)

    for round_idx in range(rounds + 1):
        results = retriever.retrieve_by_vector(current_vector, k=top_k)
        display_results(results, round_idx)

        if round_idx >= rounds:
            print("\nMax rounds reached. Done.")
            break

        raw = input(
            "\nMark relevant papers (e.g. '1 3 5'), or Enter to skip, 'q' to quit: "
        ).strip()
        if raw.lower() == "q":
            print("Exiting.")
            break

        selected_idx = parse_selections(raw, len(results))
        if not selected_idx:
            # pseudo-relevance: treat top result as relevant
            selected_idx = [0]
            print("  No selection — using top result as pseudo-relevant.")

        hits = [results[i] for i in selected_idx]
        positive_vectors = [retriever.get_embedding(r["paper_id"]) for r in hits]
        feedback_text = "These papers are relevant. Find more like: " + "; ".join(
            r["title"] for r in hits[:3]
        )

        print(f"\n  Applying {method} feedback on {len(hits)} paper(s)...")

        if method == "rocchio":
            current_vector = apply_rocchio(current_vector, positive_vectors)

        elif method == "llm":
            refinement = llm_refiner.refine_query(
                original_query=query,
                current_query=current_query,
                retrieved_titles=[r["title"] for r in results],
                user_feedback_text=feedback_text,
            )
            current_query = refinement.rewritten_query
            current_vector = apply_facet_weights(
                retriever.encode_query(current_query), refinement.facet_weights
            )
            print(f'  Rewritten query: "{current_query}"')

        elif method == "combined":
            rocchio_vector = apply_rocchio(current_vector, positive_vectors)
            seed_results = retriever.retrieve_by_vector(rocchio_vector, k=top_k)
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
            print(f'  Rewritten query: "{current_query}"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive relevance feedback loop")
    parser.add_argument(
        "--method", choices=["rocchio", "llm", "combined"], default="rocchio"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--dense-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parser.parse_args()
    check_artifacts(args.dense_model)

    print(f"\nLoading retriever ({args.dense_model})...")
    retriever = DenseRetriever(dense_model_name=args.dense_model)
    llm_refiner = LLMRefinement() if args.method in ("llm", "combined") else None

    print(
        f"Method: {args.method}  |  Top-K: {args.top_k}  |  Max rounds: {args.rounds}"
    )
    query = input("\nEnter your query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        return

    run_interactive(retriever, query, args.method, args.rounds, args.top_k, llm_refiner)


if __name__ == "__main__":
    main()
