"""
Qualitative feedback loop trace for a single query.

Outputs a markdown table showing top-K retrieved paper titles at each round
for Rocchio, LLM, and Combined feedback methods, with relevant papers marked.
Optionally generates a UMAP/PCA plot showing query vector movement.

Usage:
    python run_qualitative.py --query-id q098
    python run_qualitative.py --query-id q098 --umap --pca
"""

import argparse
from pathlib import Path

import numpy as np

from src.features.feature_representation import reduce_dimensions
from src.feedback.feedback_logic import apply_facet_weights, apply_rocchio
from src.feedback.llm_refinement import LLMRefinement
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.evaluate import ndcg_at_k
from src.utils.helpers import load_jsonl


def normalize_id(paper_id: str) -> str:
    pid = str(paper_id).lower().strip()
    if pid.startswith("arxiv:"):
        pid = pid.replace("arxiv:", "")
    if "v" in pid:
        pid = pid.split("v")[0]
    return pid


def run_method_trace(
    retriever: DenseRetriever,
    query_row: dict,
    method: str,
    rounds: int,
    top_k: int,
    llm_refiner: LLMRefinement | None = None,
) -> list[dict]:
    relevant_ids = {normalize_id(i) for i in query_row.get("relevant_arxiv_ids", [])}
    original_query = query_row["query_text"]
    current_query = original_query
    current_vector = retriever.encode_query(current_query)

    trace = []
    for round_idx in range(rounds + 1):
        results = retriever.retrieve_by_vector(current_vector, k=top_k)
        trace.append(
            {
                "round": round_idx,
                "query": current_query,
                "query_vector": current_vector.copy(),
                "papers": [
                    {
                        "title": r["title"],
                        "arxiv_id": r["arxiv_id"],
                        "paper_id": r["paper_id"],
                        "is_relevant": normalize_id(r["arxiv_id"]) in relevant_ids,
                    }
                    for r in results
                ],
            }
        )

        if round_idx >= rounds:
            break

        hits = [r for r in results if normalize_id(r["arxiv_id"]) in relevant_ids]
        if not hits:
            hits = [results[0]] if results else []
        if not hits:
            break

        positive_vectors = [retriever.get_embedding(r["paper_id"]) for r in hits]
        feedback_text = "These papers are relevant. Find more like: " + "; ".join(
            r["title"] for r in hits[:3]
        )

        if method == "rocchio":
            current_vector = apply_rocchio(current_vector, positive_vectors)
        elif method == "llm":
            refinement = llm_refiner.refine_query(
                original_query=original_query,
                current_query=current_query,
                retrieved_titles=[r["title"] for r in results],
                user_feedback_text=feedback_text,
            )
            current_query = refinement.rewritten_query
            current_vector = apply_facet_weights(
                retriever.encode_query(current_query), refinement.facet_weights
            )
        elif method == "combined":
            rocchio_vector = apply_rocchio(current_vector, positive_vectors)
            seed_results = retriever.retrieve_by_vector(rocchio_vector, k=top_k)
            refinement = llm_refiner.refine_query(
                original_query=original_query,
                current_query=current_query,
                retrieved_titles=[r["title"] for r in seed_results],
                user_feedback_text=feedback_text,
            )
            current_query = refinement.rewritten_query
            current_vector = apply_facet_weights(
                retriever.encode_query(current_query), refinement.facet_weights
            )

    return trace


METHOD_DISPLAY = {"rocchio": "Rocchio", "llm": "LLM", "combined": "Combined"}


def format_markdown(query_row: dict, method_traces: dict, top_k: int) -> str:
    relevant_ids = [normalize_id(i) for i in query_row.get("relevant_arxiv_ids", [])]
    n_relevant = len(relevant_ids)
    lines = [
        f"## Query {query_row['query_id']}",
        f"> {query_row['query_text']}",
        f"\n**Relevant papers in corpus:** {n_relevant}\n",
    ]

    for method, trace in method_traces.items():
        display_name = METHOD_DISPLAY.get(method, method.capitalize())
        ndcg_k = min(10, top_k)
        ndcg_progression = " → ".join(
            f"R{t['round']}={ndcg_at_k([normalize_id(p['arxiv_id']) for p in t['papers']], relevant_ids, ndcg_k):.3f}"
            for t in trace
        )
        lines.append(f"### {display_name} — NDCG@{ndcg_k}: {ndcg_progression}\n")

        header = "| Rank | " + " | ".join(f"Round {t['round']}" for t in trace) + " |"
        sep = "|:----:|" + "|".join(":---" for _ in trace) + "|"
        lines += [header, sep]

        for rank in range(top_k):
            row = f"| {rank + 1} |"
            for t in trace:
                if rank < len(t["papers"]):
                    p = t["papers"][rank]
                    marker = "✓ " if p["is_relevant"] else ""
                    title = p["title"][:65] + ("…" if len(p["title"]) > 65 else "")
                    row += f" {marker}{title} |"
                else:
                    row += " — |"
            lines.append(row)

        if method in ("llm", "combined"):
            rewritten = [
                t
                for t in trace
                if t["round"] > 0 and t["query"] != query_row["query_text"]
            ]
            if rewritten:
                lines.append("")
                lines.append("**Query evolution:**")
                for t in rewritten:
                    lines.append(f'- R{t["round"]}: *"{t["query"]}"*')

        lines.append("")

    return "\n".join(lines)


def generate_umap_plot(
    retriever: DenseRetriever,
    method_traces: dict,
    query_row: dict,
    output_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"UMAP plot skipped: {exc}")
        return

    relevant_ids = {normalize_id(i) for i in query_row.get("relevant_arxiv_ids", [])}

    # Sample papers + always include relevant ones
    rng = np.random.default_rng(42)
    n_sample = 3000
    all_paper_idx = np.arange(len(retriever.paper_ids))
    sampled_idx = rng.choice(
        all_paper_idx, size=min(n_sample, len(all_paper_idx)), replace=False
    )
    relevant_corpus_idx = [
        i
        for i, paper in enumerate(retriever.papers)
        if normalize_id(paper["arxiv_id"]) in relevant_ids
    ]
    all_idx = list(set(sampled_idx.tolist() + relevant_corpus_idx))
    is_relevant_mask = [
        normalize_id(retriever.papers[i]["arxiv_id"]) in relevant_ids for i in all_idx
    ]

    # Collect query vectors: one per round per method, in order
    all_query_vecs = []
    method_vec_slices = {}
    for method, trace in method_traces.items():
        start = len(all_query_vecs)
        for t in trace:
            all_query_vecs.append(t["query_vector"])
        method_vec_slices[method] = (start, len(all_query_vecs))

    paper_embeddings = retriever.dense_embeddings[all_idx]
    all_vectors = np.vstack(
        [paper_embeddings] + [v.reshape(1, -1) for v in all_query_vecs]
    )

    print("Fitting UMAP...")
    reduced = reduce_dimensions(
        all_vectors, method="umap", n_neighbors=15, min_dist=0.1, random_state=42
    )

    paper_2d = reduced[: len(all_idx)]
    query_2d = reduced[len(all_idx) :]

    method_colors = {"rocchio": "#2196F3", "llm": "#FF9800", "combined": "#4CAF50"}
    methods = list(method_traces.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # Background papers
        bg = [p for p, rel in zip(paper_2d, is_relevant_mask) if not rel]
        rel_pts = [p for p, rel in zip(paper_2d, is_relevant_mask) if rel]
        if bg:
            ax.scatter(*np.array(bg).T, s=2, alpha=0.25, color="lightgray")
        if rel_pts:
            ax.scatter(
                *np.array(rel_pts).T,
                s=50,
                alpha=0.9,
                color="gold",
                edgecolors="darkorange",
                linewidths=0.8,
                zorder=3,
                label="Relevant",
            )

        # Query vector path
        s, e = method_vec_slices[method]
        qvecs = query_2d[s:e]
        color = method_colors.get(method, "steelblue")
        ax.plot(qvecs[:, 0], qvecs[:, 1], color=color, linewidth=1.5, zorder=4)
        # R0=star (shared), R1=circle, R2=diamond
        round_markers = {
            0: ("*", 240, "white"),
            1: ("o", 140, color),
            2: ("D", 140, color),
        }
        rel_ids = [normalize_id(x) for x in query_row.get("relevant_arxiv_ids", [])]
        r0_plotted = False
        for i, qv in enumerate(qvecs):
            papers = method_traces[method][i]["papers"]
            ndcg = ndcg_at_k(
                [normalize_id(p["arxiv_id"]) for p in papers],
                rel_ids,
                min(10, len(papers)),
            )
            mk, sz, fc = round_markers.get(i, ("o", 140, color))
            if i == 0 and not r0_plotted:
                ax.scatter(
                    qv[0],
                    qv[1],
                    s=sz,
                    color=fc,
                    marker=mk,
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=6,
                    label=f"R0 — initial (NDCG={ndcg:.2f})",
                )
                r0_plotted = True
            elif i > 0:
                ax.scatter(
                    qv[0],
                    qv[1],
                    s=sz,
                    color=fc,
                    marker=mk,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=5,
                    label=f"R{i} (NDCG={ndcg:.2f})",
                )

        # Zoom into query path region
        pad = (
            max(
                (qvecs[:, 0].max() - qvecs[:, 0].min()),
                (qvecs[:, 1].max() - qvecs[:, 1].min()),
            )
            * 1.5
            + 0.5
        )
        ax.set_xlim(qvecs[:, 0].mean() - pad, qvecs[:, 0].mean() + pad)
        ax.set_ylim(qvecs[:, 1].mean() - pad, qvecs[:, 1].mean() + pad)
        ax.set_title(
            f"{METHOD_DISPLAY.get(method, method.capitalize())}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=8, loc="best", markerscale=0.6, labelspacing=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Query Vector Movement — {query_row['query_id']}\n"
        f"{query_row['query_text'][:80]}…",
        fontsize=11,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"UMAP saved to {output_path}")
    plt.close()


def generate_pca_plot(
    method_traces: dict,
    query_row: dict,
    output_path: str,
) -> None:
    """PCA projection of query vectors only — shows method divergence clearly."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError as exc:
        print(f"PCA plot skipped: {exc}")
        return

    relevant_ids = [normalize_id(i) for i in query_row.get("relevant_arxiv_ids", [])]
    all_vecs = []
    for trace in method_traces.values():
        for t in trace:
            all_vecs.append(t["query_vector"])
    all_vecs = np.array(all_vecs)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_vecs)

    method_colors = {"rocchio": "#2196F3", "llm": "#FF9800", "combined": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(7, 5))

    # R0 is identical for all methods — plot once as a shared marker
    r0_pt = projected[0]
    r0_trace = next(iter(method_traces.values()))[0]
    r0_ndcg = ndcg_at_k(
        [normalize_id(p["arxiv_id"]) for p in r0_trace["papers"]],
        relevant_ids,
        min(10, len(r0_trace["papers"])),
    )
    ax.scatter(
        r0_pt[0],
        r0_pt[1],
        s=200,
        color="white",
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=6,
        label=f"R0 — initial (NDCG={r0_ndcg:.2f})",
    )

    offset = 0
    for method, trace in method_traces.items():
        n = len(trace)
        pts = projected[offset : offset + n]
        color = method_colors.get(method, "steelblue")
        display = METHOD_DISPLAY.get(method, method)
        # Draw line from R0 through all rounds
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, zorder=3)
        # R1=circle, R2=diamond; skip R0 (already plotted as shared star)
        round_markers = {1: "o", 2: "D"}
        for i, (pt, t) in enumerate(zip(pts, trace)):
            if i == 0:
                continue
            ndcg = ndcg_at_k(
                [normalize_id(p["arxiv_id"]) for p in t["papers"]],
                relevant_ids,
                min(10, len(t["papers"])),
            )
            ax.scatter(
                pt[0],
                pt[1],
                s=160,
                color=color,
                marker=round_markers.get(i, "o"),
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
                label=f"{display} R{i} (NDCG={ndcg:.2f})",
            )
        offset += n

    ax.set_title(
        f"Query Vector Trajectory (PCA) — {query_row['query_id']}\n"
        f"{query_row['query_text'][:80]}…",
        fontsize=10,
    )
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.legend(fontsize=9, loc="lower right", markerscale=0.6, labelspacing=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"PCA plot saved to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative feedback loop trace")
    parser.add_argument("--queries", default="data/queries_test.jsonl")
    parser.add_argument("--query-id", default="q098")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=["rocchio", "llm", "combined"])
    parser.add_argument(
        "--dense-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--output-md", default="outputs/qualitative_trace.md")
    parser.add_argument("--skip-md", action="store_true", help="Skip markdown output")
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Generate UMAP plot (paper cloud + query path)",
    )
    parser.add_argument("--output-umap", default="outputs/qualitative_umap.png")
    parser.add_argument(
        "--pca", action="store_true", help="Generate PCA query trajectory plot"
    )
    parser.add_argument("--output-pca", default="outputs/qualitative_pca.png")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.queries))
    query_row = next((r for r in rows if r["query_id"] == args.query_id), None)
    if query_row is None:
        print(f"Query ID '{args.query_id}' not found in {args.queries}")
        return

    print(f"Query: {query_row['query_id']} — {query_row['query_text'][:80]}...")
    print(f"Relevant papers: {len(query_row.get('relevant_arxiv_ids', []))}\n")

    retriever = DenseRetriever(dense_model_name=args.dense_model)
    needs_llm = any(m in ("llm", "combined") for m in args.methods)
    llm_refiner = LLMRefinement() if needs_llm else None

    method_traces = {}
    for method in args.methods:
        print(f"Running {method}...")
        method_traces[method] = run_method_trace(
            retriever, query_row, method, args.rounds, args.top_k, llm_refiner
        )

    if not args.skip_md:
        md = format_markdown(query_row, method_traces, args.top_k)
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md)
        print(f"\nMarkdown saved to {args.output_md}")
        print("\n" + md)

    if args.umap:
        generate_umap_plot(retriever, method_traces, query_row, args.output_umap)

    if args.pca:
        generate_pca_plot(method_traces, query_row, args.output_pca)


if __name__ == "__main__":
    main()
