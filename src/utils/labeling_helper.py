import argparse
import textwrap

from src.retrieval.retrieval import PaperRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Show candidate papers to help you fill relevant_paper_ids manually"
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mode", choices=["tfidf", "dense", "hybrid"], default="tfidf")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--snippet-chars", type=int, default=240)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    retriever = PaperRetriever(config_path=args.config)
    if args.mode == "tfidf":
        results = retriever.retrieve_tfidf(args.query, k=args.k)
    elif args.mode == "dense":
        results = retriever.retrieve_dense(args.query, k=args.k)
    else:
        results = retriever.retrieve_hybrid(
            args.query,
            k=args.k,
            alpha=args.alpha,
            dense_candidates=args.dense_candidates,
        )

    print(f"\nLabeling candidates for query: {args.query}\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank:>2}. arxiv_id: {result['arxiv_id']}")
        print(f"    paper_id: {result['paper_id']}")
        print(f"    title: {result['title']}")
        print(f"    score: {result['score']:.4f}")
        snippet = result["abstract"][: args.snippet_chars].replace("\n", " ").strip()
        if len(result["abstract"]) > args.snippet_chars:
            snippet += "..."
        print(f"    abstract: {textwrap.fill(snippet, width=88, subsequent_indent='              ')}")
        print()


if __name__ == "__main__":
    main()
