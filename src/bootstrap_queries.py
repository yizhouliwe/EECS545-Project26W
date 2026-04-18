import argparse
from pathlib import Path

from src.part2_utils import load_config, load_papers, save_jsonl


TOPIC_TEMPLATES = [
    "retrieval augmented generation",
    "information retrieval ranking",
    "transformer based information retrieval",
    "large language models for question answering",
    "machine learning for scientific document retrieval",
    "computer vision foundation models",
    "multimodal retrieval",
    "reinforcement learning for language models",
    "diffusion models for image generation",
    "benchmarking reasoning in language models",
    "contrastive learning for representation learning",
    "document ranking with neural models",
]


def main():
    parser = argparse.ArgumentParser(description="Create a starter paper-level labeled query file")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="data/labeled_queries.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_path = Path(cfg["data_collection"]["output_path"]).parent / "arxiv_corpus_cleaned.jsonl"
    papers = load_papers(str(corpus_path))

    rows = []
    for idx, topic in enumerate(TOPIC_TEMPLATES, start=1):
        seed_titles = [paper["title"] for paper in papers if topic.split()[0] in paper["cleaned_text"]][:3]
        rows.append(
            {
                "query_id": f"q{idx:03d}",
                "query_text": topic,
                "relevant_arxiv_ids": [],
                "notes": "Fill in relevant_arxiv_ids after manual review.",
                "seed_titles": seed_titles,
            }
        )

    save_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} starter queries to {args.output}")
    print("Next step: open the file and fill in relevant_arxiv_ids manually.")


if __name__ == "__main__":
    main()
