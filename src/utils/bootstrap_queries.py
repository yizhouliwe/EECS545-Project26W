import argparse
from pathlib import Path

from src.utils.helpers import load_config, load_papers, save_jsonl


TOPIC_TEMPLATES = [
    # Retrieval & RAG
    "dense retrieval methods for low-resource scientific domains that combine contrastive pre-training with sparse lexical signals and evaluate on biomedical benchmarks",
    "retrieval-augmented generation systems that address hallucination in long-form scientific question answering using iterative retrieval and self-consistency verification",
    "hybrid sparse-dense retrieval architectures that dynamically weight lexical and semantic signals based on query complexity for open-domain question answering",
    "passage reranking approaches that leverage cross-encoder models trained on scientific literature with domain-adaptive fine-tuning strategies",
    "zero-shot dense retrieval methods that generalize across domains without task-specific training data using instruction-tuned language models",
    # LLMs & Reasoning
    "large language models that improve multi-step reasoning through chain-of-thought prompting combined with external tool use and iterative self-refinement",
    "methods for reducing hallucination in large language models during knowledge-intensive tasks by grounding generation in retrieved evidence",
    "instruction-following language models that decompose complex research queries into sub-questions and synthesize answers across multiple retrieved documents",
    "benchmarking compositional reasoning capabilities of large language models on tasks requiring integration of multiple scientific concepts",
    "parameter-efficient fine-tuning strategies for adapting pre-trained language models to scientific document understanding with limited labeled data",
    # Representation Learning
    "contrastive learning frameworks for scientific document embeddings that leverage citation graphs and co-authorship signals as supervision",
    "pre-trained language models specialized for scientific text that outperform general-domain models on information extraction and entity recognition tasks",
    "unsupervised domain adaptation methods for dense retrieval that align embedding spaces across scientific subfields without parallel corpora",
    "sentence embedding models that capture fine-grained semantic similarity between scientific claims and distinguish between supporting and contradicting evidence",
    "multi-vector document representations that encode different aspects of scientific papers separately for more precise retrieval",
    # Multimodal & Cross-modal
    "multimodal retrieval systems that jointly encode figures, tables, and text from scientific papers for cross-modal query answering",
    "vision-language models that retrieve relevant figures from scientific literature given natural language descriptions of experimental results",
    "cross-lingual scientific document retrieval methods that align multilingual embeddings without parallel training data",
    # Evaluation & Benchmarks
    "evaluation frameworks for open-domain scientific question answering that measure faithfulness, coverage, and citation accuracy of generated answers",
    "benchmark datasets for information retrieval that reflect realistic researcher behavior with complex multi-constraint queries and annotated relevance judgments",
    "methods for constructing hard negative examples in information retrieval training that improve model discrimination between semantically similar but non-relevant documents",
    # Neural IR & Ranking
    "neural ranking models that incorporate document structure and section-level signals for improving retrieval of long scientific documents",
    "query expansion techniques using large language models that generate contextually relevant pseudo-documents to improve sparse retrieval recall",
    "knowledge-grounded neural retrieval models that integrate structured knowledge graphs with unstructured text representations for scientific search",
    "learning-to-rank approaches that combine multiple retrieval signals including citation count, recency, and semantic similarity for scientific paper search",
    # Feedback & Interactive Search
    "interactive query refinement systems that incorporate explicit and implicit user feedback to iteratively improve retrieval results in academic search",
    "pseudo-relevance feedback methods that use top-ranked documents to expand query representations in both sparse and dense retrieval settings",
    "reinforcement learning approaches for optimizing multi-turn information seeking dialogues in scientific question answering systems",
    # Efficiency & Scalability
    "approximate nearest neighbor search methods that maintain retrieval quality while scaling to hundreds of millions of scientific document embeddings",
    "knowledge distillation techniques for compressing large bi-encoder retrieval models while preserving ranking performance on scientific benchmarks",
    "index compression and quantization strategies for dense retrieval systems that reduce memory footprint without significant recall degradation",
]


def main():
    parser = argparse.ArgumentParser(
        description="Create a starter paper-level labeled query file"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="data/labeled_queries.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_path = (
        Path(cfg["data_collection"]["output_path"]).parent
        / "arxiv_corpus_cleaned.jsonl"
    )
    papers = load_papers(str(corpus_path))

    rows = []
    for idx, topic in enumerate(TOPIC_TEMPLATES, start=1):
        seed_titles = [
            paper["title"]
            for paper in papers
            if topic.split()[0] in paper["cleaned_text"]
        ][:3]
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
