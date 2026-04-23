from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.features.dense_encoder import DenseEncoder
from src.utils.helpers import chunk_embedding_filename, load_config, load_jsonl


def build_chunk_input_texts(chunks: list[dict], paper_lookup: dict[str, dict], model_name: str) -> list[str]:
    texts = []
    for chunk in chunks:
        paper = paper_lookup.get(chunk["paper_id"], {})
        title = paper.get("title", "").strip()
        chunk_text = chunk["chunk_text"].strip()
        if model_name == "allenai/specter2_base":
            texts.append(f"{title} [SEP] {chunk_text}" if title else chunk_text)
        else:
            texts.append(f"{title}. {chunk_text}" if title else chunk_text)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Build dense embeddings for abstract chunks")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["data_collection"]["output_path"]).parent
    dense_model = args.dense_model or cfg["embeddings"]["dense_model"]
    batch_size = args.batch_size or cfg["embeddings"]["dense_batch_size"]
    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = None

    chunks = load_jsonl(data_dir / "arxiv_chunks.jsonl")
    papers = load_jsonl(data_dir / "arxiv_corpus_cleaned.jsonl")
    paper_lookup = {paper["paper_id"]: paper for paper in papers}

    encoder = DenseEncoder(
        model_name=dense_model,
        max_length=cfg["embeddings"]["dense_max_length"],
        device=device,
    )
    texts = build_chunk_input_texts(chunks, paper_lookup, dense_model)
    embeddings = encoder.encode_documents(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    output_path = data_dir / chunk_embedding_filename(dense_model)
    np.save(output_path, embeddings.astype(np.float32))
    print(f"Saved chunk embeddings to {output_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
