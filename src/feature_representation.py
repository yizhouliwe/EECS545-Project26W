import argparse
from datetime import datetime, timezone
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.dense_encoder import DenseEncoder
from src.part2_utils import (
    dense_embedding_filename,
    dense_embedding_metadata_filename,
    dense_faiss_index_filename,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def build_tfidf_features(
    cleaned_texts: List[str],
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[object, object]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger.info(f"Building TF-IDF features (max_features={max_features}, ngram={ngram_range})")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}, "
                f"vocab size: {len(vectorizer.vocabulary_)}, "
                f"nnz: {tfidf_matrix.nnz}")
    return vectorizer, tfidf_matrix

def generate_dense_embeddings(
    texts: List[str],
    model_name: str = "allenai/specter2_base",
    batch_size: int = 64,
    max_length: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    try:
        logger.info(f"Loading model: {model_name}")
        encoder = DenseEncoder(
            model_name=model_name,
            max_length=max_length,
            device=device,
        )
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        embeddings = encoder.encode_documents(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        logger.info(f"Dense embeddings shape: {embeddings.shape}")
        return embeddings

    except ImportError:
        logger.warning("sentence-transformers not installed; using simulated embeddings.")
        logger.warning("Install with:  pip install sentence-transformers")
        return generate_simulated_embeddings(texts, model_name)


def generate_simulated_embeddings(
    texts: List[str],
    model_name: str = "simulated",
    dim: int = 768,
) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    logger.info(f"Generating simulated {dim}-dim embeddings via TF-IDF + SVD")

    tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95)
    tfidf_matrix = tfidf.fit_transform(texts)

    actual_dim = min(dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    if actual_dim < dim:
        padding = np.zeros((embeddings.shape[0], dim - actual_dim), dtype=np.float32)
        embeddings = np.hstack([embeddings, padding])

    embeddings = normalize(embeddings, norm="l2")
    logger.info(f"Simulated embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_dense_input_texts(papers: List[dict], model_name: str) -> List[str]:
    if model_name == "allenai/specter2_base":
        # AllenAI's documented SPECTER2 retrieval recipe separates title and abstract with BERT's [SEP] token.
        return [f"{p['title']} [SEP] {p['abstract']}" for p in papers]
    return [f"{p['title']}. {p['abstract']}" for p in papers]

def feature_comparison_table(
    tfidf_vectorizer,
    tfidf_matrix,
    dense_embeddings: np.ndarray,
    dense_model_name: str,
) -> Dict:
    table = {
        "representations": [
            {
                "name": "TF-IDF (sparse)",
                "dimensionality": tfidf_matrix.shape[1],
                "vocab_size": len(tfidf_vectorizer.vocabulary_),
                "sparsity_pct": 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])),
                "avg_nonzero_per_doc": tfidf_matrix.nnz / tfidf_matrix.shape[0],
                "dtype": str(tfidf_matrix.dtype),
                "storage_mb": tfidf_matrix.data.nbytes / 1e6,
            },
            {
                "name": f"Dense ({dense_model_name.split('/')[-1]})",
                "dimensionality": dense_embeddings.shape[1],
                "vocab_size": "N/A (contextual)",
                "sparsity_pct": 0.0,
                "avg_nonzero_per_doc": dense_embeddings.shape[1],
                "dtype": str(dense_embeddings.dtype),
                "storage_mb": dense_embeddings.nbytes / 1e6,
            },
        ],
        "n_documents": tfidf_matrix.shape[0],
    }
    return table


def print_feature_table(table: dict):
    print("\n" + "=" * 80)
    print("  FEATURE REPRESENTATION COMPARISON")
    print("=" * 80)
    print(f"  {'Metric':<30} {'TF-IDF (sparse)':>22} {'Dense embeddings':>22}")
    print("-" * 80)

    tfidf = table["representations"][0]
    dense = table["representations"][1]
    rows = [
        ("Dimensionality", f"{tfidf['dimensionality']:,}", f"{dense['dimensionality']:,}"),
        ("Vocabulary size", f"{tfidf['vocab_size']:,}", str(dense['vocab_size'])),
        ("Sparsity (%)", f"{tfidf['sparsity_pct']:.2f}%", f"{dense['sparsity_pct']:.1f}%"),
        ("Avg non-zero/doc", f"{tfidf['avg_nonzero_per_doc']:.0f}", f"{dense['avg_nonzero_per_doc']:,}"),
        ("Storage (MB)", f"{tfidf['storage_mb']:.1f}", f"{dense['storage_mb']:.1f}"),
        ("dtype", tfidf['dtype'], dense['dtype']),
    ]
    for name, v1, v2 in rows:
        print(f"  {name:<30} {v1:>22} {v2:>22}")
    print(f"\n  Documents: {table['n_documents']:,}")
    print("=" * 80 + "\n")

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    perplexity: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embedding dimensions for visualization."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        logger.info(f"Running t-SNE (perplexity={perplexity})")
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        )
        coords = reducer.fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap
            logger.info(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                metric="cosine",
            )
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed; falling back to t-SNE")
            return reduce_dimensions(
                embeddings, "tsne", n_components, perplexity,
                random_state=random_state,
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Reduced shape: {coords.shape}")
    return coords


def _get_colormap(name: str, n: int):
    """Get a colormap, compatible with both old and new matplotlib."""
    import matplotlib
    import matplotlib.pyplot as plt
    try:
        cmap = matplotlib.colormaps.get_cmap(name)
        if hasattr(cmap, "resampled"):
            return cmap.resampled(n)
        return cmap
    except AttributeError:
        return plt.cm.get_cmap(name, n)


def create_embedding_visualization(
    coords: np.ndarray,
    categories: List[str],
    method: str,
    output_path: str,
    dpi: int = 150,
    title_suffix: str = "",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    primary_cats = [c[0] if isinstance(c, list) and c else c for c in categories]
    cat_counts = Counter(primary_cats)
    top_cats = [c for c, _ in cat_counts.most_common(8)]
    display_cats = [c if c in top_cats else "other" for c in primary_cats]

    unique_cats = sorted(set(display_cats))
    cmap = _get_colormap("Set2", len(unique_cats))
    color_map = {cat: cmap(i) for i, cat in enumerate(unique_cats)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for cat in unique_cats:
        idxs = [i for i, c in enumerate(display_cats) if c == cat]
        ax.scatter(
            coords[idxs, 0], coords[idxs, 1],
            c=[color_map[cat]], label=f"{cat} ({cat_counts.get(cat, 0):,})",
            s=8, alpha=0.5, edgecolors="none",
        )

    method_label = "t-SNE" if method == "tsne" else "UMAP"
    ax.set_title(
        f"{method_label} Visualization of Paper Embeddings by arXiv Category{title_suffix}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel(f"{method_label} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method_label} Dimension 2", fontsize=11)
    ax.legend(loc="best", fontsize=8, framealpha=0.8, markerscale=3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Visualization saved to {output_path}")


def create_preprocessing_examples_figure(
    examples: List[dict],
    output_path: str,
    dpi: int = 150,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    n = min(len(examples), 3)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle("Preprocessing: Before vs. After Examples",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, ex in enumerate(examples[:n]):
        ax_before = axes[i][0] if n > 1 else axes[0]
        before_text = textwrap.fill(ex["original_abstract"][:300] + "...", width=60)
        ax_before.text(
            0.05, 0.95, before_text, transform=ax_before.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", alpha=0.8),
        )
        ax_before.set_title(f"Original ({ex['num_original_tokens']} tokens)", fontsize=9)
        ax_before.axis("off")

        ax_after = axes[i][1] if n > 1 else axes[1]
        after_text = textwrap.fill(ex["cleaned_text"][:300] + "...", width=60)
        ax_after.text(
            0.05, 0.95, after_text, transform=ax_after.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4edda", alpha=0.8),
        )
        ax_after.set_title(
            f"Cleaned ({ex['num_cleaned_tokens']} tokens, {ex['num_chunks']} chunks)",
            fontsize=9,
        )
        ax_after.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Preprocessing examples figure saved to {output_path}")

def build_faiss_index(
    embeddings: np.ndarray,
    output_path: str,
    use_ivf: bool = False,
    nlist: int = 100,
):
    """Build and save a FAISS index from dense embeddings."""
    try:
        import faiss

        dim = embeddings.shape[1]
        embeddings_f32 = embeddings.astype(np.float32)

        if use_ivf and embeddings.shape[0] > 10000:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings_f32)
            index.add(embeddings_f32)
            logger.info(f"Built IVF-Flat index: {index.ntotal} vectors, nlist={nlist}")
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings_f32)
            logger.info(f"Built Flat index: {index.ntotal} vectors")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, output_path)
        logger.info(f"FAISS index saved to {output_path}")
        return index

    except ImportError:
        logger.warning("FAISS not installed; skipping index build.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Build feature representations")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense embedding generation")
    parser.add_argument("--simulated", action="store_true",
                        help="Use simulated embeddings (no GPU needed)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    emb_cfg = cfg["embeddings"]
    vis_cfg = cfg["visualization"]
    data_dir = Path(cfg["data_collection"]["output_path"]).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = data_dir / "arxiv_corpus_cleaned.jsonl"
    logger.info(f"Loading cleaned corpus from {cleaned_path}")
    papers = []
    with open(cleaned_path) as f:
        for line in f:
            papers.append(json.loads(line))

    cleaned_texts = [p["cleaned_text"] for p in papers]
    raw_texts = build_dense_input_texts(papers, emb_cfg["dense_model"])
    categories = [p["categories"] for p in papers]

    vectorizer, tfidf_matrix = build_tfidf_features(
        cleaned_texts,
        max_features=emb_cfg["tfidf_max_features"],
        ngram_range=tuple(emb_cfg["tfidf_ngram_range"]),
    )

    with open(data_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    from scipy import sparse
    sparse.save_npz(str(data_dir / "tfidf_matrix.npz"), tfidf_matrix)
    logger.info("TF-IDF artifacts saved.")

    if not args.skip_dense:
        simulated_mode = bool(args.simulated)
        if args.simulated:
            dense_embeddings = generate_simulated_embeddings(
                raw_texts, emb_cfg["dense_model"], dim=768
            )
        else:
            dense_embeddings = generate_dense_embeddings(
                raw_texts,
                model_name=emb_cfg["dense_model"],
                batch_size=emb_cfg["dense_batch_size"],
                max_length=emb_cfg["dense_max_length"],
            )

        embeddings_path = data_dir / dense_embedding_filename(emb_cfg["dense_model"])
        metadata_path = data_dir / dense_embedding_metadata_filename(emb_cfg["dense_model"])
        faiss_path = data_dir / dense_faiss_index_filename(emb_cfg["dense_model"])

        np.save(str(embeddings_path), dense_embeddings)
        logger.info("Dense embeddings saved to %s.", embeddings_path)

        dense_meta = {
            "model_name": emb_cfg["dense_model"],
            "simulated": simulated_mode,
            "embedding_dim": int(dense_embeddings.shape[1]),
            "n_documents": int(dense_embeddings.shape[0]),
            "source_text": "title [SEP] abstract" if emb_cfg["dense_model"] == "allenai/specter2_base" else "title + abstract",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_file": embeddings_path.name,
            "encoder_backend": (
                "specter2_adapters"
                if emb_cfg["dense_model"] == "allenai/specter2_base" and not simulated_mode
                else "hf_mean_pooling"
                if emb_cfg["dense_model"] == "allenai/scibert_scivocab_uncased" and not simulated_mode
                else "sentence_transformers_or_simulated"
            ),
            "document_adapter": "allenai/specter2" if emb_cfg["dense_model"] == "allenai/specter2_base" and not simulated_mode else None,
            "query_adapter": "allenai/specter2_adhoc_query" if emb_cfg["dense_model"] == "allenai/specter2_base" and not simulated_mode else None,
        }
        with open(metadata_path, "w") as f:
            json.dump(dense_meta, f, indent=2)
        logger.info("Dense embedding metadata saved to %s.", metadata_path)

        build_faiss_index(dense_embeddings, str(faiss_path))

        table = feature_comparison_table(
            vectorizer, tfidf_matrix, dense_embeddings, emb_cfg["dense_model"]
        )
        print_feature_table(table)

        with open(data_dir / "feature_comparison.json", "w") as f:
            for rep in table["representations"]:
                rep["sparsity_pct"] = float(rep["sparsity_pct"])
                rep["avg_nonzero_per_doc"] = float(rep["avg_nonzero_per_doc"])
                rep["storage_mb"] = float(rep["storage_mb"])
            json.dump(table, f, indent=2)

        sample_size = min(vis_cfg["sample_size"], len(papers))
        rng = np.random.RandomState(vis_cfg["random_state"])
        sample_idx = rng.choice(len(papers), size=sample_size, replace=False)

        sample_embeddings = dense_embeddings[sample_idx]
        sample_categories = [categories[i] for i in sample_idx]

        coords = reduce_dimensions(
            sample_embeddings,
            method=vis_cfg["method"],
            n_components=vis_cfg["n_components"],
            perplexity=vis_cfg.get("perplexity", 30),
            n_neighbors=vis_cfg.get("n_neighbors", 15),
            min_dist=vis_cfg.get("min_dist", 0.1),
            random_state=vis_cfg["random_state"],
        )

        fig_dir = vis_cfg["output_dir"]
        create_embedding_visualization(
            coords, sample_categories, vis_cfg["method"],
            output_path=f"{fig_dir}/embedding_visualization_{vis_cfg['method']}.png",
            dpi=vis_cfg["figure_dpi"],
        )

    examples_path = data_dir / "preprocessing_examples.json"
    if examples_path.exists():
        with open(examples_path) as f:
            examples = json.load(f)
        create_preprocessing_examples_figure(
            examples,
            output_path=f"{vis_cfg['output_dir']}/preprocessing_before_after.png",
            dpi=vis_cfg["figure_dpi"],
        )

    logger.info("Feature representation complete.")


if __name__ == "__main__":
    main()
