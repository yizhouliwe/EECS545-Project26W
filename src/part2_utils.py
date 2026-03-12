import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import yaml
from scipy import sparse


@dataclass
class CorpusRecord:
    paper_id: str
    arxiv_id: str
    title: str
    abstract: str
    categories: List[str]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_papers(corpus_path: str) -> List[dict]:
    with open(corpus_path) as f:
        return [json.loads(line) for line in f]


def build_paper_lookup(papers: Sequence[dict]) -> Dict[str, dict]:
    return {paper["paper_id"]: paper for paper in papers}


def load_tfidf_artifacts(data_dir: Path):
    with open(data_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    matrix = sparse.load_npz(data_dir / "tfidf_matrix.npz")
    return vectorizer, matrix


def load_dense_embeddings(data_dir: Path) -> np.ndarray:
    return np.load(data_dir / "dense_embeddings.npy")


def load_dense_embedding_metadata(data_dir: Path) -> dict | None:
    meta_path = data_dir / "dense_embeddings_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score - min_score < 1e-8:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - min_score) / (max_score - min_score)


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
