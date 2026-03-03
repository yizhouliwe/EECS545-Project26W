# EECS545-Project26W

## Stage 1 — Data Collection, Preprocessing & Feature Representation

This stage builds the full document corpus and its feature representations that later stages (retrieval, ranking, evaluation) will consume. Concretely it:

1. **Collects** 20,000 arXiv papers (cs.AI, cs.CL, cs.LG, cs.IR, cs.CV; 2024–2026) via the arXiv API.
2. **Preprocesses** each abstract — Unicode normalization, LaTeX stripping, stopword removal, lowercasing, tokenization — and splits every abstract into overlapping sentence-level chunks (≤128 tokens, 32-token overlap).
3. **Builds two feature representations** for all papers: a sparse TF-IDF matrix (50 k-dim unigram/bigram) and dense L2-normalized SPECTER2 embeddings (768-dim).

Run the whole pipeline with:

```bash
python run_part1.py          # full pipeline (arXiv API + GPU)
python run_part1.py --demo   # synthetic corpus, simulated embeddings
```

---

## Repository Structure

### Source files (`src/`)

| File                            | Description                                                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `src/collect_data.py`           | Queries the arXiv Atom API in batches and serializes raw paper metadata to `data/arxiv_corpus.jsonl`.                                 |
| `src/preprocess.py`             | Cleans and tokenizes abstracts (LaTeX removal, stopwords, Unicode normalization) and produces the cleaned corpus and sentence chunks. |
| `src/feature_representation.py` | Builds the TF-IDF sparse matrix and SPECTER2 dense embeddings from the cleaned texts, then saves them to disk.                        |

### Config & entry point

| File                  | Description                                                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `configs/config.yaml` | Central configuration for all pipeline stages (API params, preprocessing options, embedding model, visualization settings). |
| `run_part1.py`        | Orchestrator script that runs collect → preprocess → feature representation in sequence.                                    |

### Data files (`data/`)

| File                          | Description                                                                                                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `arxiv_corpus.jsonl`          | **Raw corpus** — 20,000 arXiv papers; each line is a JSON object with fields `paper_id`, `arxiv_id`, `title`, `authors`, `abstract`, `categories`, `published_at`, `updated_at`, `pdf_url`.                  |
| `arxiv_corpus_cleaned.jsonl`  | **Cleaned corpus** — same 20,000 papers as above plus two new fields: `cleaned_text` (preprocessed abstract string) and `tokens` (list of tokens after stopword removal).                                    |
| `arxiv_chunks.jsonl`          | **Sentence chunks** — 40,477 overlapping chunks (≤128 tokens, 32-token overlap) derived from the cleaned abstracts; fields: `chunk_id`, `paper_id`, `chunk_type`, `chunk_text`, `chunk_index`, `tokens_est`. |
| `tfidf_matrix.npz`            | **TF-IDF features** — scipy sparse matrix of shape `(20000, 50000)`; rows = papers, columns = unigram/bigram vocabulary; load with `scipy.sparse.load_npz`.                                                  |
| `dense_embeddings.npy`        | **Dense embeddings** — numpy float32 array of shape `(20000, 768)`; L2-normalized SPECTER2 embeddings, one row per paper in the same order as the JSONL files; load with `numpy.load`.                       |
| `corpus_stats.json`           | Summary statistics of the raw corpus (total papers, category distribution, year range, average abstract length).                                                                                             |
| `preprocessing_stats.json`    | Statistics from the preprocessing step (token counts before/after cleaning, chunk counts, token reduction %).                                                                                                |
| `preprocessing_examples.json` | A small set of before/after preprocessing examples for sanity-checking the text cleaning pipeline.                                                                                                           |
| `feature_comparison.json`     | Side-by-side comparison of TF-IDF vs. dense representations (dimensionality, sparsity %, storage size in MB).                                                                                                |

> **For downstream stages:** The primary inputs you will need are `arxiv_corpus_cleaned.jsonl` (text + metadata), `arxiv_chunks.jsonl` (retrieval chunks), `dense_embeddings.npy` (dense retrieval), and `tfidf_matrix.npz` (sparse retrieval). Row order in `.npy`/`.npz` matches line order in `arxiv_corpus_cleaned.jsonl`.
