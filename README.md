# EECS545-Project26W

## Stage 1 — Data Collection, Preprocessing & Feature Representation

This stage builds the full document corpus and its feature representations that later stages (retrieval, ranking, evaluation) will consume. Concretely it:

1. **Collects** 20,000 arXiv papers (cs.AI, cs.CL, cs.LG, cs.IR, cs.CV; 2024–2026) via the arXiv API.
2. **Preprocesses** each abstract — Unicode normalization, LaTeX stripping, stopword removal, lowercasing, tokenization — and splits every abstract into overlapping sentence-level chunks (≤128 tokens, 32-token overlap).
3. **Builds two feature representations** for all papers: a sparse TF-IDF matrix (50 k-dim unigram/bigram) and dense L2-normalized pretrained sentence-transformer embeddings.

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
| `src/feature_representation.py` | Builds the TF-IDF sparse matrix and pretrained dense sentence-transformer embeddings from the cleaned texts, then saves them to disk. |

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
| `dense_embeddings.npy`        | **Dense embeddings** — numpy float32 array of shape `(20000, d)`; L2-normalized pretrained sentence-transformer embeddings, one row per paper in the same order as the JSONL files; load with `numpy.load`.   |
| `corpus_stats.json`           | Summary statistics of the raw corpus (total papers, category distribution, year range, average abstract length).                                                                                             |
| `preprocessing_stats.json`    | Statistics from the preprocessing step (token counts before/after cleaning, chunk counts, token reduction %).                                                                                                |
| `preprocessing_examples.json` | A small set of before/after preprocessing examples for sanity-checking the text cleaning pipeline.                                                                                                           |
| `feature_comparison.json`     | Side-by-side comparison of TF-IDF vs. dense representations (dimensionality, sparsity %, storage size in MB).                                                                                                |

> **For downstream stages:** The primary inputs you will need are `arxiv_corpus_cleaned.jsonl` (text + metadata), `arxiv_chunks.jsonl` (retrieval chunks), `dense_embeddings.npy` (dense retrieval), and `tfidf_matrix.npz` (sparse retrieval). Row order in `.npy`/`.npz` matches line order in `arxiv_corpus_cleaned.jsonl`.

---

## Stage 2 — Retrieval Pipeline & Initial Results

This stage builds a **paper-level retrieval pipeline** on top of the Part 1 artifacts and evaluates three retrieval modes:

1. **TF-IDF baseline** over the paper-level sparse matrix.
2. **Dense retrieval** over the paper-level dense embeddings.
3. **Hybrid reranking** that uses dense retrieval for candidate generation and combines dense + sparse scores for the final ranking.

### Stage 2 workflow

1. **Create a labeled query set** in `data/labeled_queries.jsonl`.
2. **Split queries** into train / validation / test using a 60% / 20% / 20% hold-out CV protocol.
3. **Run retrieval** in `tfidf`, `dense`, or `hybrid` mode.
4. **Evaluate** Precision@K, Recall@K, NDCG@10, and MAP on validation or test queries.

### Source files added for Stage 2

| File                    | Description                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| `src/part2_utils.py`    | Shared helpers for loading config, JSONL files, TF-IDF artifacts, dense embeddings, and score normalization. |
| `src/retrieval.py`      | Implements paper-level TF-IDF, dense, and hybrid retrieval.                                                  |
| `src/evaluate.py`       | Evaluates retrieval outputs with Precision@K, Recall@K, NDCG@10, and MAP.                                   |
| `src/bootstrap_queries.py` | Creates a starter `data/labeled_queries.jsonl` file with paper-level query templates.                    |
| `src/labeling_helper.py` | Prints candidate papers plus abstract snippets to help manual query labeling.                               |
| `src/split_queries.py`  | Splits labeled queries into `queries_train.jsonl`, `queries_val.jsonl`, and `queries_test.jsonl`.           |
| `run_part2.py`          | Convenience runner for bootstrapping queries, splitting labels, and launching evaluation.                    |

### Additional data files used in Stage 2

| File                  | Description                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `labeled_queries.jsonl` | Paper-level benchmark queries with fields `query_id`, `query_text`, and `relevant_arxiv_ids`. |
| `queries_train.jsonl` | 60% train split of labeled queries.                                                              |
| `queries_val.jsonl`   | 20% validation split used for tuning.                                                            |
| `queries_test.jsonl`  | 20% held-out test split used for final reporting.                                                |

### Step-by-step usage

Create starter queries:

```bash
python3 run_part2.py --bootstrap-queries
```

Inspect likely relevant papers for a query and manually fill `relevant_arxiv_ids`:

```bash
python3 -m src.labeling_helper --query "retrieval augmented generation" --k 10
```

Split labeled queries into train / validation / test:

```bash
python3 run_part2.py --split-queries
```

Run TF-IDF retrieval for a single query:

```bash
python3 -m src.retrieval --mode tfidf --query "retrieval augmented generation" --k 5
```

Evaluate all three methods on the validation split:

```bash
python3 -m src.evaluate --queries data/queries_val.jsonl --top-k 5 --alpha 0.5 --dense-candidates 100
```

Evaluate on the held-out test split after tuning:

```bash
python3 -m src.evaluate --queries data/queries_test.jsonl --top-k 5 --alpha 0.5 --dense-candidates 100
```

### Notes on dense retrieval

- Dense retrieval expects the stored paper embeddings and the query encoder to be consistent.
- Dense artifacts are now model-specific:
  - `allenai/specter2_base` -> `data/dense_embeddings_specter2.npy`, `data/dense_embeddings_specter2_meta.json`
  - `sentence-transformers/all-MiniLM-L6-v2` -> legacy `data/dense_embeddings.npy`, `data/dense_embeddings_meta.json`
- Use `--dense-model` with `src.retrieval`, `src.evaluate`, or `run_part2.py` to switch between SPECTER2 and MiniLM for ablations.
- To regenerate dense artifacts and metadata, rerun Part 1:

```bash
python3 run_part1.py
```

### Evaluation protocol

- **Retrieval unit:** paper-level
- **Cross-validation setup:** hold-out CV with 60% train / 20% validation / 20% test
- **Tuning split:** validation
- **Final reporting split:** test
- **Methods compared:** TF-IDF, Dense, Hybrid
- **Metrics:** Precision@K, Recall@K, NDCG@10, MAP

### Current validation results

Using the current validation split (`data/queries_val.jsonl`) with `top-k=10` and `dense-candidates=100`, the current paper-level results are:

| Method | Precision@10 | Recall@10 | NDCG@10 | MAP |
| ------ | ------------ | --------- | ------- | --- |
| TF-IDF | 0.4421 | 0.1957 | 0.4452 | 0.1131 |
| Dense (MiniLM) | 0.5105 | 0.2334 | 0.5275 | 0.1498 |
| Hybrid (MiniLM) | 0.5263 | 0.2441 | 0.5515 | 0.1642 |
| Dense (SPECTER2) | 0.3474 | 0.1739 | 0.3781 | 0.1069 |
| Hybrid (SPECTER2) | 0.4526 | 0.2222 | 0.5012 | 0.1538 |

On the current 19-query validation split, **MiniLM remains the strongest dense encoder in this repo**, while **SPECTER2 is now fully supported as an ablation-ready scientific-domain alternative** with separate artifacts and metadata. Hybrid reranking still outperforms dense-only retrieval for both encoders.

### Reporting guidance for Part 2

For the report, document:

1. The three-stage pipeline: dense candidate generation -> hybrid reranking -> top-N selection.
2. The hold-out query split and which split was used for tuning.
3. The hyperparameters used: embedding model, top-K, dense candidate pool, hybrid weight `alpha`.
4. A results table comparing `TF-IDF`, `Dense`, and `Hybrid`.
5. A comparison plot across the reported metrics.


## Stage 3 — Interactive Feedback & Retrieval-Augmented Generation (RAG)
### Stage 3 Workflow

1.  **Initial Retrieval**: Performs dense retrieval using an L2-normalized query vector.
2.  **User/Pseudo Feedback**: Collects relevance labels. Supports **Pseudo-Relevance Feedback (PRF)**, which automatically assumes the Top-1 result is relevant to enable testing without manual labels.
3.  **Query Refinement**: Updates the query vector using the **Rocchio Algorithm**, shifting the search embedding toward the relevant document cluster and away from non-relevant ones.
4.  **Grounded Generation**: Feeds the refined Top-K abstracts into a Large Language Model (**UM GPT-oss-120B**) to produce a research summary with precise inline citations.

### Source Files Added for Stage 3

| File | Description |
| :--- | :--- |
| `src/retrieval_extended.py` | Extends the Part 2 retriever with vector-based search and model-specific dense artifact loading (SPECTER2 by default). |
| `src/feedback_logic.py` | Implements the core Rocchio algorithm: $Q_{new} = \alpha Q_{old} + \beta \mu(D_{pos}) - \gamma \mu(D_{neg})$. |
| `src/qa_engine.py` | Orchestrates the RAG pipeline, including prompt engineering for scientific grounding and citation parsing. |
| `run_part3.py` | The main orchestrator for the interactive loop, feedback processing, and grounded QA evaluation. |

### Step-by-Step Usage

Run the full interactive pipeline (e.g., 1 round of feedback):

```bash
python run_part3.py --queries data/queries_val.jsonl --rounds 1
