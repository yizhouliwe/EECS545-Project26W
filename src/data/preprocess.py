import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours
ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those
through to too under until up very was wasn't we we'd we'll we're we've
were weren't what what's when when's where where's which while who who's
whom why why's will with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves
""".split())

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text)


def remove_special_characters(text: str) -> str:
    text = re.sub(r"[^\w\s\-\.,;:()']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_latex_artifacts(text: str) -> str:
    text = re.sub(r"\$[^$]+\$", " [MATH] ", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", "", text)
    return text


def tokenize(text: str) -> List[str]:
    raw_tokens = text.split()
    tokens = []
    for t in raw_tokens:
        cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", t)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def clean_and_tokenize(
    text: str,
    lowercase: bool = True,
    remove_stops: bool = True,
    remove_special: bool = True,
    min_token_length: int = 2,
) -> Tuple[str, List[str]]:
    text = normalize_unicode(text)
    text = remove_latex_artifacts(text)
    if remove_special:
        text = remove_special_characters(text)
    if lowercase:
        text = text.lower()
    tokens = tokenize(text)
    if remove_stops:
        tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [t for t in tokens if len(t) >= min_token_length]

    cleaned_text = " ".join(tokens)
    return cleaned_text, tokens

def sentence_split(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(
    text: str,
    paper_id: str,
    max_tokens: int = 128,
    overlap_tokens: int = 32,
) -> List[Dict]:
    sentences = sentence_split(text)
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    chunk_idx = 0
    char_pos = 0

    for sent in sentences:
        sent_tokens = len(sent.split())

        if current_token_count + sent_tokens > max_tokens and current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                "paper_id": paper_id,
                "chunk_type": "abstract",
                "chunk_text": chunk_text,
                "chunk_index": chunk_idx,
                "tokens_est": current_token_count,
            })
            chunk_idx += 1
            overlap_count = 0
            overlap_sentences = []
            for s in reversed(current_chunk_sentences):
                s_len = len(s.split())
                if overlap_count + s_len > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += s_len

            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_count

        current_chunk_sentences.append(sent)
        current_token_count += sent_tokens

    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append({
            "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
            "paper_id": paper_id,
            "chunk_type": "abstract",
            "chunk_text": chunk_text,
            "chunk_index": chunk_idx,
            "tokens_est": current_token_count,
        })

    return chunks


def chunk_by_fixed_window(
    text: str,
    paper_id: str,
    max_tokens: int = 128,
    overlap_tokens: int = 32,
) -> List[Dict]:
    words = text.split()
    chunks = []
    step = max_tokens - overlap_tokens
    chunk_idx = 0

    for i in range(0, len(words), step):
        window = words[i : i + max_tokens]
        if not window:
            break
        chunk_text = " ".join(window)
        chunks.append({
            "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
            "paper_id": paper_id,
            "chunk_type": "abstract",
            "chunk_text": chunk_text,
            "chunk_index": chunk_idx,
            "tokens_est": len(window),
        })
        chunk_idx += 1

    return chunks

def preprocess_corpus(
    papers: List[dict], cfg: dict
) -> Tuple[List[dict], List[dict], List[dict]]:
    pp = cfg["preprocessing"]
    cleaned_papers = []
    all_chunks = []
    examples = []

    for i, paper in enumerate(papers):
        raw_abstract = paper["abstract"]

        cleaned_text, tokens = clean_and_tokenize(
            raw_abstract,
            lowercase=pp["lowercase"],
            remove_stops=pp["remove_stopwords"],
            remove_special=pp["remove_special_chars"],
            min_token_length=pp["min_token_length"],
        )

        paper_clean = {**paper, "cleaned_text": cleaned_text, "tokens": tokens}
        cleaned_papers.append(paper_clean)

        chunk_fn = chunk_by_sentences if pp["chunk_strategy"] == "sentence" else chunk_by_fixed_window
        chunks = chunk_fn(
            raw_abstract,
            paper["paper_id"],
            max_tokens=pp["chunk_max_tokens"],
            overlap_tokens=pp["chunk_overlap_tokens"],
        )
        all_chunks.extend(chunks)

        if i < 5:
            examples.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "original_abstract": raw_abstract[:500],
                "cleaned_text": cleaned_text[:500],
                "num_original_tokens": len(raw_abstract.split()),
                "num_cleaned_tokens": len(tokens),
                "num_chunks": len(chunks),
            })

        if (i + 1) % 2000 == 0:
            logger.info(f"Preprocessed {i + 1}/{len(papers)} papers")

    logger.info(f"Preprocessing complete: {len(cleaned_papers)} papers, {len(all_chunks)} chunks")
    return cleaned_papers, all_chunks, examples

def preprocessing_statistics(cleaned_papers: List[dict], all_chunks: List[dict]) -> dict:
    original_lengths = [len(p["abstract"].split()) for p in cleaned_papers]
    cleaned_lengths = [len(p["tokens"]) for p in cleaned_papers]
    chunk_lengths = [c["tokens_est"] for c in all_chunks]

    stats = {
        "total_papers": len(cleaned_papers),
        "total_chunks": len(all_chunks),
        "avg_chunks_per_paper": len(all_chunks) / max(len(cleaned_papers), 1),
        "avg_original_tokens": sum(original_lengths) / max(len(original_lengths), 1),
        "avg_cleaned_tokens": sum(cleaned_lengths) / max(len(cleaned_lengths), 1),
        "token_reduction_pct": 100 * (1 - sum(cleaned_lengths) / max(sum(original_lengths), 1)),
        "avg_chunk_tokens": sum(chunk_lengths) / max(len(chunk_lengths), 1),
        "min_chunk_tokens": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_tokens": max(chunk_lengths) if chunk_lengths else 0,
    }
    return stats

def main():
    parser = argparse.ArgumentParser(description="Preprocess arXiv corpus")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    corpus_path = cfg["data_collection"]["output_path"]
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path) as f:
        papers = [json.loads(line) for line in f]

    cleaned_papers, all_chunks, examples = preprocess_corpus(papers, cfg)

    data_dir = Path(corpus_path).parent

    cleaned_path = data_dir / "arxiv_corpus_cleaned.jsonl"
    with open(cleaned_path, "w") as f:
        for p in cleaned_papers:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Cleaned corpus saved to {cleaned_path}")

    chunks_path = data_dir / "arxiv_chunks.jsonl"
    with open(chunks_path, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    logger.info(f"Chunks saved to {chunks_path}")

    examples_path = data_dir / "preprocessing_examples.json"
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)
    logger.info(f"Examples saved to {examples_path}")

    stats = preprocessing_statistics(cleaned_papers, all_chunks)
    print("\n" + "=" * 60)
    print("  PREPROCESSING STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>20.2f}")
        else:
            print(f"  {k:<35} {v:>20}")
    print("=" * 60 + "\n")

    stats_path = data_dir / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
