import argparse
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARXIV_NS = {
    "atom":    "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv":   "http://arxiv.org/schemas/atom",
}

@dataclass
class Paper:
    paper_id: str
    arxiv_id: str
    version: int
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_at: str
    updated_at: str
    pdf_url: str
    source: str = "arxiv"

    @staticmethod
    def from_atom_entry(entry: ET.Element) -> Optional["Paper"]:
        try:
            raw_id = entry.find("atom:id", ARXIV_NS).text
            arxiv_id = raw_id.split("/abs/")[-1]
            version = 1
            if "v" in arxiv_id:
                parts = arxiv_id.rsplit("v", 1)
                arxiv_id = parts[0]
                version = int(parts[1]) if parts[1].isdigit() else 1

            title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ARXIV_NS).text.strip().replace("\n", " ")

            authors = [
                a.find("atom:name", ARXIV_NS).text
                for a in entry.findall("atom:author", ARXIV_NS)
            ]
            categories = [
                c.get("term")
                for c in entry.findall("atom:category", ARXIV_NS)
            ]
            published = entry.find("atom:published", ARXIV_NS).text
            updated   = entry.find("atom:updated", ARXIV_NS).text

            pdf_links = [
                l.get("href") for l in entry.findall("atom:link", ARXIV_NS)
                if l.get("title") == "pdf"
            ]
            pdf_url = pdf_links[0] if pdf_links else ""

            return Paper(
                paper_id=str(uuid.uuid4()),
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published_at=published,
                updated_at=updated,
                pdf_url=pdf_url,
            )
        except Exception as e:
            logger.warning(f"Skipping entry due to parse error: {e}")
            return None

def fetch_batch(
    base_url: str, query: str, start: int, batch_size: int,
    max_retries: int = 5,
) -> List[Paper]:
    params = (
        f"?search_query={urllib.parse.quote(query)}"
        f"&start={start}&max_results={batch_size}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    url = base_url + params
    logger.info(f"  Fetching start={start}, size={batch_size}")

    xml_bytes = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ResearchAssistant/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                xml_bytes = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1) * 3, 120)
                logger.warning(
                    f"  HTTP {e.code} on attempt {attempt+1}/{max_retries}; "
                    f"retrying in {wait}s …"
                )
                time.sleep(wait)
            else:
                raise
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1) * 3, 120)
                logger.warning(
                    f"  URL error ({e.reason}) on attempt {attempt+1}; "
                    f"retrying in {wait}s …"
                )
                time.sleep(wait)
            else:
                raise

    if xml_bytes is None:
        return []

    root = ET.fromstring(xml_bytes)
    entries = root.findall("atom:entry", ARXIV_NS)

    papers = []
    parse_failures = 0
    for entry in entries:
        p = Paper.from_atom_entry(entry)
        if p is not None:
            papers.append(p)
        else:
            parse_failures += 1

    if parse_failures > 0:
        logger.warning(
            f"  Parsed {len(papers)}, failed {parse_failures} "
            f"out of {len(entries)} entries"
        )
    if len(entries) == 0:
        logger.warning(f"  No <entry> elements. Root tag: {root.tag}")

    return papers

def collect_single_category(
    base_url: str,
    category: str,
    max_papers: int,
    batch_size: int,
    delay: float,
    start_year: int,
    end_year: int,
    seen_ids: set,
) -> List[Paper]:
    query = f"cat:{category}"
    logger.info(f"── Collecting category: {category} (budget ≤{max_papers}) ──")

    papers = []
    start = 0
    consecutive_failures = 0
    max_consecutive_failures = 3

    while len(papers) < max_papers:
        try:
            batch = fetch_batch(base_url, query, start, batch_size)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.warning(
                f"  Fetch failed for {category} at offset {start}: {e} "
                f"({consecutive_failures}/{max_consecutive_failures})"
            )
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"  Stopping {category} after {consecutive_failures} "
                    f"consecutive failures. Got {len(papers)} papers."
                )
                break
            time.sleep(delay * 3)
            continue

        if not batch:
            logger.info(f"  No more results for {category}; got {len(papers)}.")
            break

        batch_added = 0
        for p in batch:
            if p.arxiv_id not in seen_ids:
                pub_year = int(p.published_at[:4])
                if start_year <= pub_year <= end_year:
                    papers.append(p)
                    seen_ids.add(p.arxiv_id)
                    batch_added += 1

        start += batch_size
        logger.info(f"  {category}: {len(papers)} papers (+{batch_added} this batch)")

        if len(batch) < batch_size:
            logger.info(f"  Partial batch — end of {category} results.")
            break

        time.sleep(delay)

    return papers


def collect_corpus(cfg: dict) -> List[Paper]:
    dc = cfg["data_collection"]
    categories = dc["target_categories"]
    max_papers = dc["max_papers"]
    batch_size = dc["batch_size"]
    delay = dc["api_delay_seconds"]
    start_year = int(dc["date_range"]["start"][:4])
    end_year = int(dc["date_range"]["end"][:4])

    logger.info(f"Date filter: {start_year}–{end_year}")
    logger.info(f"Target: {max_papers} papers across {len(categories)} categories")

    per_cat_budget = int(max_papers / len(categories) * 1.5)
    logger.info(f"Per-category budget: ~{per_cat_budget}")

    all_papers: List[Paper] = []
    seen_ids: set = set()

    for cat in categories:
        if len(all_papers) >= max_papers:
            break

        remaining = max_papers - len(all_papers)
        budget = min(per_cat_budget, remaining + 1000)  

        cat_papers = collect_single_category(
            base_url=dc["arxiv_base_url"],
            category=cat,
            max_papers=budget,
            batch_size=batch_size,
            delay=delay,
            start_year=start_year,
            end_year=end_year,
            seen_ids=seen_ids,
        )
        all_papers.extend(cat_papers)
        logger.info(
            f"  After {cat}: {len(all_papers)} total papers, "
            f"{len(seen_ids)} unique IDs"
        )

        if cat != categories[-1]:
            logger.info("  Pausing 5s between categories…")
            time.sleep(5)

    logger.info(f"Collection complete: {len(all_papers)} papers total")
    return all_papers[:max_papers]

def save_corpus(papers: List[Paper], path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in papers:
            f.write(json.dumps(asdict(p)) + "\n")
    logger.info(f"Saved {len(papers)} papers to {out}")


def load_corpus(path: str) -> List[dict]:
    papers = []
    with open(path) as f:
        for line in f:
            papers.append(json.loads(line))
    return papers

def corpus_statistics(papers: List[dict]) -> dict:
    from collections import Counter

    total = len(papers)
    cat_counter = Counter()
    years = []
    abstract_lengths = []

    for p in papers:
        for c in p["categories"]:
            cat_counter[c] += 1
        years.append(int(p["published_at"][:4]))
        abstract_lengths.append(len(p["abstract"].split()))

    year_range = (min(years), max(years)) if years else (None, None)
    top_categories = cat_counter.most_common(10)

    return {
        "total_papers": total,
        "unique_categories": len(cat_counter),
        "year_range": year_range,
        "top_categories": top_categories,
        "avg_abstract_words": sum(abstract_lengths) / max(len(abstract_lengths), 1),
        "median_abstract_words": sorted(abstract_lengths)[len(abstract_lengths) // 2]
        if abstract_lengths else 0,
    }


def print_stats_table(stats: dict):
    print("\n" + "=" * 60)
    print("  CORPUS STATISTICS")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Value':>20}")
    print("-" * 60)
    print(f"  {'Total papers':<35} {stats['total_papers']:>20,}")
    print(f"  {'Unique categories':<35} {stats['unique_categories']:>20}")
    print(f"  {'Year range':<35} {str(stats['year_range']):>20}")
    print(f"  {'Avg abstract length (words)':<35} {stats['avg_abstract_words']:>20.1f}")
    print(f"  {'Median abstract length (words)':<35} {stats['median_abstract_words']:>20}")
    print("-" * 60)
    print("  Top categories:")
    for cat, count in stats["top_categories"]:
        pct = 100 * count / stats["total_papers"]
        print(f"    {cat:<30} {count:>8,}  ({pct:5.1f}%)")
    print("=" * 60 + "\n")


def generate_demo_corpus(n: int = 5000) -> List[Paper]:
    import random
    random.seed(42)

    categories_pool = ["cs.CL", "cs.IR", "cs.AI", "cs.LG", "cs.CV"]
    method_terms = [
        "transformer", "attention mechanism", "BERT", "graph neural network",
        "reinforcement learning", "contrastive learning", "diffusion model",
        "retrieval-augmented generation", "knowledge distillation", "prompt tuning",
        "federated learning", "neural architecture search", "self-supervised",
        "vision transformer", "language model", "embedding", "fine-tuning",
        "multi-task learning", "few-shot learning", "zero-shot",
    ]
    domain_terms = {
        "cs.CL": ["natural language processing", "text classification",
                   "machine translation", "sentiment analysis",
                   "named entity recognition", "question answering",
                   "summarization", "dialogue systems", "language understanding"],
        "cs.IR": ["information retrieval", "search engines",
                   "recommendation systems", "query expansion",
                   "relevance feedback", "document ranking",
                   "collaborative filtering", "click-through prediction"],
        "cs.AI": ["artificial intelligence", "planning", "reasoning",
                   "knowledge representation", "multi-agent systems",
                   "constraint satisfaction", "automated reasoning"],
        "cs.LG": ["machine learning", "deep learning", "optimization",
                   "generalization", "regularization", "Bayesian methods",
                   "ensemble methods", "kernel methods"],
        "cs.CV": ["computer vision", "image classification",
                   "object detection", "semantic segmentation",
                   "image generation", "visual recognition",
                   "3D reconstruction", "video understanding"],
    }

    papers = []
    for i in range(n):
        primary_cat = random.choice(categories_pool)
        n_cats = random.randint(1, 3)
        cats = list(set([primary_cat] + random.choices(categories_pool, k=n_cats - 1)))
        year = random.randint(2020, 2026)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        pub_date = f"{year}-{month:02d}-{day:02d}T00:00:00Z"
        method = random.choice(method_terms)
        domain = random.choice(domain_terms[primary_cat])
        title = (
            f"{method.title()} for {domain.title()}: "
            f"A {'Novel' if random.random() > 0.5 else 'Comprehensive'} Approach"
        )
        abstract_sentences = [
            f"We present a novel approach to {domain} using {method}.",
            f"Recent advances in {random.choice(method_terms)} have shown promising results in {domain}.",
            f"Our method builds on {random.choice(method_terms)} and extends it to handle {domain} tasks.",
            f"We evaluate our approach on standard benchmarks and demonstrate "
            f"{'state-of-the-art' if random.random() > 0.3 else 'competitive'} performance.",
            f"Experiments show improvements of {random.uniform(1, 15):.1f}% over strong baselines.",
            f"We further analyze the impact of "
            f"{random.choice(['model size', 'training data', 'hyperparameters', 'architecture choices'])} "
            f"on performance.",
            f"Our code and models are {'publicly available' if random.random() > 0.4 else 'available upon request'}.",
        ]
        n_sentences = random.randint(4, 7)
        abstract = " ".join(random.sample(abstract_sentences, min(n_sentences, len(abstract_sentences))))
        n_authors = random.randint(1, 6)
        first_names = ["Wei", "Jia", "Yun", "Alex", "Sarah", "Chen", "Maria", "Raj", "Kim", "Omar"]
        last_names = ["Zhang", "Wang", "Li", "Smith", "Johnson", "Kumar", "Garcia", "Chen", "Park", "Ahmed"]
        authors = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_authors)]
        arxiv_id = f"{year % 100:02d}{month:02d}.{random.randint(10000, 99999)}"
        papers.append(Paper(
            paper_id=str(uuid.uuid4()), arxiv_id=arxiv_id, version=1,
            title=title, authors=authors, abstract=abstract, categories=cats,
            published_at=pub_date, updated_at=pub_date,
            pdf_url=f"http://arxiv.org/pdf/{arxiv_id}v1",
        ))
    return papers

def main():
    parser = argparse.ArgumentParser(description="Collect arXiv corpus")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic demo corpus (no API calls)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.demo:
        logger.info("Generating synthetic demo corpus...")
        papers = generate_demo_corpus(cfg["data_collection"]["max_papers"])
    else:
        papers = collect_corpus(cfg)

    output_path = cfg["data_collection"]["output_path"]
    save_corpus(papers, output_path)

    loaded = load_corpus(output_path)
    stats = corpus_statistics(loaded)
    print_stats_table(stats)

    stats_path = Path(output_path).parent / "corpus_stats.json"
    stats_json = {
        **stats,
        "top_categories": [[c, n] for c, n in stats["top_categories"]],
        "year_range": list(stats["year_range"]),
    }
    with open(stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    logger.info(f"Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()