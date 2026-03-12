import argparse
import random
from pathlib import Path

from src.part2_utils import load_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Split labeled queries into train/val/test")
    parser.add_argument("--input", default="data/labeled_queries.jsonl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Query file not found: {input_path}")
        print("Create it first with: python3 run_part2.py --bootstrap-queries")
        return

    rows = load_jsonl(input_path)
    labeled_rows = [row for row in rows if row.get("relevant_arxiv_ids")]
    if len(labeled_rows) != len(rows):
        print(
            "Warning: some queries have empty relevant_arxiv_ids and were excluded from the split. "
            "Finish labeling them before final evaluation."
        )
    rows = labeled_rows

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_total = len(rows)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    output_dir = Path(args.output_dir)
    save_jsonl(output_dir / "queries_train.jsonl", train_rows)
    save_jsonl(output_dir / "queries_val.jsonl", val_rows)
    save_jsonl(output_dir / "queries_test.jsonl", test_rows)

    print(f"Train: {len(train_rows)}")
    print(f"Val:   {len(val_rows)}")
    print(f"Test:  {len(test_rows)}")


if __name__ == "__main__":
    main()
