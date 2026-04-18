import argparse
import subprocess
import sys
from pathlib import Path


def run_step(description: str, module: str, extra_args=None):
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'=' * 70}")
    print(f"  STEP: {description}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run Part 2 helper workflow")
    parser.add_argument("--bootstrap-queries", action="store_true")
    parser.add_argument("--split-queries", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--queries", default="data/queries_val.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dense-candidates", type=int, default=100)
    args = parser.parse_args()

    if args.bootstrap_queries:
        run_step("Create starter labeled query file", "src.bootstrap_queries")
    if args.split_queries:
        run_step("Split labeled queries into train/val/test", "src.split_queries")
    if args.evaluate:
        run_step(
            "Evaluate TF-IDF, dense, and hybrid retrieval",
            "src.evaluate",
            [
                "--queries",
                args.queries,
                "--top-k",
                str(args.top_k),
                "--alpha",
                str(args.alpha),
                "--dense-candidates",
                str(args.dense_candidates),
            ],
        )

    if not any([args.bootstrap_queries, args.split_queries, args.evaluate]):
        parser.print_help()


if __name__ == "__main__":
    main()
