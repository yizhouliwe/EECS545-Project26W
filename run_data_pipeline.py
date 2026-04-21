"""
Usage:
    python run_data_pipeline.py                    # Full pipeline (requires API + GPU)
    python run_data_pipeline.py --demo             # Synthetic corpus, simulated embeddings
    python run_data_pipeline.py --demo --skip-dense  # Minimal run (no embeddings)
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_step(description: str, module: str, extra_args: list = None):
    cmd = [sys.executable, "-m", module, "--config", "configs/config.yaml"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*70}")
    print(f"  STEP: {description}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        logger.error(f"Step failed: {description}")
        sys.exit(1)
    logger.info(f"Step completed: {description}")


def main():
    parser = argparse.ArgumentParser(description="Run Part 1 pipeline")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo corpus")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense embedding generation")
    parser.add_argument("--simulated", action="store_true",
                        help="Use simulated embeddings (TF-IDF+SVD, no GPU)")
    args = parser.parse_args()

    step1_args = ["--demo"] if args.demo else []
    run_step("Data Collection", "src.collect_data", step1_args)

    run_step("Preprocessing", "src.preprocess")
    step3_args = []
    if args.skip_dense:
        step3_args.append("--skip-dense")
    if args.simulated or args.demo:
        step3_args.append("--simulated")
    run_step("Feature Representation", "src.feature_representation", step3_args)

    data_dir = Path("data")
    print(f"\n{'='*70}")
    print("  PART 1 COMPLETE — Output Files")
    print(f"{'='*70}")
    for f in sorted(data_dir.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {size_mb:>8.2f} MB")
    fig_dir = Path("outputs/figures")
    if fig_dir.exists():
        for f in sorted(fig_dir.glob("*.png")):
            size_mb = f.stat().st_size / 1e6
            print(f"  figures/{f.name:<34} {size_mb:>8.2f} MB")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
