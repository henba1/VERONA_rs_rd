from __future__ import annotations

import argparse
import logging
from pathlib import Path

from compute_sum_stat import DEFAULT_RESULTS_DIR, compute_total_summary_df

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile total summary stats from k summary_dfs for a given experiment tag"
    )
    parser.add_argument("experiment_tag", type=str, help="Experiment tag prefix to match")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing experiment subdirs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write total_summary_df CSV (default: print to stdout)",
    )
    args = parser.parse_args()

    df = compute_total_summary_df(
        args.experiment_tag,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
    )
    if args.output is None:
        print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
