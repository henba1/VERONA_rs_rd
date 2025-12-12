#!/usr/bin/env python3
"""
Simple script to compare multiple result CSV files and generate comparison plots.

Usage:
    python compare_results.py path/to/results1.csv path/to/results2.csv [path/to/results3.csv ...]
    python compare_results.py --dataset CIFAR-10 path/to/results1.csv path/to/results2.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from generate_comparison_plots import generate_verifier_comparison_plots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and return as DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    logger.info(f"Loading CSV from {path}")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple result CSV files and generate comparison plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="Paths to CSV files containing results to compare",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR-10",
        help="Dataset name (default: CIFAR-10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots. If not specified, uses RESULTS_DIR/dataset_name",
    )
    parser.add_argument(
        "--hue-by",
        type=str,
        choices=["network", "verifier"],
        default="verifier",
        help="Column to use for hue/grouping in plots: 'network' or 'verifier' (default: verifier)",
    )
    parser.add_argument(
        "--custom-labels",
        nargs="*",
        type=str,
        default=None,
        help=(
            "Optional custom labels for the hue column (network/verifier). "
            "If provided, must have the same length as the number of CSV paths; "
            "each label will overwrite the corresponding CSV's value in the chosen hue column."
        ),
    )

    args = parser.parse_args()

    if args.custom_labels is not None and len(args.custom_labels) != len(args.csv_paths):
            logger.error(
                "Number of --custom-labels (%d) must match number of CSV paths (%d)",
                len(args.custom_labels),
                len(args.csv_paths),
            )
            sys.exit(1)

    logger.info(f"Loading {len(args.csv_paths)} CSV file(s)")
    dataframes = []
    for idx, csv_path in enumerate(args.csv_paths):
        try:
            df = load_csv(csv_path)

            if args.custom_labels is not None:
                label = args.custom_labels[idx]
                hue_col = args.hue_by
                if hue_col not in df.columns:
                    logger.warning(
                        "Hue column '%s' not found in %s; creating it with custom label '%s'",
                        hue_col,
                        csv_path,
                        label,
                    )
                df[hue_col] = label

            dataframes.append(df)
            logger.info(f"  Loaded {csv_path.name}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load {csv_path}: {e}")
            sys.exit(1)

    if len(dataframes) == 0:
        logger.error("No dataframes loaded. Exiting.")
        sys.exit(1)

    # Generate comparison plots
    logger.info(f"Generating comparison plots for {len(dataframes)} dataset(s)")
    try:
        plot_paths = generate_verifier_comparison_plots(
            dataframes=dataframes,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            hue_by=args.hue_by,
        )
        logger.info("Successfully generated all plots:")
        for plot_name, plot_path in plot_paths.items():
            logger.info(f"  {plot_name}: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

