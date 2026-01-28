import argparse
import logging
import sys
from pathlib import Path

from compare_two_dfs import generate_4_compar_plots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 4-way comparison plots from two CSVs (df1: sdpcrown vs attack, df2: RS lb vs EOTPGD ub).",
    )
    parser.add_argument("df1_csv", type=Path, help="Path to df1 CSV (e.g., compiled verifier results)")
    parser.add_argument("df2_csv", type=Path, help="Path to df2 CSV (e.g., all_attacks.csv)")

    args = parser.parse_args()

    try:
        plot_paths = generate_4_compar_plots(args.df1_csv, args.df2_csv)
    except Exception as e:
        logger.error("Failed to generate plots: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Generated plots:")
    for name, path in plot_paths.items():
        logger.info("  %s: %s", name, path)


if __name__ == "__main__":
    main()
