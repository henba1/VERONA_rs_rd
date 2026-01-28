"""
CLI script for plotting RS vs EOTPGD results.
"""

import argparse
import logging
import sys
from pathlib import Path

from plot_rs_eotpgd import DEFAULT_OUTPUT_DIR, plot_rs_eotpgd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RS lower bound and EOTPGD upper bound from an attack CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Path to CSV file with RS attack results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (e.g., 'conv_large'). If not provided, extracts from network column",
    )
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="Dataset name (default: CIFAR-10)")
    parser.add_argument(
        "--custom-colors",
        nargs="*",
        type=str,
        default=None,
        help=("Optional custom colors for verifiers (will override defaults). Format: 'verifier:color verifier:color'"),
    )

    args = parser.parse_args()

    custom_color_map = None
    if args.custom_colors is not None:
        custom_color_map = {}
        for color_spec in args.custom_colors:
            if ":" not in color_spec:
                logger.error("Invalid color specification: %s. Expected format: 'verifier:color'", color_spec)
                sys.exit(1)
            verifier, color = color_spec.split(":", 1)
            custom_color_map[verifier] = color
        logger.info("Using custom colors: %s", custom_color_map)

    try:
        plot_paths = plot_rs_eotpgd(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            custom_colors=custom_color_map,
        )
        logger.info("Successfully generated plots:")
        for plot_name, plot_path in plot_paths.items():
            logger.info("  %s: %s", plot_name, plot_path)
    except Exception as e:
        logger.error("Failed to generate plots: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
