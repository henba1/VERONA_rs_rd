import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from agg_report_creator import generate_verifier_comparison_plots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and return as DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    logger.info("Loading CSV from %s", path)
    return pd.read_csv(path)


def main() -> None:
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
    parser.add_argument(
        "--custom-colors",
        nargs="*",
        type=str,
        default=None,
        help=(
            "Optional custom colors for the hue values. "
            "If provided, must have the same length as --custom-labels (or number of CSV paths if no custom labels). "
            "Colors can be specified as hex codes (e.g., '#9b59b6'), named colors (e.g., 'purple'), or RGB tuples. "
            "Each color will be mapped to the corresponding label in order."
        ),
    )

    args = parser.parse_args()

    # Validate custom labels, if any
    if args.custom_labels is not None and len(args.custom_labels) != len(args.csv_paths):
        logger.error(
            "Number of --custom-labels (%d) must match number of CSV paths (%d)",
            len(args.custom_labels),
            len(args.csv_paths),
        )
        sys.exit(1)

    # Validate custom colors, if any
    if args.custom_colors is not None:
        num_labels = len(args.custom_labels) if args.custom_labels is not None else len(args.csv_paths)
        if len(args.custom_colors) != num_labels:
            logger.error(
                "Number of --custom-colors (%d) must match number of labels (%d)",
                len(args.custom_colors),
                num_labels,
            )
            sys.exit(1)

    logger.info("Loading %d CSV file(s)", len(args.csv_paths))
    dataframes: list[pd.DataFrame] = []
    for idx, csv_path in enumerate(args.csv_paths):
        try:
            df = load_csv(csv_path)

            # If custom labels are provided, overwrite the chosen hue column
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
            logger.info("  Loaded %s: %d rows", csv_path.name, len(df))
        except Exception as e:  # pragma: no cover - CLI error handling
            logger.error("Failed to load %s: %s", csv_path, e)
            sys.exit(1)

    if not dataframes:
        logger.error("No dataframes loaded. Exiting.")
        sys.exit(1)

    # Build color mapping if custom colors are provided
    custom_color_map = None
    if args.custom_colors is not None:
        if args.custom_labels is not None:
            # Use custom labels if provided
            labels = args.custom_labels
        else:
            # Otherwise, get unique hue values from each dataframe
            labels = []
            for df in dataframes:
                hue_col = args.hue_by
                if hue_col in df.columns:
                    unique_values = df[hue_col].unique()
                    if len(unique_values) > 0:
                        labels.append(unique_values[0])
                    else:
                        logger.warning("No values found in hue column '%s' for one of the dataframes", hue_col)
                        labels.append(f"unknown_{len(labels)}")
                else:
                    logger.warning("Hue column '%s' not found in one of the dataframes", hue_col)
                    labels.append(f"unknown_{len(labels)}")
        custom_color_map = dict(zip(labels, args.custom_colors, strict=True))
        logger.info("Using custom color mapping: %s", custom_color_map)

    # Generate comparison plots
    logger.info("Generating comparison plots for %d dataset(s)", len(dataframes))
    try:
        plot_paths = generate_verifier_comparison_plots(
            dataframes=dataframes,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            hue_by=args.hue_by,
            custom_colors=custom_color_map,
        )
        logger.info("Successfully generated all plots:")
        for plot_name, plot_path in plot_paths.items():
            logger.info("  %s: %s", plot_name, plot_path)
    except Exception as e:  # pragma: no cover - CLI error handling
        logger.error("Failed to generate plots: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
