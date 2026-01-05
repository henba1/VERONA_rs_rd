"""
Automatically compile dataframes from experiments with the same experiment_tag prefix.

This script scans the results directory for all experiment folders matching a given
experiment_tag prefix and compiles a specified dataframe (e.g., result_df, summary_df)
from all matching directories.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from agg_report_creator import generate_verifier_comparison_plots

from ada_verona import get_results_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_experiment_directories(results_dir: Path, experiment_tag: str, dataset_name: str) -> list[Path]:
    """
    Find all experiment directories matching the experiment_tag prefix.

    Args:
        results_dir: Base results directory (already includes dataset_name if from get_results_dir)
        experiment_tag: Experiment tag prefix to search for (e.g., "base_predict")
        dataset_name: Dataset name (used for logging only)

    Returns:
        List of Path objects pointing to matching experiment directories
    """
    # get_results_dir(dataset_name) already includes the dataset name in the path
    # So results_dir is already the dataset-specific directory
    dataset_dir = results_dir
    if not dataset_dir.exists():
        logger.error("Dataset directory does not exist: %s", dataset_dir)
        return []

    matching_dirs = []
    prefix = f"{experiment_tag}_"

    for directory in dataset_dir.iterdir():
        if not directory.is_dir():
            continue

        if directory.name.startswith(prefix):
            matching_dirs.append(directory)

    matching_dirs.sort()  # Sort for consistent ordering

    logger.info("Found %d matching experiment directory(ies):", len(matching_dirs))
    for idx, directory in enumerate(matching_dirs):
        # Extract timestamp or distinguishing part from directory name for easier identification
        dir_parts = directory.name.split("_")
        timestamp_part = "_".join(dir_parts[-3:]) if len(dir_parts) >= 3 else directory.name[-30:]
        logger.info("  [%d] %s | Timestamp: %s | Path: %s", idx, directory.name, timestamp_part, directory)

    return matching_dirs


def load_dataframe_from_directory(directory: Path, df_name: str) -> pd.DataFrame | None:
    """
    Load a specific dataframe CSV from an experiment directory.

    Args:
        directory: Path to experiment directory
        df_name: Name of dataframe (e.g., "result_df", "summary_df")
                  Will look for {df_name}.csv

    Returns:
        DataFrame if found, None otherwise
    """
    csv_path = directory / f"{df_name}.csv"
    if not csv_path.exists():
        logger.warning("Dataframe '%s' not found in %s", df_name, directory.name)
        return None

    try:
        df = pd.read_csv(csv_path)
        logger.info("Loaded %s from %s: %d rows", df_name, directory.name, len(df))
        return df
    except Exception as e:
        logger.error("Failed to load %s from %s: %s", df_name, directory.name, e)
        return None


def compile_dataframes_by_tag(
    experiment_tag: str,
    df_name: str,
    dataset_name: str = "CIFAR-10",
    results_dir: Path | None = None,
    generate_plots: bool = True,
    output_dir: Path | None = None,
    hue_by: str = "verifier",
    custom_labels: list[str] | None = None,
    custom_colors: list[str] | None = None,
    explicit_dirs: list[Path] | None = None,
) -> tuple[list[pd.DataFrame], list[Path]]:
    """
    Compile dataframes from all experiments matching an experiment_tag prefix.

    Args:
        experiment_tag: Experiment tag prefix (e.g., "base_predict", "sigma_sweep")
        df_name: Name of dataframe to compile (e.g., "result_df", "summary_df")
        dataset_name: Dataset name (default: "CIFAR-10")
        results_dir: Optional custom results directory. If None, uses get_results_dir()
        generate_plots: Whether to generate comparison plots (default: True)
        output_dir: Output directory for plots/compiled CSV. If None, uses results_dir/plots/<experiment_tag>
        hue_by: Column to use for hue/grouping in plots: 'network' or 'verifier'
        custom_labels: Optional custom labels for the hue column
        custom_colors: Optional custom colors for the hue values

    Returns:
        Tuple of (list of dataframes, list of source directories)
    """
    if results_dir is None:
        results_dir = get_results_dir(dataset_name)

    if output_dir is None:
        output_dir = results_dir / "plots" / experiment_tag
        logger.info("No output_dir specified, using default: %s", output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use explicit directories if provided, otherwise find automatically
    if explicit_dirs is not None:
        # Validate that explicit directories exist and match the tag
        matching_dirs = []
        prefix = f"{experiment_tag}_"
        for exp_dir in explicit_dirs:
            exp_dir = Path(exp_dir)
            if not exp_dir.exists():
                logger.warning("Explicit directory does not exist: %s", exp_dir)
                continue
            if not exp_dir.name.startswith(prefix):
                logger.warning(
                    "Explicit directory '%s' does not match experiment_tag prefix '%s', skipping",
                    exp_dir.name,
                    prefix,
                )
                continue
            matching_dirs.append(exp_dir)
        logger.info("Using %d explicitly specified directory(ies)", len(matching_dirs))
        # Log the explicit directories with their order
        for idx, directory in enumerate(matching_dirs):
            dir_parts = directory.name.split("_")
            timestamp_part = "_".join(dir_parts[-3:]) if len(dir_parts) >= 3 else directory.name[-30:]
            logger.info("  [%d] %s | Timestamp: %s | Path: %s", idx, directory.name, timestamp_part, directory)
    else:
        matching_dirs = find_experiment_directories(results_dir, experiment_tag, dataset_name)

    if not matching_dirs:
        logger.error("No matching experiment directories found for tag '%s'", experiment_tag)
        return [], []

    dataframes = []
    loaded_dirs = []

    for directory in matching_dirs:
        df = load_dataframe_from_directory(directory, df_name)
        if df is not None:
            dataframes.append(df)
            loaded_dirs.append(directory)

    if not dataframes:
        logger.error("No dataframes loaded. Exiting.")
        return [], []

    logger.info("Successfully loaded %d dataframe(s)", len(dataframes))

    # Log the mapping between directories and labels for transparency
    if custom_labels is not None and len(custom_labels) == len(loaded_dirs):
        logger.info("Label-to-directory mapping:")
        for idx, (label, directory) in enumerate(zip(custom_labels, loaded_dirs, strict=True)):
            dir_parts = directory.name.split("_")
            timestamp_part = "_".join(dir_parts[-3:]) if len(dir_parts) >= 3 else directory.name[-30:]
            logger.info("  [%d] '%s' -> %s | Timestamp: %s", idx, label, directory.name, timestamp_part)
    elif custom_labels is not None:
        logger.warning("Custom labels provided but count doesn't match directories. Mapping may be incorrect.")

    # Apply custom labels if provided
    if custom_labels is not None:
        if len(custom_labels) != len(dataframes):
            logger.warning(
                "Number of custom labels (%d) does not match number of dataframes (%d). "
                "Labels may not be applied correctly.",
                len(custom_labels),
                len(dataframes),
            )
        else:
            for idx, df in enumerate(dataframes):
                label = custom_labels[idx]
                hue_col = hue_by
                if hue_col not in df.columns:
                    logger.warning(
                        "Hue column '%s' not found in dataframe %d; creating it with custom label '%s'",
                        hue_col,
                        idx,
                        label,
                    )
                df[hue_col] = label
            logger.info("Applied custom labels to dataframes")

    logger.info("Generating comparison plots for %d dataset(s)", len(dataframes))
    try:
        # Build color mapping if custom colors are provided
        custom_color_map = None
        if custom_colors is not None:
            if custom_labels is not None:
                labels = custom_labels
            else:
                # Extract labels from dataframes
                labels = []
                for df in dataframes:
                    hue_col = hue_by
                    if hue_col in df.columns:
                        unique_values = df[hue_col].unique()
                        if len(unique_values) > 0:
                            labels.append(str(unique_values[0]))
                        else:
                            labels.append(f"unknown_{len(labels)}")
                    else:
                        labels.append(f"unknown_{len(labels)}")
            custom_color_map = dict(zip(labels, custom_colors, strict=True))
            logger.info("Using custom color mapping: %s", custom_color_map)

        plot_paths = generate_verifier_comparison_plots(
            dataframes=dataframes,
            dataset_name=dataset_name,
            output_dir=output_dir,
            hue_by=hue_by,
            custom_colors=custom_color_map,
            experiment_dirs=loaded_dirs,
            custom_labels=custom_labels,
            concatenated_df_filename=f"{experiment_tag}_{df_name}_compiled",
        )
        logger.info("Successfully generated all plots:")
        for plot_name, plot_path in plot_paths.items():
            logger.info("  %s: %s", plot_name, plot_path)
    except Exception as e:
        logger.error("Failed to generate plots: %s", e)
        raise

    return dataframes, loaded_dirs


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compile dataframes from experiments with the same experiment_tag prefix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile result_df from all base_predict experiments
  python compile_by_tag.py base_predict result_df --dataset CIFAR-10

  # Compile summary_df and generate plots
  python compile_by_tag.py sigma_sweep summary_df --output-dir ./plots

  # Compile with custom labels and colors
  python compile_by_tag.py base_predict result_df \\
      --custom-labels "ViT-B/16 sigma=0.25" "ViT-B/16 sigma=0.5" "ViT-B/16 sigma=1.0" \\
      --custom-colors "#d7bde2" "#bb8fce" "#7d3c98"
        """,
    )
    parser.add_argument(
        "experiment_tag",
        type=str,
        help="Experiment tag prefix to search for (e.g., 'base_predict', 'sigma_sweep')",
    )
    parser.add_argument(
        "df_name",
        type=str,
        help="Name of dataframe to compile (e.g., 'result_df', 'summary_df', 'misclassified_df')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR-10",
        help="Dataset name (default: CIFAR-10)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Custom results directory. If not specified, uses get_results_dir(dataset_name)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating comparison plots (plots are generated by default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for compiled CSV and plots. If not specified, uses RESULTS_DIR/plots/<experiment_tag>",
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
            "Optional custom labels for the hue column. "
            "If provided, must have the same length as the number of matching experiments."
        ),
    )
    parser.add_argument(
        "--custom-colors",
        nargs="*",
        type=str,
        default=None,
        help=(
            "Optional custom colors for the hue values. "
            "If provided, must have the same length as --custom-labels (or number of experiments if no custom labels)."
        ),
    )
    parser.add_argument(
        "--list-dirs",
        action="store_true",
        help=(
            "List all matching directories with their indices and exit "
            "(useful for verifying order before providing labels)"
        ),
    )
    parser.add_argument(
        "--explicit-dirs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Explicitly specify experiment directories in the desired order. "
            "If provided, only these directories will be used (must match experiment_tag prefix). "
            "Useful when directories have very similar names."
        ),
    )

    args = parser.parse_args()

    # Handle --list-dirs option
    if args.list_dirs:
        results_dir = get_results_dir(args.dataset) if args.results_dir is None else args.results_dir

        matching_dirs = find_experiment_directories(results_dir, args.experiment_tag, args.dataset)
        if not matching_dirs:
            logger.info("No matching directories found for tag '%s'", args.experiment_tag)
            sys.exit(0)

        logger.info("\n" + "=" * 80)
        logger.info("Matching directories (in alphabetical order):")
        for idx, directory in enumerate(matching_dirs):
            dir_parts = directory.name.split("_")
            timestamp_part = "_".join(dir_parts[-3:]) if len(dir_parts) >= 3 else directory.name[-30:]
            logger.info("  [%d] %s | Timestamp: %s | Path: %s", idx, directory.name, timestamp_part, directory)
        sys.exit(0)

    # Validate custom labels/colors if provided
    if (
        args.custom_labels is not None
        and args.custom_colors is not None
        and len(args.custom_labels) != len(args.custom_colors)
    ):
        logger.error(
            "Number of --custom-labels (%d) must match number of --custom-colors (%d)",
            len(args.custom_labels),
            len(args.custom_colors),
        )
        sys.exit(1)

    try:
        dataframes, loaded_dirs = compile_dataframes_by_tag(
            experiment_tag=args.experiment_tag,
            df_name=args.df_name,
            dataset_name=args.dataset,
            results_dir=args.results_dir,
            generate_plots=not args.no_plots,
            output_dir=args.output_dir,
            hue_by=args.hue_by,
            custom_labels=args.custom_labels,
            custom_colors=args.custom_colors,
            explicit_dirs=args.explicit_dirs,
        )

        if args.custom_labels is not None and len(args.custom_labels) != len(dataframes):
            logger.warning(
                "Number of --custom-labels (%d) does not match number of loaded dataframes (%d). "
                "Labels may not be applied correctly.",
                len(args.custom_labels),
                len(dataframes),
            )

        logger.info("Compilation complete. Processed %d experiment(s):", len(loaded_dirs))
        for directory in loaded_dirs:
            logger.info("  - %s", directory.name)

    except Exception as e:
        logger.error("Failed to compile dataframes: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
