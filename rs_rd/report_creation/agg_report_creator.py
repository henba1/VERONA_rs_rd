#!/usr/bin/env python3
"""
Core utilities to generate verifier/network comparison plots from one or more result DataFrames.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from paths import get_results_dir 

from ada_verona.analysis.report_creator import ReportCreator
from ada_verona.analysis.report_creator_verifier import ReportCreatorVerifier

logger = logging.getLogger(__name__)


def generate_verifier_comparison_plots(
    dataframes: list[pd.DataFrame],
    dataset_name: str,
    output_dir: str | None = None,
    hue_by: str = "verifier",
) -> dict[str, Path]:
    """
    Generate verifier or network comparison plots from multiple dataframes and save them.

    Args:
        dataframes: List of pandas DataFrames containing results.
        dataset_name: Name of the dataset (used for directory structure/logging).
        output_dir: Optional custom output directory. If None, uses RESULTS_DIR/dataset_name.
        hue_by: Column to use for hue/grouping in plots: 'network' or 'verifier' (default: 'verifier').

    Returns:
        Dictionary mapping plot names to their file paths, plus a 'concatenated_dataframe' CSV path.
    """

    if output_dir is None:
        results_dir = get_results_dir()
        output_dir = Path(results_dir)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"compiled_results_{timestamp}"

    # Concatenate dataframes and save
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    df_path = output_dir / f"{base_filename}_concatenated_results.csv"
    concatenated_df.to_csv(df_path, index=False)
    logger.info("Saved concatenated dataframe to %s", df_path)

    # Choose the appropriate report creator based on hue_by parameter
    if hue_by == "network":
        logger.info("Generating network comparison plots for %s", dataset_name)
        report_creator = ReportCreator(concatenated_df)
    elif hue_by == "verifier":
        logger.info("Generating verifier comparison plots for %s", dataset_name)
        report_creator = ReportCreatorVerifier(dataframes)
    else:
        raise ValueError(f"Invalid hue_by value: {hue_by}. Must be 'network' or 'verifier'.")

    plot_paths: dict[str, Path] = {}

    # Hist plot
    hist_fig = report_creator.create_hist_figure()
    hist_path = output_dir / f"{base_filename}_histogram.png"
    hist_fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    plot_paths["histogram"] = hist_path
    logger.info("Saved histogram plot to %s", hist_path)

    # Box plot
    box_fig = report_creator.create_box_figure()
    box_path = output_dir / f"{base_filename}_boxplot.png"
    box_fig.savefig(box_path, dpi=300, bbox_inches="tight")
    plot_paths["boxplot"] = box_path
    logger.info("Saved boxplot to %s", box_path)

    # KDE plot
    kde_fig = report_creator.create_kde_figure()
    kde_path = output_dir / f"{base_filename}_kde.png"
    kde_fig.savefig(kde_path, dpi=300, bbox_inches="tight")
    plot_paths["kde"] = kde_path
    logger.info("Saved KDE plot to %s", kde_path)

    # ECDF plot
    ecdf_fig = report_creator.create_ecdf_figure()
    ecdf_path = output_dir / f"{base_filename}_ecdf.png"
    ecdf_fig.savefig(ecdf_path, dpi=300, bbox_inches="tight")
    plot_paths["ecdf"] = ecdf_path
    logger.info("Saved ECDF plot to %s", ecdf_path)

    # Anne plot (if supported by report creator)
    if hasattr(report_creator, "create_anneplot"):
        anne_ax = report_creator.create_anneplot()
        anne_fig = anne_ax.get_figure()
        anne_path = output_dir / f"{base_filename}_anneplot.png"
        anne_fig.savefig(anne_path, dpi=300, bbox_inches="tight")
        plot_paths["anneplot"] = anne_path
        logger.info("Saved anneplot to %s", anne_path)

    plot_paths["concatenated_dataframe"] = df_path
    logger.info("All plots and concatenated dataframe saved to %s", output_dir)
    return plot_paths


