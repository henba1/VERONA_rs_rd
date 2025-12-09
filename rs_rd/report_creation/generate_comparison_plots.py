import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rs_rd_research.paths import get_results_dir

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
    Generate verifier comparison plots from multiple dataframes and save them.

    Args:
        dataframes: List of pandas DataFrames containing results from different verifiers
        dataset_name: Name of the dataset (used for directory structure)
        output_dir: Optional custom output directory. If None, uses RESULTS_DIR/dataset_name
        hue_by: Column to use for hue/grouping in plots: 'network' or 'verifier' (default: verifier)

    Returns:
        Dictionary mapping plot names to their file paths, and concatenated dataframe path
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
    logger.info(f"Saved concatenated dataframe to {df_path}")

    # Choose the appropriate report creator based on hue_by parameter
    if hue_by == "network":
        logger.info(f"Generating network comparison plots for {dataset_name}")
        report_creator = ReportCreator(concatenated_df)
    elif hue_by == "verifier":
        logger.info(f"Generating verifier comparison plots for {dataset_name}")
        report_creator = ReportCreatorVerifier(dataframes)
    else:
        raise ValueError(f"Invalid hue_by value: {hue_by}. Must be 'network' or 'verifier'.")

    plot_paths = {}

    # Hist plot
    hist_fig = report_creator.create_hist_figure()
    hist_path = output_dir / f"{base_filename}_histogram.png"
    hist_fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    plot_paths["histogram"] = hist_path
    logger.info(f"Saved histogram plot to {hist_path}")

    # Box plot
    box_fig = report_creator.create_box_figure()
    box_path = output_dir / f"{base_filename}_boxplot.png"
    box_fig.savefig(box_path, dpi=300, bbox_inches="tight")
    plot_paths["boxplot"] = box_path
    logger.info(f"Saved boxplot to {box_path}")

    # KDE plot
    kde_fig = report_creator.create_kde_figure()
    kde_path = output_dir / f"{base_filename}_kde.png"
    kde_fig.savefig(kde_path, dpi=300, bbox_inches="tight")
    plot_paths["kde"] = kde_path
    logger.info(f"Saved KDE plot to {kde_path}")

    # ECDF plot
    ecdf_fig = report_creator.create_ecdf_figure()
    ecdf_path = output_dir / f"{base_filename}_ecdf.png"
    ecdf_fig.savefig(ecdf_path, dpi=300, bbox_inches="tight")
    plot_paths["ecdf"] = ecdf_path
    logger.info(f"Saved ECDF plot to {ecdf_path}")

    plot_paths["concatenated_dataframe"] = df_path
    logger.info(f"All plots and concatenated dataframe saved to {output_dir}")
    return plot_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results_dir = get_results_dir()
    dataset_name = "CIFAR-10"
    # result_dir_1 = Path(results_dir) / "conv_big_standard_0.5_0.001_100_1000000_200"
    # result_dir_1 = Path(results_dir) /
    # "verona_rs_rd_pgd_l2_CIFAR-10_300_sample_correct_True_sample_stratified_True/
    # pgd_l2_18-11-2025+14_14/results"
    result_dir_1 = (
        Path(results_dir)
        / "verona_rs_rd_pgd_linf_CIFAR-10_300_sample_correct_True_sample_stratified_True"
        / "pgd_linf_18-11-2025+15_51"
        / "results"
    )

    # result_dir_2 = Path(results_dir) / "conv_big_standard_0.5_0.001_100_100000_200"
    # result_dir_3 = Path(results_dir) / "conv_big_standard_0.5_0.0001_100_100000_200"
    # result_dir_4 = Path(results_dir) / "conv_big_standard_0.25_0.0001_100_100000_200"
    result_dir_2 = Path(results_dir) / "conv_big_standard_0.25_0.001_100_100000_200"

    # assume only one csv file in each directory
    def load_single_csv_in_dir(directory):
        csv_files = list(directory.glob("*.csv"))
        if len(csv_files) != 1:
            raise FileNotFoundError(f"Expected exactly one CSV file in {directory}, found {len(csv_files)}")
            # in this case, the run was terminated early,
        return pd.read_csv(csv_files[0])

    df1 = load_single_csv_in_dir(result_dir_1)
    df2 = load_single_csv_in_dir(result_dir_2)
    # df3 = load_single_csv_in_dir(result_dir_3)
    # df4 = load_single_csv_in_dir(result_dir_4)
    # df5 = load_single_csv_in_dir(result_dir_5)

    dfs = [df1, df2]
    plot_paths = generate_verifier_comparison_plots(dataframes=dfs, dataset_name="CIFAR-10")

    for plot_name, plot_path in plot_paths.items():
        print(f"{plot_name}: {plot_path}")
