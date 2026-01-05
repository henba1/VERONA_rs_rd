import logging
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ada_verona import get_results_dir
from ada_verona.analysis.report_creator import ReportCreator
from ada_verona.analysis.report_creator_verifier import ReportCreatorVerifier

matplotlib.use("Agg")
sns.set_style("darkgrid")
logger = logging.getLogger(__name__)


def create_classifier_accuracy_barplot(
    experiment_dirs: list[Path],
    output_dir: Path,
    base_filename: str,
    custom_colors: dict[str, str] | None = None,
    custom_labels: list[str] | None = None,
) -> Path | None:
    """
    Create a bar plot comparing classifier accuracies from summary_df.csv files.

    Args:
        experiment_dirs: List of paths to experiment directories (one per classifier)
        output_dir: Output directory for the plot
        base_filename: Base filename for the plot
        custom_colors: Optional custom color mapping for classifiers
        custom_labels: Optional custom labels for classifiers (must match order of experiment_dirs)

    Returns:
        Path to saved plot file, or None if no summary files found
    """
    accuracy_data = []

    for idx, exp_dir in enumerate(experiment_dirs):
        summary_csv = exp_dir / "summary_df.csv"
        if not summary_csv.exists():
            logger.warning("summary_df.csv not found in %s, skipping", exp_dir.name)
            continue

        try:
            summary_df = pd.read_csv(summary_csv)
            if summary_df.empty:
                logger.warning("Empty summary_df.csv in %s, skipping", exp_dir.name)
                continue

            # Extract accuracy and model name (assuming single row in summary)
            row = summary_df.iloc[0]
            overall_accuracy = float(row.get("overall_accuracy", 0.0))
            accuracy_without_abstain = float(row.get("accuracy_without_abstain", overall_accuracy))

            # Use custom label if provided, otherwise use model_name from summary
            if custom_labels is not None and idx < len(custom_labels):
                classifier_name = custom_labels[idx]
            else:
                classifier_name = str(row.get("model_name", "unknown"))

            accuracy_data.append(
                {
                    "classifier_name": classifier_name,
                    "overall_accuracy": overall_accuracy,
                    "accuracy_without_abstain": accuracy_without_abstain,
                }
            )
        except Exception as e:
            logger.warning("Error reading summary_df.csv from %s: %s", exp_dir.name, e)
            continue

    if not accuracy_data:
        logger.warning("No valid summary data found for accuracy comparison plot")
        return None

    # Create DataFrame from collected data
    acc_df = pd.DataFrame(accuracy_data)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by accuracy for better visualization
    acc_df = acc_df.sort_values("overall_accuracy", ascending=False)

    # Use custom colors if provided and classifier names match, otherwise use default palette
    colors = None
    color_map = {}
    if custom_colors is not None:
        color_list = []
        all_matched = True
        for classifier_name in acc_df["classifier_name"]:
            color = custom_colors.get(classifier_name)
            if color is None:
                all_matched = False
                break
            color_list.append(color)
            color_map[classifier_name] = color

        if all_matched and len(color_list) == len(acc_df):
            colors = color_list

    # If no custom colors matched, use default palette and create color map for legend
    if colors is None:
        default_palette = sns.color_palette("husl", len(acc_df))
        colors = default_palette
        color_map = {name: color for name, color in zip(acc_df["classifier_name"], colors, strict=True)}

    # Create bar plot with colors
    ax.bar(
        range(len(acc_df)),
        acc_df["overall_accuracy"] * 100,  # Convert to percentage
        color=colors,
    )

    # Customize plot
    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks([])  # Remove x-axis ticks since we'll use legend
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (_idx, row) in enumerate(acc_df.iterrows()):
        height = row["overall_accuracy"] * 100
        ax.text(
            i,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Create legend with colored bars
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[name], label=name) for name in acc_df["classifier_name"]
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    plt.tight_layout()

    # Save plot
    barplot_path = output_dir / f"{base_filename}_classifier_accuracy.png"
    fig.savefig(barplot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved classifier accuracy bar plot to %s", barplot_path)

    return barplot_path


def generate_verifier_comparison_plots(
    dataframes: list[pd.DataFrame],
    dataset_name: str,
    output_dir: str | None = None,
    hue_by: str = "verifier",
    custom_colors: dict[str, str] | None = None,
    experiment_dirs: list[Path] | None = None,
    custom_labels: list[str] | None = None,
    concatenated_df_filename: str | None = None,
) -> dict[str, Path]:
    """
    Generate verifier or network comparison plots from multiple dataframes and save them.

    Args:
        dataframes: List of pandas DataFrames containing results.
        dataset_name: Name of the dataset (used for directory structure/logging).
        output_dir: Optional custom output directory. If None, uses RESULTS_DIR/dataset_name.
        hue_by: Column to use for hue/grouping in plots: 'network' or 'verifier' (default: 'verifier').
        concatenated_df_filename: Optional custom filename for concatenated dataframe (without .csv extension).
                                 If None, uses default timestamp-based filename.

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
    if concatenated_df_filename is not None:
        df_path = output_dir / f"{concatenated_df_filename}.csv"
    else:
        df_path = output_dir / f"{base_filename}_concatenated_results.csv"
    concatenated_df.to_csv(df_path, index=False)
    logger.info("Saved concatenated dataframe to %s", df_path)

    # Choose the appropriate report creator based on hue_by parameter
    if hue_by == "network":
        logger.info("Generating network comparison plots for %s", dataset_name)
        report_creator = ReportCreator(concatenated_df, custom_colors=custom_colors)
    elif hue_by == "verifier":
        logger.info("Generating verifier comparison plots for %s", dataset_name)
        report_creator = ReportCreatorVerifier(dataframes, custom_colors=custom_colors)
    else:
        raise ValueError(f"Invalid hue_by value: {hue_by}. Must be 'network' or 'verifier'.")

    plot_paths: dict[str, Path] = {}

    # Hist plot
    try:
        hist_fig = report_creator.create_hist_figure()
        hist_path = output_dir / f"{base_filename}_histogram.png"
        hist_fig.savefig(hist_path, dpi=300, bbox_inches="tight")
        plot_paths["histogram"] = hist_path
        logger.info("Saved histogram plot to %s", hist_path)
    except Exception as e:
        logger.warning("Failed to generate histogram plot: %s", e)

    # Box plot
    try:
        box_fig = report_creator.create_box_figure()
        box_path = output_dir / f"{base_filename}_boxplot.png"
        box_fig.savefig(box_path, dpi=300, bbox_inches="tight")
        plot_paths["boxplot"] = box_path
        logger.info("Saved boxplot to %s", box_path)
    except Exception as e:
        logger.warning("Failed to generate boxplot: %s", e)

    # KDE plot
    try:
        kde_fig = report_creator.create_kde_figure()
        kde_path = output_dir / f"{base_filename}_kde.png"
        kde_fig.savefig(kde_path, dpi=300, bbox_inches="tight")
        plot_paths["kde"] = kde_path
        logger.info("Saved KDE plot to %s", kde_path)
    except Exception as e:
        logger.warning("Failed to generate KDE plot: %s", e)
        logger.info(
            "This can happen when all data points have the same value "
            "(e.g., all zeros for radius in base_predict experiments)"
        )

    # ECDF plot
    try:
        ecdf_fig = report_creator.create_ecdf_figure()
        ecdf_path = output_dir / f"{base_filename}_ecdf.png"
        ecdf_fig.savefig(ecdf_path, dpi=300, bbox_inches="tight")
        plot_paths["ecdf"] = ecdf_path
        logger.info("Saved ECDF plot to %s", ecdf_path)
    except Exception as e:
        logger.warning("Failed to generate ECDF plot: %s", e)

    # Anne plot (if supported by report creator)
    if hasattr(report_creator, "create_anneplot"):
        try:
            anne_ax = report_creator.create_anneplot()
            anne_fig = anne_ax.get_figure()
            anne_path = output_dir / f"{base_filename}_anneplot.png"
            anne_fig.savefig(anne_path, dpi=300, bbox_inches="tight")
            plot_paths["anneplot"] = anne_path
            logger.info("Saved anneplot to %s", anne_path)
        except Exception as e:
            logger.warning("Failed to generate anneplot: %s", e)

    # Classifier accuracy bar plot
    if experiment_dirs is not None:
        accuracy_plot_path = create_classifier_accuracy_barplot(
            experiment_dirs=experiment_dirs,
            output_dir=output_dir,
            base_filename=base_filename,
            custom_colors=custom_colors,
            custom_labels=custom_labels,
        )
        if accuracy_plot_path is not None:
            plot_paths["classifier_accuracy"] = accuracy_plot_path

    plot_paths["concatenated_dataframe"] = df_path
    logger.info("All plots and concatenated dataframe saved to %s", output_dir)
    return plot_paths
