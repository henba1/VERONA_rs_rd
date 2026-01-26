"""
Plotting function for RS attack dataframes.

This module provides functionality to plot randomized smoothing (RS) lower and upper bounds
from attack CSV files, treating them as separate verifiers for comparison.
"""

import logging
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from agg_report_creator import generate_verifier_comparison_plots

matplotlib.use("Agg")
sns.set_style("darkgrid")
logger = logging.getLogger(__name__)


def extract_model_name(network_str: str) -> str:
    """
    Extract model name from network string.

    Args:
        network_str: Network string like "RS_cifar10_convlarge_cifar10_uncond_50M_500K_sigma_0.25"

    Returns:
        Extracted model name (e.g., "convlarge" or "conv_large" if pattern matches)
    """
    # Pattern: RS_<dataset>_<model>_...
    # Try to extract the model name part
    match = re.match(r"RS_[^_]+_(.+?)_(?:cifar10|uncond)", network_str)
    if match:
        model_name = match.group(1)
        # Convert common patterns: "convlarge" -> "conv_large"
        if model_name == "convlarge":
            return "conv_large"
        return model_name
    # Fallback: try to extract anything after RS_<dataset>_
    parts = network_str.split("_")
    if len(parts) >= 3 and parts[0] == "RS":
        model_name = parts[2]
        if model_name == "convlarge":
            return "conv_large"
        return model_name
    return "unknown_model"


def prepare_rs_attack_dataframes(
    csv_path: str | Path,
    model_name: str | None = None,
    keep_image_info: bool = False,
) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    Prepare dataframes from RS attack CSV for plotting.

    Args:
        csv_path: Path to CSV file with RS attack results
        model_name: Optional model name. If None, extracts from network column
        keep_image_info: If True, also returns the original dataframe with image_id and cert_pred

    Returns:
        List of two dataframes: [RS_lower_bound_df, EOTPGD_upper_bound_df]
        If keep_image_info=True, also returns the original dataframe as third element
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded CSV with %d rows", len(df))

    # Extract model name if not provided
    if model_name is None:
        if "network" not in df.columns:
            raise ValueError("CSV must contain 'network' column to extract model name")
        # Get first network value to extract model name
        first_network = df["network"].iloc[0]
        extracted_name = extract_model_name(str(first_network))
        model_name = extracted_name
        logger.info("Extracted model name: %s", model_name)

    # Check required columns
    required_cols = ["cert_radius_l2", "min_adv_radius_l2"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    # Create RS lower bound dataframe (certified radius)
    rs_df = df[["cert_radius_l2"]].copy()
    rs_df = rs_df.rename(columns={"cert_radius_l2": "epsilon_value"})
    # Convert to numeric, converting empty strings to NaN
    rs_df["epsilon_value"] = pd.to_numeric(rs_df["epsilon_value"], errors="coerce")
    rs_df["verifier"] = f"RS_{model_name}"
    # Filter out NaN values
    rs_df = rs_df.dropna(subset=["epsilon_value"])

    # Create EOTPGD upper bound dataframe (adversarial radius)
    eotpgd_df = df[["min_adv_radius_l2"]].copy()
    eotpgd_df = eotpgd_df.rename(columns={"min_adv_radius_l2": "epsilon_value"})
    # Convert to numeric, converting empty strings to NaN
    eotpgd_df["epsilon_value"] = pd.to_numeric(eotpgd_df["epsilon_value"], errors="coerce")
    eotpgd_df["verifier"] = "EOTPGD"
    # Filter out NaN values
    eotpgd_df = eotpgd_df.dropna(subset=["epsilon_value"])

    logger.info("Created RS dataframe: %d rows", len(rs_df))
    logger.info("Created EOTPGD dataframe: %d rows", len(eotpgd_df))

    if keep_image_info:
        return [rs_df, eotpgd_df], df
    return [rs_df, eotpgd_df]


def plot_rs_attacks(
    csv_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
    dataset_name: str = "CIFAR-10",
    custom_colors: dict[str, str] | None = None,
) -> dict[str, Path]:
    """
    Plot RS lower and upper bounds as separate verifiers.

    Args:
        csv_path: Path to CSV file with RS attack results
        output_dir: Output directory for plots
        model_name: Optional model name. If None, extracts from network column
        dataset_name: Dataset name (default: "CIFAR-10")
        custom_colors: Optional custom color mapping for verifiers

    Returns:
        Dictionary mapping plot names to their file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataframes
    dataframes = prepare_rs_attack_dataframes(csv_path, model_name=model_name)

    # Generate plots using the existing infrastructure
    plot_paths = generate_verifier_comparison_plots(
        dataframes=dataframes,
        dataset_name=dataset_name,
        output_dir=output_dir,
        hue_by="verifier",
        custom_colors=custom_colors,
        concatenated_df_filename="rs_attack_comparison",
        save_dataframe=True,
    )

    # Generate paired comparison plot
    try:
        paired_plot_path = create_paired_comparison_plot(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name=model_name,
            custom_colors=custom_colors,
        )
        plot_paths["paired_comparison"] = paired_plot_path
    except Exception as e:
        logger.warning("Failed to generate paired comparison plot: %s", e)

    logger.info("Generated plots in %s", output_dir)
    return plot_paths


def create_paired_comparison_plot(
    csv_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
    custom_colors: dict[str, str] | None = None,
) -> Path:
    """
    Create a paired comparison plot showing RS and EOTPGD epsilon values
    sorted by cert_pred, where each y-position corresponds to the same image_id.

    Args:
        csv_path: Path to CSV file with RS attack results
        output_dir: Output directory for the plot
        model_name: Optional model name. If None, extracts from network column
        custom_colors: Optional custom color mapping for verifiers

    Returns:
        Path to saved plot file
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Extract model name if not provided
    if model_name is None:
        if "network" not in df.columns:
            raise ValueError("CSV must contain 'network' column to extract model name")
        first_network = df["network"].iloc[0]
        model_name = extract_model_name(str(first_network))

    # Check required columns
    required_cols = ["image_id", "cert_pred", "cert_radius_l2", "min_adv_radius_l2"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns for paired plot: {missing_cols}")

    # Create dataframe with both radii and image info
    plot_df = df[["image_id", "cert_pred", "cert_radius_l2", "min_adv_radius_l2"]].copy()

    # Convert to numeric
    plot_df["cert_radius_l2"] = pd.to_numeric(plot_df["cert_radius_l2"], errors="coerce")
    plot_df["min_adv_radius_l2"] = pd.to_numeric(plot_df["min_adv_radius_l2"], errors="coerce")
    plot_df["cert_pred"] = pd.to_numeric(plot_df["cert_pred"], errors="coerce")

    # Filter out rows where either radius is NaN
    plot_df = plot_df.dropna(subset=["cert_radius_l2", "min_adv_radius_l2", "cert_pred"])

    # Sort by cert_pred (smallest to largest) - ensure numeric sorting
    plot_df = plot_df.sort_values("cert_pred", ascending=True, na_position="last").reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare colors
    rs_color = custom_colors.get(f"RS_{model_name}") if custom_colors else None
    eotpgd_color = custom_colors.get("EOTPGD") if custom_colors else None

    # Create y-axis: sorted index (0 to n-1)
    y_positions = range(len(plot_df))

    # Plot RS lower bound
    ax.plot(
        plot_df["cert_radius_l2"],
        y_positions,
        label=f"RS_{model_name}",
        color=rs_color,
        linewidth=1.5,
        alpha=0.8,
    )

    # Plot EOTPGD upper bound
    ax.plot(
        plot_df["min_adv_radius_l2"],
        y_positions,
        label="EOTPGD",
        color=eotpgd_color,
        linewidth=1.5,
        alpha=0.8,
    )

    # Customize plot
    ax.set_xlabel("Epsilon (radius)", fontsize=12)
    ax.set_ylabel("Image index (sorted by cert_pred)", fontsize=12)
    ax.set_title("Paired Comparison: RS Lower Bound vs EOTPGD Upper Bound", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "rs_attack_comparison_paired.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved paired comparison plot to %s", plot_path)

    return plot_path
