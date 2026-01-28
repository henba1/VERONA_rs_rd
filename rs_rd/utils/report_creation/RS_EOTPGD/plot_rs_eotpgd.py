from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/gpfs/work2/0/prjs1681/runs/results/figures/3_RS_EOTPGD")

_REPORT_CREATION_DIR = Path(__file__).resolve().parents[1]
if str(_REPORT_CREATION_DIR) not in sys.path:
    sys.path.insert(0, str(_REPORT_CREATION_DIR))

from agg_report_creator import generate_verifier_comparison_plots  # noqa: E402


def _write_inputs_file(output_dir: Path, csv_path: str | Path) -> Path:
    path = output_dir / "inputs.txt"
    content = "\n".join(
        [
            f"timestamp={datetime.now().isoformat(timespec='seconds')}",
            f"csv_path={Path(csv_path).resolve()}",
            "",
        ]
    )
    path.write_text(content)
    return path


def _paired_violet_colors(rs_verifier: str) -> dict[str, str]:
    """
    Color convention: lb is darker, ub is lighter.
    For RS vs EOTPGD we use the Paired palette's violet pair.
    """
    palette = sns.color_palette("Paired", 12).as_hex()
    light_violet, violet = palette[8], palette[9]
    return {rs_verifier: violet, "EOTPGD": light_violet}


def extract_model_name(network_str: str) -> str:
    match = re.match(r"RS_[^_]+_(.+?)_(?:cifar10|uncond)", str(network_str))
    if match:
        model_name = match.group(1)
        return "conv_large" if model_name == "convlarge" else model_name

    parts = str(network_str).split("_")
    if len(parts) >= 3 and parts[0] == "RS":
        model_name = parts[2]
        return "conv_large" if model_name == "convlarge" else model_name

    return "unknown_model"


def extract_rs_hue_label(network_str: str) -> str:
    parts = str(network_str).split("_")
    if len(parts) >= 9:
        return f"{parts[2]}_{parts[7]}_{parts[8]}"
    return extract_model_name(network_str)


def prepare_rs_attack_dataframes(
    csv_path: str | Path,
    model_name: str | None = None,
    keep_image_info: bool = False,
) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], pd.DataFrame]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded CSV with %d rows", len(df))

    if model_name is None:
        if "network" not in df.columns:
            raise ValueError("CSV must contain 'network' column to extract RS label")
        model_name = extract_rs_hue_label(str(df["network"].iloc[0]))
        logger.info("Extracted RS hue label: %s", model_name)

    required_cols = ["cert_radius_l2", "min_adv_radius_l2"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    rs_df = df[["cert_radius_l2"]].rename(columns={"cert_radius_l2": "epsilon_value"})
    rs_df["epsilon_value"] = pd.to_numeric(rs_df["epsilon_value"], errors="coerce")
    rs_df["verifier"] = f"RS_{model_name}"
    rs_df = rs_df.dropna(subset=["epsilon_value"]).reset_index(drop=True)

    eotpgd_df = df[["min_adv_radius_l2"]].rename(columns={"min_adv_radius_l2": "epsilon_value"})
    eotpgd_df["epsilon_value"] = pd.to_numeric(eotpgd_df["epsilon_value"], errors="coerce")
    eotpgd_df["verifier"] = "EOTPGD"
    eotpgd_df = eotpgd_df.dropna(subset=["epsilon_value"]).reset_index(drop=True)

    if keep_image_info:
        return [rs_df, eotpgd_df], df
    return [rs_df, eotpgd_df]


def create_paired_ecdf_plot(
    csv_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
    custom_colors: dict[str, str] | None = None,
) -> Path:
    """
    "Paired ECDF" plot: sort by lb and show ub gap per instance.

    - x-axis: epsilon value
    - y-axis: fraction epsilon values found (based on lb ordering)
    - for each sample: draw a horizontal segment from lb -> ub
    """
    df = pd.read_csv(Path(csv_path))

    if model_name is None:
        if "network" not in df.columns:
            raise ValueError("CSV must contain 'network' column to extract RS label")
        model_name = extract_rs_hue_label(str(df["network"].iloc[0]))

    required_cols = ["cert_radius_l2", "min_adv_radius_l2"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns for paired ECDF plot: {missing_cols}")

    plot_df = df[["cert_radius_l2", "min_adv_radius_l2"]].copy()
    plot_df["cert_radius_l2"] = pd.to_numeric(plot_df["cert_radius_l2"], errors="coerce")
    plot_df["min_adv_radius_l2"] = pd.to_numeric(plot_df["min_adv_radius_l2"], errors="coerce")
    plot_df = (
        plot_df.dropna(subset=["cert_radius_l2", "min_adv_radius_l2"])
        .sort_values("cert_radius_l2")
        .reset_index(drop=True)
    )

    lb = plot_df["cert_radius_l2"].to_numpy()
    ub = plot_df["min_adv_radius_l2"].to_numpy()

    n = len(plot_df)
    y = np.zeros(n) if n <= 1 else np.linspace(0, 1, n)

    rs_verifier = f"RS_{model_name}"
    colors = _paired_violet_colors(rs_verifier)
    if custom_colors:
        colors.update(custom_colors)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(lb, y, color=colors.get(rs_verifier), linewidth=1.8, label=rs_verifier)
    ax.hlines(y, lb, ub, color=colors.get("EOTPGD"), alpha=0.35, linewidth=2.5, label="_nolegend_")
    ax.set_xlabel("Epsilon value")
    ax.set_ylabel("Fraction epsilon values found")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rs_eotpgd_paired_ecdf.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rs_eotpgd(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    dataset_name: str = "CIFAR-10",
    custom_colors: dict[str, str] | None = None,
) -> dict[str, Path]:
    output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_inputs_file(output_dir, csv_path)

    dataframes = prepare_rs_attack_dataframes(csv_path, model_name=model_name)
    rs_verifier = str(dataframes[0]["verifier"].iloc[0])

    colors = _paired_violet_colors(rs_verifier)
    if custom_colors:
        colors.update(custom_colors)

    plot_paths = generate_verifier_comparison_plots(
        dataframes=dataframes,
        dataset_name=dataset_name,
        output_dir=str(output_dir),
        hue_by="verifier",
        custom_colors=colors,
        concatenated_df_filename="rs_eotpgd_concatenated",
        save_dataframe=False,
    )

    # Replace the old "paired/index plot" with the paired ECDF gap plot.
    try:
        paired_ecdf_path = create_paired_ecdf_plot(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name=model_name,
            custom_colors=colors,
        )
        plot_paths["paired_ecdf"] = paired_ecdf_path
    except Exception as e:
        logger.warning("Failed to generate paired ECDF plot: %s", e)

    plot_paths["inputs"] = output_dir / "inputs.txt"
    return plot_paths
