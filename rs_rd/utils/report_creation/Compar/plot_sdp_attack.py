from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_palette(sns.color_palette("Paired"))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SdpAttackConfig:
    output_dir: Path = Path("/gpfs/work2/0/prjs1681/runs/results/figures/4_Compar")
    lb_verifier: str = "sdpcrown"


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_inputs_file(output_dir: Path, df_csv: str | Path) -> Path:
    path = output_dir / "inputs_sdp_attack.txt"
    content = "\n".join(
        [
            f"timestamp={datetime.now().isoformat(timespec='seconds')}",
            f"df_csv={Path(df_csv).resolve()}",
            "",
        ]
    )
    path.write_text(content)
    return path


def _choose_ub_verifier(df: pd.DataFrame, lb_verifier: str) -> str:
    if "verifier" not in df.columns:
        raise ValueError("df is missing required column: 'verifier'")
    candidates = df.loc[df["verifier"].astype(str) != lb_verifier, "verifier"].astype(str)
    if candidates.empty:
        raise ValueError(f"No verifier entries other than '{lb_verifier}'")
    counts = candidates.value_counts()
    if len(counts) > 1:
        logger.warning("Multiple non-%s verifiers found; using most frequent: %s", lb_verifier, counts.index[0])
    return str(counts.index[0])


def _default_colors(lb_label: str, ub_label: str) -> dict[str, str]:
    palette = sns.color_palette("Paired", 12).as_hex()
    light_blue, blue = palette[0], palette[1]
    # Convention: lb is darker, ub is lighter.
    return {lb_label: blue, ub_label: light_blue}


def _make_verifier_df(values: pd.Series, verifier_label: str) -> pd.DataFrame:
    out = pd.DataFrame({"epsilon_value": pd.to_numeric(values, errors="coerce")})
    out = out.dropna(subset=["epsilon_value"])
    out["verifier"] = verifier_label
    out["smallest_sat_value"] = out["epsilon_value"]
    return out


def _prepare_two_series(df_csv: str | Path, *, config: SdpAttackConfig) -> tuple[list[pd.DataFrame], str, str]:
    df = pd.read_csv(Path(df_csv))
    required = {"epsilon_value", "verifier"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    lb = config.lb_verifier
    ub = _choose_ub_verifier(df, lb)

    lb_vals = df.loc[df["verifier"].astype(str) == lb, "epsilon_value"]
    ub_vals = df.loc[df["verifier"].astype(str) == ub, "epsilon_value"]

    return [_make_verifier_df(lb_vals, lb), _make_verifier_df(ub_vals, ub)], lb, ub


def _pair_gap_df(df_csv: str | Path, lb: str, ub: str) -> pd.DataFrame:
    df = pd.read_csv(Path(df_csv))
    required = {"image_id", "verifier", "epsilon_value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"df is missing required columns for gap-ECDF: {missing}")

    df_lb = df.loc[df["verifier"].astype(str) == lb, ["image_id", "epsilon_value"]].rename(
        columns={"epsilon_value": "lb"}
    )
    df_ub = df.loc[df["verifier"].astype(str) == ub, ["image_id", "epsilon_value"]].rename(
        columns={"epsilon_value": "ub"}
    )
    pair = df_lb.merge(df_ub, on="image_id", how="inner")
    pair["lb"] = pd.to_numeric(pair["lb"], errors="coerce")
    pair["ub"] = pd.to_numeric(pair["ub"], errors="coerce")
    pair = pair.dropna(subset=["lb", "ub"]).sort_values("lb").reset_index(drop=True)
    return pair


def generate_sdp_attack_plots(df_csv: str | Path, *, config: SdpAttackConfig | None = None) -> dict[str, Path]:
    if config is None:
        config = SdpAttackConfig()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = _write_inputs_file(output_dir, df_csv)

    dfs, lb, ub = _prepare_two_series(df_csv, config=config)
    plot_df = pd.concat(dfs, ignore_index=True)
    colors = _default_colors(lb, ub)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"sdp_attack_{timestamp}"

    plot_paths: dict[str, Path] = {"inputs_sdp_attack": inputs_path}

    hist_ax = sns.histplot(data=plot_df, x="epsilon_value", hue="verifier", multiple="stack", palette=colors)
    hist_ax.set_xlabel("Epsilon value")
    hist_fig = hist_ax.get_figure()
    hist_path = output_dir / f"{base}_histogram.png"
    _save_fig(hist_fig, hist_path)
    plot_paths["histogram"] = hist_path

    box_ax = sns.boxplot(data=plot_df, x="verifier", y="epsilon_value", palette=colors)
    box_ax.set_ylabel("Epsilon value")
    box_ax.set_xticklabels(box_ax.get_xticklabels(), rotation=90)
    box_fig = box_ax.get_figure()
    box_path = output_dir / f"{base}_boxplot.png"
    _save_fig(box_fig, box_path)
    plot_paths["boxplot"] = box_path

    kde_ax = sns.kdeplot(data=plot_df, x="epsilon_value", hue="verifier", multiple="stack", palette=colors)
    kde_ax.set_xlabel("Epsilon value")
    kde_fig = kde_ax.get_figure()
    kde_path = output_dir / f"{base}_kde.png"
    _save_fig(kde_fig, kde_path)
    plot_paths["kde"] = kde_path

    ecdf_ax = sns.ecdfplot(data=plot_df, x="epsilon_value", hue="verifier", palette=colors)
    ecdf_ax.set_xlabel("Epsilon value")
    ecdf_ax.set_ylabel("Fraction epsilon values found")
    ecdf_fig = ecdf_ax.get_figure()
    ecdf_path = output_dir / f"{base}_ecdf.png"
    _save_fig(ecdf_fig, ecdf_path)
    plot_paths["ecdf"] = ecdf_path

    # Gap-ECDF (lb-oriented)
    try:
        pair = _pair_gap_df(df_csv, lb, ub)
        n = len(pair)
        y = np.zeros(n) if n <= 1 else np.linspace(0, 1, n)
        lb_vals = pair["lb"].to_numpy()
        ub_vals = pair["ub"].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.hlines(y, lb_vals, ub_vals, color=colors[ub], alpha=0.35, linewidth=2.5, zorder=1)
        ax.plot(lb_vals, y, color=colors[lb], linewidth=1.8, zorder=2)
        ax.set_xlabel("Epsilon value")
        ax.set_ylabel("Fraction epsilon values found")
        ax.grid(True, alpha=0.3)
        handles = [
            Patch(facecolor=colors[lb], edgecolor="none", label=lb),
            Patch(facecolor=colors[ub], edgecolor="none", label=ub),
        ]
        ax.legend(handles=handles, fontsize=10, handlelength=1, handleheight=1)
        plt.tight_layout()

        gap_path = output_dir / f"{base}_ecdf_gap.png"
        _save_fig(fig, gap_path)
        plot_paths["ecdf_gap"] = gap_path
    except Exception as e:
        logger.warning("Failed to generate gap-ECDF plot: %s", e)

    logger.info("Saved plots to %s", output_dir)
    return plot_paths
