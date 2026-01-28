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
class ComparConfig:
    output_dir: Path = Path("/gpfs/work2/0/prjs1681/runs/results/figures/4_Compar")
    df1_lb_verifier: str = "sdpcrown"
    df2_ub_prefix: str = "eotpgd"


def _as_int_if_possible(x: object) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v.is_integer():
        return str(int(v))
    return str(v)


def _extract_df2_lb_label(network: str) -> str:
    parts = str(network).split("_")
    if len(parts) < 9:
        return str(network)
    return f"{parts[2]}_{parts[7]}_{parts[8]}"


def _choose_df1_ub_verifier(df1: pd.DataFrame, lb_verifier: str) -> str:
    if "verifier" not in df1.columns:
        raise ValueError("df1 is missing required column: 'verifier'")
    candidates = df1.loc[df1["verifier"].astype(str) != lb_verifier, "verifier"].astype(str)
    if candidates.empty:
        raise ValueError(f"df1 has no verifier entries other than '{lb_verifier}'")
    counts = candidates.value_counts()
    if len(counts) > 1:
        logger.warning("df1 has multiple non-%s verifiers; using most frequent: %s", lb_verifier, counts.index[0])
    return str(counts.index[0])


def _make_verifier_df(values: pd.Series, verifier_label: str) -> pd.DataFrame:
    out = pd.DataFrame({"epsilon_value": pd.to_numeric(values, errors="coerce")})
    out = out.dropna(subset=["epsilon_value"])
    out["verifier"] = verifier_label
    out["smallest_sat_value"] = out["epsilon_value"]
    return out


def prepare_4_compar_dataframes(
    df1_csv: str | Path, df2_csv: str | Path, *, config: ComparConfig
) -> list[pd.DataFrame]:
    df1 = pd.read_csv(Path(df1_csv))
    df2 = pd.read_csv(Path(df2_csv))

    if "epsilon_value" not in df1.columns:
        raise ValueError("df1 is missing required column: 'epsilon_value'")

    df1_lb_verifier = config.df1_lb_verifier
    df1_ub_verifier = _choose_df1_ub_verifier(df1, df1_lb_verifier)

    df1_lb = df1.loc[df1["verifier"].astype(str) == df1_lb_verifier, "epsilon_value"]
    df1_ub = df1.loc[df1["verifier"].astype(str) == df1_ub_verifier, "epsilon_value"]

    df1_lb_df = _make_verifier_df(df1_lb, df1_lb_verifier)
    df1_ub_df = _make_verifier_df(df1_ub, df1_ub_verifier)

    required_df2 = {"network", "cert_radius_l2", "min_adv_radius_l2", "search_num_iter"}
    missing_df2 = sorted(required_df2 - set(df2.columns))
    if missing_df2:
        raise ValueError(f"df2 is missing required columns: {missing_df2}")

    df2_lb_label = _extract_df2_lb_label(str(df2["network"].iloc[0]))
    df2_ub_label = f"{config.df2_ub_prefix}_{_as_int_if_possible(df2['search_num_iter'].mode().iloc[0])}"

    df2_lb_df = _make_verifier_df(df2["cert_radius_l2"], df2_lb_label)
    df2_ub_df = _make_verifier_df(df2["min_adv_radius_l2"], df2_ub_label)

    return [df1_lb_df, df1_ub_df, df2_lb_df, df2_ub_df]


def _default_4_compar_colors(dfs: list[pd.DataFrame]) -> dict[str, str]:
    palette = sns.color_palette("Paired", 12).as_hex()
    light_blue, blue = palette[0], palette[1]
    light_violet, violet = palette[8], palette[9]

    labels = [str(df["verifier"].iloc[0]) for df in dfs]
    if len(labels) != 4:
        raise ValueError(f"Expected 4 verifier labels, got {len(labels)}")

    # Convention: lb is darker, ub is lighter.
    return {labels[0]: blue, labels[1]: light_blue, labels[2]: violet, labels[3]: light_violet}


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_inputs_file(output_dir: Path, df1_csv: str | Path, df2_csv: str | Path) -> Path:
    path = output_dir / "inputs.txt"
    content = "\n".join(
        [
            f"df1_csv={Path(df1_csv).resolve()}",
            f"df2_csv={Path(df2_csv).resolve()}",
            "",
        ]
    )
    path.write_text(content)
    return path


def generate_4_compar_plots(
    df1_csv: str | Path, df2_csv: str | Path, *, config: ComparConfig | None = None
) -> dict[str, Path]:
    if config is None:
        config = ComparConfig()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_inputs_file(output_dir, df1_csv, df2_csv)

    dfs = prepare_4_compar_dataframes(df1_csv, df2_csv, config=config)
    plot_df = pd.concat(dfs, ignore_index=True)
    color_map = _default_4_compar_colors(dfs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"4_compar_{timestamp}"

    plot_paths: dict[str, Path] = {"inputs": output_dir / "inputs.txt"}

    hist_ax = sns.histplot(data=plot_df, x="epsilon_value", hue="verifier", multiple="stack", palette=color_map)
    hist_ax.set_xlabel("Epsilon value")
    hist_fig = hist_ax.get_figure()
    hist_path = output_dir / f"{base}_histogram.png"
    _save_fig(hist_fig, hist_path)
    plot_paths["histogram"] = hist_path

    box_ax = sns.boxplot(data=plot_df, x="verifier", y="epsilon_value", palette=color_map)
    box_ax.set_xticklabels(box_ax.get_xticklabels(), rotation=90)
    box_fig = box_ax.get_figure()
    box_path = output_dir / f"{base}_boxplot.png"
    _save_fig(box_fig, box_path)
    plot_paths["boxplot"] = box_path

    kde_ax = sns.kdeplot(data=plot_df, x="epsilon_value", hue="verifier", multiple="stack", palette=color_map)
    kde_ax.set_xlabel("Epsilon value")
    kde_fig = kde_ax.get_figure()
    kde_path = output_dir / f"{base}_kde.png"
    _save_fig(kde_fig, kde_path)
    plot_paths["kde"] = kde_path

    ecdf_ax = sns.ecdfplot(data=plot_df, x="epsilon_value", hue="verifier", palette=color_map)
    ecdf_ax.set_xlabel("Epsilon value")
    ecdf_ax.set_ylabel("Fraction epsilon values found")
    ecdf_fig = ecdf_ax.get_figure()
    ecdf_path = output_dir / f"{base}_ecdf.png"
    _save_fig(ecdf_fig, ecdf_path)
    plot_paths["ecdf"] = ecdf_path

    fig, ax = plt.subplots()
    for verifier, group_df in plot_df.groupby("verifier", sort=False):
        xs = group_df["epsilon_value"].sort_values().reset_index(drop=True)
        ys = pd.Series(range(len(xs)), dtype=float) / max(len(xs) - 1, 1)
        ax.plot(xs, ys, label=verifier, color=color_map.get(verifier))
    ax.set_xlabel("Epsilon value")
    ax.set_ylabel("Fraction epsilon values found")
    ax.set_xlim(0, max(0.1, float(plot_df["epsilon_value"].max()) * 1.05))
    ax.legend()
    ax.grid(True, alpha=0.3)

    anne_path = output_dir / f"{base}_anneplot.png"
    _save_fig(fig, anne_path)
    plot_paths["anneplot"] = anne_path

    # Pair-wise lb-oriented CDF with ub gaps (lb uses darker color)
    try:
        palette = sns.color_palette("Paired", 12).as_hex()
        light_blue, blue = palette[0], palette[1]
        light_violet, violet = palette[8], palette[9]

        df1 = pd.read_csv(Path(df1_csv))
        df2 = pd.read_csv(Path(df2_csv))

        df1_lb_verifier = config.df1_lb_verifier
        df1_ub_verifier = _choose_df1_ub_verifier(df1, df1_lb_verifier)

        required_df1 = {"image_id", "verifier", "epsilon_value"}
        missing_df1 = sorted(required_df1 - set(df1.columns))
        if missing_df1:
            raise ValueError(f"df1 is missing required columns for pair CDF: {missing_df1}")

        df1_lb = df1.loc[df1["verifier"].astype(str) == df1_lb_verifier, ["image_id", "epsilon_value"]].rename(
            columns={"epsilon_value": "lb"}
        )
        df1_ub = df1.loc[df1["verifier"].astype(str) == df1_ub_verifier, ["image_id", "epsilon_value"]].rename(
            columns={"epsilon_value": "ub"}
        )
        df1_pair = df1_lb.merge(df1_ub, on="image_id", how="inner")
        df1_pair["lb"] = pd.to_numeric(df1_pair["lb"], errors="coerce")
        df1_pair["ub"] = pd.to_numeric(df1_pair["ub"], errors="coerce")
        df1_pair = df1_pair.dropna(subset=["lb", "ub"]).sort_values("lb").reset_index(drop=True)

        required_df2 = {"image_id", "cert_radius_l2", "min_adv_radius_l2"}
        missing_df2 = sorted(required_df2 - set(df2.columns))
        if missing_df2:
            raise ValueError(f"df2 is missing required columns for pair CDF: {missing_df2}")

        df2_pair = df2[["image_id", "cert_radius_l2", "min_adv_radius_l2"]].rename(
            columns={"cert_radius_l2": "lb", "min_adv_radius_l2": "ub"}
        )
        df2_pair["lb"] = pd.to_numeric(df2_pair["lb"], errors="coerce")
        df2_pair["ub"] = pd.to_numeric(df2_pair["ub"], errors="coerce")
        df2_pair = df2_pair.dropna(subset=["lb", "ub"]).sort_values("lb").reset_index(drop=True)

        df2_lb_label = _extract_df2_lb_label(str(df2["network"].iloc[0])) if "network" in df2.columns else "RS"
        df2_ub_label = f"{config.df2_ub_prefix}_{_as_int_if_possible(df2['search_num_iter'].mode().iloc[0])}"

        df1_label = df1_lb_verifier
        df1_label_ub = df1_ub_verifier
        df2_label = df2_lb_label
        df2_label_ub = df2_ub_label

        def _save_pair_gap(
            path: Path,
            pair_df: pd.DataFrame,
            lb_color: str,
            ub_color: str,
            lb_name: str,
            ub_name: str,
        ) -> None:
            fig, ax = plt.subplots(figsize=(8, 6))
            n = len(pair_df)
            y = np.zeros(n) if n <= 1 else np.linspace(0, 1, n)
            lb = pair_df["lb"].to_numpy()
            ub = pair_df["ub"].to_numpy()

            ax.hlines(y, lb, ub, color=ub_color, alpha=0.35, linewidth=2.5, zorder=1)
            ax.plot(lb, y, color=lb_color, linewidth=1.8, zorder=2)
            ax.set_xlabel("Epsilon value")
            ax.set_ylabel("Fraction epsilon values found")
            ax.grid(True, alpha=0.3)
            handles = [
                Patch(facecolor=lb_color, edgecolor="none", label=lb_name),
                Patch(facecolor=ub_color, edgecolor="none", label=ub_name),
            ]
            ax.legend(handles=handles, fontsize=10, handlelength=1, handleheight=1)
            plt.tight_layout()
            _save_fig(fig, path)

        df1_gap_path = output_dir / f"{base}_pair_gap_cdf_df1.png"
        _save_pair_gap(df1_gap_path, df1_pair, blue, light_blue, df1_label, df1_label_ub)
        plot_paths["pair_gap_cdf_df1"] = df1_gap_path

        df2_gap_path = output_dir / f"{base}_pair_gap_cdf_df2.png"
        _save_pair_gap(df2_gap_path, df2_pair, violet, light_violet, df2_label, df2_label_ub)
        plot_paths["pair_gap_cdf_df2"] = df2_gap_path

        # Combined plot: both pairs on one axes
        fig, ax = plt.subplots(figsize=(10, 7))
        for pair_df, lb_color, ub_color in [
            (df1_pair, blue, light_blue),
            (df2_pair, violet, light_violet),
        ]:
            n = len(pair_df)
            y = np.zeros(n) if n <= 1 else np.linspace(0, 1, n)
            lb = pair_df["lb"].to_numpy()
            ub = pair_df["ub"].to_numpy()

            ax.hlines(y, lb, ub, color=ub_color, alpha=0.28, linewidth=2.0, zorder=1)
            ax.plot(lb, y, color=lb_color, linewidth=1.8, zorder=2)

        ax.set_xlabel("Epsilon value")
        ax.set_ylabel("Fraction epsilon values found")
        ax.grid(True, alpha=0.3)
        handles = [
            Patch(facecolor=blue, edgecolor="none", label=df1_label),
            Patch(facecolor=light_blue, edgecolor="none", label=df1_label_ub),
            Patch(facecolor=violet, edgecolor="none", label=df2_label),
            Patch(facecolor=light_violet, edgecolor="none", label=df2_label_ub),
        ]
        ax.legend(handles=handles, fontsize=10, handlelength=1, handleheight=1)
        plt.tight_layout()

        pair_gap_path = output_dir / f"{base}_pair_gap_cdf.png"
        _save_fig(fig, pair_gap_path)
        plot_paths["pair_gap_cdf"] = pair_gap_path
    except Exception as e:
        logger.warning("Failed to generate pair gap CDF plot: %s", e)

    logger.info("Saved plots to %s", output_dir)
    return plot_paths
