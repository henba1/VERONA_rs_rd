from __future__ import annotations

import logging
import re
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

DEFAULT_RESULTS_DIR = Path("/gpfs/work2/0/prjs1681/runs/results/CIFAR-10")
DEFAULT_OUTPUT_DIR = Path("/gpfs/work2/0/prjs1681/runs/results/figures/0_R_uncert_band")
DEFAULT_K = 10


def extract_model_name(network_str: str) -> str:
    parts = str(network_str).split("_")
    if len(parts) >= 2 and parts[0] == "RS":
        model_name = parts[1]
        return "conv_large" if model_name == "convlarge" else model_name

    match = re.match(r"RS_([^_]+)", str(network_str))
    if match:
        model_name = match.group(1)
        return "conv_large" if model_name == "convlarge" else model_name

    return "unknown_model"


def extract_sigma(network_str: str) -> float | None:
    match = re.search(r"_([0-9]*\.?[0-9]+)_([0-9]*\.?[0-9]+)_(\d+)_(\d+)$", str(network_str))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _find_df_csv_paths(search_root: Path, df_name: str) -> list[Path]:
    filename = f"{df_name}.csv"
    preferred = [
        search_root / filename,
        search_root / "results" / filename,
    ]
    hits: list[Path] = [p for p in preferred if p.exists()]
    if hits:
        return hits
    return sorted(search_root.rglob(filename))


def _read_overrides_seed(exp_dir: Path) -> int | None:
    overrides_path = exp_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        return None
    for line in overrides_path.read_text().splitlines():
        s = line.strip().lstrip("-").strip()
        if "certify_seed" not in s:
            continue
        match = re.search(r"(?:\+\+)?certify_seed\s*=\s*([0-9]+)", s)
        if match:
            return int(match.group(1))
    return None


def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _ecdf_on_grid(values: np.ndarray, eps_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(eps_grid, dtype=float)
    values.sort()
    return np.searchsorted(values, eps_grid, side="right") / float(values.size)


def plot_r_uncertainty_band_by_tag(
    experiment_tag: str,
    *,
    dataset_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_k: int = DEFAULT_K,
) -> dict[str, Path]:
    prefix = f"{experiment_tag}_"
    exp_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found for tag '{experiment_tag}' under {dataset_dir}")

    if expected_k > 0 and len(exp_dirs) != expected_k:
        logger.warning("Expected %d dirs for tag '%s', found %d", expected_k, experiment_tag, len(exp_dirs))

    run_records: list[dict] = []
    for exp_dir in exp_dirs:
        csv_candidates = _find_df_csv_paths(exp_dir, "result_df")
        if not csv_candidates:
            raise FileNotFoundError(f"result_df.csv not found under {exp_dir}")
        csv_path = csv_candidates[0]
        df = pd.read_csv(csv_path)
        if "epsilon_value" not in df.columns:
            raise ValueError(f"Missing 'epsilon_value' column in {csv_path}")
        if "network" not in df.columns:
            raise ValueError(f"Missing 'network' column in {csv_path}")

        radii = pd.to_numeric(df["epsilon_value"], errors="coerce").dropna().to_numpy(dtype=float)
        seed = _read_overrides_seed(exp_dir)
        if seed is None:
            seed = -1
        run_records.append(
            {
                "exp_dir": exp_dir,
                "csv_path": csv_path,
                "seed": seed,
                "network": str(df["network"].iloc[0]),
                "radii": radii,
            }
        )

    network_str = str(run_records[0]["network"])
    model_name = extract_model_name(network_str)
    sigma = extract_sigma(network_str)

    all_radii = np.concatenate([r["radii"] for r in run_records if r["radii"].size > 0], axis=0)
    eps_grid = np.sort(np.unique(all_radii))
    if eps_grid.size == 0:
        raise ValueError("No radii found across runs")

    f_mat = np.stack([_ecdf_on_grid(r["radii"], eps_grid) for r in run_records], axis=0)
    f_med = np.median(f_mat, axis=0)
    f_lo = np.percentile(f_mat, 10, axis=0)
    f_hi = np.percentile(f_mat, 90, axis=0)

    d_vals = np.max(np.abs(f_mat - f_med[None, :]), axis=1)
    ks_df = pd.DataFrame(
        {
            "seed": [int(r["seed"]) for r in run_records],
            "d_ks_like": d_vals,
        }
    ).sort_values("seed")

    tag_out_dir = output_dir / experiment_tag
    tag_out_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = tag_out_dir / "inputs.txt"
    inputs_lines = [
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"experiment_tag={experiment_tag}",
        f"dataset_dir={dataset_dir}",
        f"expected_k={expected_k}",
        "",
    ]
    for r in run_records:
        inputs_lines.append(f"seed={r['seed']}\texp_dir={r['exp_dir']}\tcsv={r['csv_path']}")
    inputs_lines.append("")
    inputs_path.write_text("\n".join(inputs_lines))

    fig, ax = plt.subplots(figsize=(10, 7))
    color = sns.color_palette("deep", 1)[0]
    ax.fill_between(eps_grid, f_lo, f_hi, step="pre", alpha=0.25, color=color, linewidth=0)
    ax.step(eps_grid, f_med, where="pre", color=color, linewidth=2.0)

    if sigma is None:
        label = model_name
        title = f"ECDF uncertainty band ({model_name})"
    else:
        label = f"{model_name} (σ={sigma:g})"
        title = f"ECDF uncertainty band ({model_name}, σ={sigma:g})"

    ax.set_title(title)
    ax.set_xlabel("Certified radius (epsilon_value)")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0, max(0.1, float(eps_grid.max()) * 1.05))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend([label], loc="lower right")
    plt.tight_layout()

    out_plot = tag_out_dir / f"ecdf_band_{_sanitize_filename(model_name)}.png"
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    out_csv = tag_out_dir / "ks_like_summary.csv"
    ks_df.to_csv(out_csv, index=False)

    stats_path = tag_out_dir / "ks_like_stats.txt"
    d_max = float(np.max(d_vals)) if d_vals.size else float("nan")
    d_med = float(np.median(d_vals)) if d_vals.size else float("nan")
    d90 = float(np.percentile(d_vals, 90)) if d_vals.size else float("nan")
    stats_path.write_text(f"d_max={d_max}\nd_med={d_med}\nd90={d90}\n")

    return {
        "plot": out_plot,
        "ks_like_summary_csv": out_csv,
        "ks_like_stats": stats_path,
        "inputs": inputs_path,
    }
