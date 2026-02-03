"""Compile summary statistics across k runs for a given experiment tag."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ada_verona import get_results_dir

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = get_results_dir("CIFAR-10")


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


def _load_single_row_summary(exp_dir: Path) -> dict:
    candidates = _find_df_csv_paths(exp_dir, "summary_df")
    if not candidates:
        return {}
    path = candidates[0]
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    out: dict[str, object] = {}
    for k, v in row.items():
        if isinstance(v, float) and not np.isfinite(v):
            out[str(k)] = None
        else:
            out[str(k)] = v
    return out


def _safe_float(x: object) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def compute_total_summary_df(
    experiment_tag: str,
    *,
    dataset_dir: Path = DEFAULT_RESULTS_DIR,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Compile total_summary_df from k summary_dfs for a given experiment tag.

    Finds all experiment dirs with prefix equal to experiment_tag, loads each
    summary_df.csv, and aggregates into a single row with summary statistics.

    Stats computed:
    - avg, max, min: overall_accuracy, accuracy_without_abstain, n_abstain, n_misclassified
    - avg only: avg_certified_radius, avg_cert_time
      (avg_cert_time = mean of sum_certification_time_seconds / (n_abstain + n_correct) per run)

    Args:
        experiment_tag: Prefix of experiment directories (e.g. "sigma_0.25_run")
        dataset_dir: Root directory containing experiment subdirs
        output_path: If provided, write the result DataFrame to this path as CSV

    Returns:
        DataFrame with one row containing the aggregated summary stats.
    """
    prefix = f"{experiment_tag}_"
    exp_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found for tag '{experiment_tag}' under {dataset_dir}")

    summaries: list[dict] = []
    for exp_dir in exp_dirs:
        row = _load_single_row_summary(exp_dir)
        if row:
            summaries.append(row)
        else:
            logger.warning("Empty or missing summary_df in %s, skipping", exp_dir.name)

    if not summaries:
        raise ValueError(f"No valid summary data found for experiment tag '{experiment_tag}'")

    df = pd.DataFrame(summaries)

    agg_cols = [
        "overall_accuracy",
        "accuracy_without_abstain",
        "n_abstain",
        "n_misclassified",
    ]
    for c in agg_cols:
        if c not in df.columns:
            logger.warning("Column %s not found in summary_df, using nan", c)

    result: dict[str, object] = {"experiment_tag": experiment_tag, "n_runs": len(summaries)}

    for col in agg_cols:
        if col in df.columns:
            vals = df[col].apply(_safe_float)
            result[f"{col}_avg"] = float(np.nanmean(vals))
            result[f"{col}_min"] = float(np.nanmin(vals))
            result[f"{col}_max"] = float(np.nanmax(vals))
        else:
            result[f"{col}_avg"] = float("nan")
            result[f"{col}_min"] = float("nan")
            result[f"{col}_max"] = float("nan")

    # avg_certified_radius: avg across runs
    if "avg_certified_radius" in df.columns:
        result["avg_certified_radius"] = float(np.nanmean(df["avg_certified_radius"].apply(_safe_float)))
    else:
        result["avg_certified_radius"] = float("nan")

    # avg_cert_time: mean of (sum_certification_time_seconds / (n_abstain + n_correct)) per run
    sum_cert = (
        df["sum_certification_time_seconds"].apply(_safe_float)
        if "sum_certification_time_seconds" in df.columns
        else pd.Series([float("nan")] * len(df))
    )
    n_abstain = df["n_abstain"].apply(_safe_float) if "n_abstain" in df.columns else pd.Series([0.0] * len(df))
    n_correct = df["n_correct"].apply(_safe_float) if "n_correct" in df.columns else pd.Series([0.0] * len(df))
    denom = n_abstain + n_correct
    cert_times = np.where(denom > 0, sum_cert / denom, float("nan"))
    result["avg_cert_time"] = float(np.nanmean(cert_times))

    out_df = pd.DataFrame([result])
    if output_path is not None:
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "total_summary_df.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        logger.info("Wrote total_summary_df to %s", output_path)
    return out_df
