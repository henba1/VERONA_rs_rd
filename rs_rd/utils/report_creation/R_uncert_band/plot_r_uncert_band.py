from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ada_verona import get_results_dir

matplotlib.use("Agg")
sns.set_style("darkgrid")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = get_results_dir("CIFAR-10")
DEFAULT_OUTPUT_DIR = Path("/gpfs/work2/0/prjs1681/runs/results/figures/0_R_uncert_band")
DEFAULT_K = 10


def _paired_violet_colors() -> tuple[str, str]:
    palette = sns.color_palette("Paired", 12).as_hex()
    light_violet, violet = palette[8], palette[9]
    return violet, light_violet


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


def _read_image_id_set(exp_dir: Path, df_name: str) -> tuple[set[int], Path | None]:
    candidates = _find_df_csv_paths(exp_dir, df_name)
    if not candidates:
        return set(), None
    path = candidates[0]
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"Missing 'image_id' column in {path}")
    ids = pd.to_numeric(df["image_id"], errors="coerce").dropna().astype(int)
    return set(ids.tolist()), path


def _presence_report_by_id(id_sets_by_seed: dict[int, set[int]]) -> dict:
    seeds_sorted = sorted(id_sets_by_seed.keys())
    sets = [id_sets_by_seed[s] for s in seeds_sorted]
    union = set().union(*sets) if sets else set()
    inter = set.intersection(*sets) if sets else set()
    inconsistent = sorted(union - inter)
    inconsistent_presence = [
        {
            "image_id": int(image_id),
            "present_in_seeds": [int(s) for s in seeds_sorted if image_id in id_sets_by_seed[s]],
        }
        for image_id in inconsistent
    ]
    missing_from_seed = {int(s): sorted(union - id_sets_by_seed[s]) for s in seeds_sorted}
    return {
        "n_seeds": len(seeds_sorted),
        "seeds": [int(s) for s in seeds_sorted],
        "n_union": len(union),
        "n_intersection": len(inter),
        "n_inconsistent": len(inconsistent),
        "inconsistent_image_ids": inconsistent,
        "inconsistent_presence": inconsistent_presence,
        "missing_from_seed": missing_from_seed,
    }


def _presence_report_by_run(id_sets_by_run: dict[str, set[int]]) -> dict:
    run_keys = sorted(id_sets_by_run.keys())
    sets = [id_sets_by_run[k] for k in run_keys]
    union = set().union(*sets) if sets else set()
    inter = set.intersection(*sets) if sets else set()
    inconsistent = sorted(union - inter)
    inconsistent_presence = [
        {
            "image_id": int(image_id),
            "present_in_runs": [k for k in run_keys if image_id in id_sets_by_run[k]],
        }
        for image_id in inconsistent
    ]
    missing_from_run = {k: sorted(union - id_sets_by_run[k]) for k in run_keys}
    return {
        "n_runs": len(run_keys),
        "runs": run_keys,
        "n_union": len(union),
        "n_intersection": len(inter),
        "n_inconsistent": len(inconsistent),
        "inconsistent_image_ids": inconsistent,
        "inconsistent_presence": inconsistent_presence,
        "missing_from_run": missing_from_run,
    }


def _load_single_row_summary(exp_dir: Path) -> tuple[dict, Path | None]:
    candidates = _find_df_csv_paths(exp_dir, "summary_df")
    if not candidates:
        return {}, None
    path = candidates[0]
    df = pd.read_csv(path)
    if df.empty:
        return {}, path
    row = df.iloc[0].to_dict()
    out: dict[str, object] = {}
    for k, v in row.items():
        if isinstance(v, float) and not np.isfinite(v):
            out[str(k)] = None
        else:
            out[str(k)] = v
    return out, path


def write_run_differences_report(
    run_records: list[dict],
    *,
    output_dir: Path,
    experiment_tag: str,
    n_common_result_ids: int,
) -> Path:
    abstained_by_run: dict[str, set[int]] = {}
    misclassified_by_run: dict[str, set[int]] = {}
    observed_by_run: dict[str, set[int]] = {}

    abstained_paths: dict[str, str | None] = {}
    misclassified_paths: dict[str, str | None] = {}
    summary_paths: dict[str, str | None] = {}

    summaries: list[dict] = []

    for r in run_records:
        exp_dir = Path(r["exp_dir"])
        run_key = r["run"]

        abstained_ids, abstained_path = _read_image_id_set(exp_dir, "abstained_df")
        misclassified_ids, misclassified_path = _read_image_id_set(exp_dir, "misclassified_df")
        summary_row, summary_path = _load_single_row_summary(exp_dir)

        abstained_by_run[run_key] = abstained_ids
        misclassified_by_run[run_key] = misclassified_ids
        observed_by_run[run_key] = set(r["result_ids_raw"]) | abstained_ids | misclassified_ids

        abstained_paths[run_key] = str(abstained_path) if abstained_path is not None else None
        misclassified_paths[run_key] = str(misclassified_path) if misclassified_path is not None else None
        summary_paths[run_key] = str(summary_path) if summary_path is not None else None

        summaries.append({"run": run_key, "exp_dir": str(exp_dir), "summary": summary_row})

    summary_metrics: dict[str, object] = {}
    if summaries and any(s["summary"] for s in summaries):
        keys = sorted({k for s in summaries for k in s["summary"]})
        rows = []
        for s in summaries:
            row = {"run": s["run"]}
            row.update({k: s["summary"].get(k) for k in keys})
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("run")
        numeric_cols = [c for c in df.columns if c != "run" and pd.api.types.is_numeric_dtype(df[c])]
        summary_metrics = {
            "per_run": df.to_dict(orient="records"),
            "numeric_summary": (df[numeric_cols].agg(["min", "median", "max"]).to_dict() if numeric_cols else {}),
        }

    result_ids_by_run = {r["run"]: set(r["result_ids_raw"]) for r in run_records}
    result_presence = _presence_report_by_run(result_ids_by_run)
    common_raw = set.intersection(*result_ids_by_run.values()) if result_ids_by_run else set()
    dropped_from_result_df_by_run = {k: sorted(v - common_raw) for k, v in result_ids_by_run.items()}

    abstained_report = _presence_report_by_run(abstained_by_run)
    misclassified_report = _presence_report_by_run(misclassified_by_run)
    abstain_inconsistent = set(abstained_report["inconsistent_image_ids"])
    misclass_inconsistent = set(misclassified_report["inconsistent_image_ids"])
    abstain_misclass_intersection = sorted(abstain_inconsistent & misclass_inconsistent)
    only_inconsistent_abstain = sorted(abstain_inconsistent - misclass_inconsistent)
    only_inconsistent_misclassify = sorted(misclass_inconsistent - abstain_inconsistent)
    abstain_vs_misclassify = {
        "inconsistent_abstain_and_misclassify_intersection": abstain_misclass_intersection,
        "n_intersection": len(abstain_misclass_intersection),
        "only_inconsistent_abstain": only_inconsistent_abstain,
        "n_only_inconsistent_abstain": len(only_inconsistent_abstain),
        "only_inconsistent_misclassify": only_inconsistent_misclassify,
        "n_only_inconsistent_misclassify": len(only_inconsistent_misclassify),
    }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_tag": experiment_tag,
        "runs": sorted([r["run"] for r in run_records]),
        "result_df": {
            "n_common_image_ids_used_for_ecdf": int(n_common_result_ids),
            "presence": result_presence,
            "dropped_from_result_df_by_run": dropped_from_result_df_by_run,
        },
        "abstained_df": abstained_report,
        "misclassified_df": misclassified_report,
        "abstain_vs_misclassify": abstain_vs_misclassify,
        "observed_any_df": _presence_report_by_run(observed_by_run),
        "paths": {
            "abstained_df": abstained_paths,
            "misclassified_df": misclassified_paths,
            "summary_df": summary_paths,
        },
        "summary_df": summary_metrics,
    }

    out_path = output_dir / "run_differences_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    summary = {
        "timestamp": report["timestamp"],
        "experiment_tag": experiment_tag,
        "n_runs": len(run_records),
        "result_df": {
            "n_common_image_ids_used_for_ecdf": int(n_common_result_ids),
            "per_run": {
                run: {"n_total": len(result_ids_by_run[run]), "n_dropped": len(dropped_from_result_df_by_run[run])}
                for run in sorted(result_ids_by_run.keys())
            },
        },
        "abstained_df": {
            "n_inconsistent": abstained_report["n_inconsistent"],
            "n_intersection": abstained_report["n_intersection"],
            "n_union": abstained_report["n_union"],
        },
        "misclassified_df": {
            "n_inconsistent": misclassified_report["n_inconsistent"],
            "n_intersection": misclassified_report["n_intersection"],
            "n_union": misclassified_report["n_union"],
        },
        "abstain_vs_misclassify": abstain_vs_misclassify,
        "summary_df_numeric": summary_metrics.get("numeric_summary", {}),
    }
    summary_path = output_dir / "run_differences_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return out_path, summary_path


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
        if "image_id" not in df.columns:
            raise ValueError(f"Missing 'image_id' column in {csv_path}")
        if "epsilon_value" not in df.columns:
            raise ValueError(f"Missing 'epsilon_value' column in {csv_path}")
        if "network" not in df.columns:
            raise ValueError(f"Missing 'network' column in {csv_path}")

        run_df = df[["image_id", "epsilon_value"]].copy()
        run_df["image_id"] = pd.to_numeric(run_df["image_id"], errors="coerce")
        run_df["epsilon_value"] = pd.to_numeric(run_df["epsilon_value"], errors="coerce")
        run_df = run_df.dropna(subset=["image_id", "epsilon_value"])
        run_df["image_id"] = run_df["image_id"].astype(int)
        run_df = run_df.drop_duplicates(subset=["image_id"], keep="first").reset_index(drop=True)
        result_ids_raw = set(run_df["image_id"].tolist())
        run_key = exp_dir.name

        run_records.append(
            {
                "exp_dir": exp_dir,
                "run": run_key,
                "csv_path": csv_path,
                "network": str(df["network"].iloc[0]),
                "df": run_df,
                "result_ids_raw": result_ids_raw,
            }
        )

    network_str = str(run_records[0]["network"])
    model_name = extract_model_name(network_str)
    sigma = extract_sigma(network_str)

    id_sets = [set(r["df"]["image_id"].tolist()) for r in run_records]
    common_ids = set.intersection(*id_sets) if id_sets else set()
    if not common_ids:
        raise ValueError("No common image_id values across runs")

    per_run_n_total = [len(r["df"]) for r in run_records]
    n_common = len(common_ids)
    if len(set(per_run_n_total + [n_common])) != 1:
        logger.warning(
            "Runs have differing numbers of radii; using intersection of image_id across runs (%d samples)", n_common
        )

    for r in run_records:
        r["df"] = r["df"].loc[r["df"]["image_id"].isin(common_ids)].sort_values("image_id").reset_index(drop=True)
        r["radii"] = r["df"]["epsilon_value"].to_numpy(dtype=float)

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
            "run": [r["run"] for r in run_records],
            "d_ks_like": d_vals,
        }
    ).sort_values("run")

    tag_out_dir = output_dir / experiment_tag
    tag_out_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = tag_out_dir / "inputs.txt"
    inputs_lines = [
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"experiment_tag={experiment_tag}",
        f"dataset_dir={dataset_dir}",
        f"expected_k={expected_k}",
        f"n_common_image_ids={n_common}",
        "",
    ]
    for idx, r in enumerate(run_records):
        n_total = per_run_n_total[idx]
        n_used = int(len(r["df"]))
        inputs_lines.append(
            f"run={r['run']}\tn_total={n_total}\tn_used={n_used}\t"
            f"dropped={n_total - n_used}\texp_dir={r['exp_dir']}\tcsv={r['csv_path']}"
        )
    inputs_lines.append("")
    inputs_path.write_text("\n".join(inputs_lines))

    diffs_path, summary_path = write_run_differences_report(
        run_records,
        output_dir=tag_out_dir,
        experiment_tag=experiment_tag,
        n_common_result_ids=n_common,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    median_color, band_color = _paired_violet_colors()
    ax.fill_between(eps_grid, f_lo, f_hi, step="pre", alpha=0.25, color=band_color, linewidth=0)
    ax.step(eps_grid, f_med, where="pre", color=median_color, alpha=0.9, linewidth=1)

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
        "run_differences_report": diffs_path,
        "run_differences_summary": summary_path,
        "inputs": inputs_path,
    }
