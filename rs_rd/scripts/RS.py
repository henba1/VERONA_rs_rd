import shutil
import subprocess
from datetime import datetime
from pathlib import Path

VERONA_ROOT = Path(__file__).resolve().parents[2]  # /.../VERONA
RS_ROOT = VERONA_ROOT.parent / "randomized_smoothing"  # /.../randomized_smoothing

RS_CERTIFY = RS_ROOT / "cifar10" / "certify.py"
RS_RESULTS_DIR = RS_ROOT / "results"  # or use rs_rd_research.paths.get_results_dir()

VERONA_EXPERIMENT_BASE = VERONA_ROOT / "examples" / "example_experiment" / "results"


def run_rs_certification(
    sigma: float = 0.25,
    sample_size: int = 100,
    N0: int = 100,
    N: int = 100_000,
    batch_size: int = 200,
    alpha: float = 0.001,
    classifier_type: str = "huggingface",
    classifier_name: str = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
) -> Path:
    """Call the RS certify.py script via subprocess and return the newest *_result_df.csv path."""
    cmd = [
        "python",
        str(RS_CERTIFY),
        "--sigma",
        str(sigma),
        "--sample_size",
        str(sample_size),
        "--N0",
        str(N0),
        "--N",
        str(N),
        "--batch_size",
        str(batch_size),
        "--alpha",
        str(alpha),
        "--classifier_type",
        classifier_type,
        "--classifier_name",
        classifier_name,
    ]
    subprocess.run(cmd, check=True)

    # Find the latest VERONA-style CSV produced by RS (created by create_verona_csv)
    rs_csvs = list(RS_RESULTS_DIR.rglob("*_result_df.csv"))
    if not rs_csvs:
        raise RuntimeError(f"No RS result_df CSV found under {RS_RESULTS_DIR}")
    latest = max(rs_csvs, key=lambda p: p.stat().st_mtime)
    return latest


def import_into_verona(latest_rs_csv: Path) -> Path:
    """Create a VERONA experiment folder and copy RS CSV as result_df.csv."""
    ts = datetime.now().strftime("%d-%m-%Y+%H_%M")
    exp_dir = VERONA_EXPERIMENT_BASE / f"rs_certification_{ts}"
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)

    target_csv = exp_dir / "results" / "result_df.csv"
    shutil.copy2(latest_rs_csv, target_csv)
    return target_csv


def main():  # TODO: determine which reslt and txt files to copy
    rs_csv = run_rs_certification()
    verona_csv = import_into_verona(rs_csv)
    print(f"RS results imported to VERONA experiment at: {verona_csv}")


if __name__ == "__main__":
    main()
