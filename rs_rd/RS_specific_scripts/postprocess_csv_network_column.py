"""
Postprocessing script to update network column values with verifier column values.

This script replaces the 'network' column with the 'verifier' column value
while retaining the 'verifier' column. This is a one-time postprocessing step.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rs_rd_research.paths import get_results_dir

logger = logging.getLogger(__name__)


def update_network_column_from_verifier(directory: Path) -> None:
    """
    Update the 'network' column with 'verifier' column values in CSV files.

    Args:
        directory: Path to directory containing the CSV file
    """
    csv_files = list(directory.glob("*.csv"))
    if len(csv_files) != 1:
        raise FileNotFoundError(f"Expected exactly one CSV file in {directory}, found {len(csv_files)}")

    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)

    if "network" not in df.columns:
        logger.warning(f"'network' column not found in {csv_file}")
        return

    if "verifier" not in df.columns:
        logger.warning(f"'verifier' column not found in {csv_file}")
        return

    # Replace network column with verifier column values
    original_network_values = df["network"].copy()
    df["network"] = df["verifier"]

    # Save back to the same file
    df.to_csv(csv_file, index=False)
    logger.info(f"Updated {csv_file}: replaced 'network' column with 'verifier' column values")
    logger.debug(f"Sample original network values: {original_network_values.head(3).tolist()}")
    logger.debug(f"Sample new network values: {df['network'].head(3).tolist()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results_dir = get_results_dir()
    result_dirs = [
        Path(results_dir) / "conv_big_standard_0.5_0.001_100_1000000_200",
        Path(results_dir) / "conv_big_standard_0.5_0.001_100_100000_200",
        Path(results_dir) / "conv_big_standard_0.5_0.0001_100_100000_200",
        Path(results_dir) / "conv_big_standard_0.25_0.0001_100_100000_200",
        Path(results_dir) / "conv_big_standard_0.25_0.001_100_100000_200",
    ]

    logger.info("Starting postprocessing of CSV files...")
    for result_dir in result_dirs:
        try:
            update_network_column_from_verifier(result_dir)
        except FileNotFoundError as e:
            logger.error(f"Error processing {result_dir}: {e}")

    logger.info("Postprocessing completed.")
