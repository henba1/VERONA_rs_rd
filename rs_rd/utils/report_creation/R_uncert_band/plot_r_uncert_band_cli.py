from __future__ import annotations

import argparse
import logging

from plot_r_uncert_band import plot_r_uncertainty_band_by_tag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ECDF uncertainty band for certified radii across reruns")
    parser.add_argument("experiment_tag", type=str, help="Experiment tag prefix to match")
    args = parser.parse_args()
    plot_r_uncertainty_band_by_tag(args.experiment_tag)


if __name__ == "__main__":
    main()
