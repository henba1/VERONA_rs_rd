import argparse
import logging
import sys
from pathlib import Path

from plot_sdp_attack import SdpAttackConfig, generate_sdp_attack_plots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sdpcrown (lb) vs attack (ub) from a compiled df1 CSV.")
    parser.add_argument("df_csv", type=Path, help="Path to df1-style CSV (needs epsilon_value, verifier, image_id)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SdpAttackConfig.output_dir,
        help=f"Output directory (default: {SdpAttackConfig.output_dir})",
    )
    parser.add_argument(
        "--lb-verifier",
        type=str,
        default=SdpAttackConfig.lb_verifier,
        help=f"Verifier name used as lower bound (default: {SdpAttackConfig.lb_verifier})",
    )

    args = parser.parse_args()

    try:
        config = SdpAttackConfig(output_dir=args.output_dir, lb_verifier=args.lb_verifier)
        plot_paths = generate_sdp_attack_plots(args.df_csv, config=config)
    except Exception as e:
        logger.error("Failed to generate plots: %s", e, exc_info=True)
        sys.exit(1)

    for name, path in plot_paths.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
