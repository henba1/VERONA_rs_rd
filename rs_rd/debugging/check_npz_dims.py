#!/usr/bin/env python3
"""Script to print image dimensions from npz sidecar files."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Print image dimensions from npz sidecar files")
    parser.add_argument("directory", type=str, help="Directory containing npz files")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    npz_files = sorted(directory.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {directory}")
        return

    print(f"Found {len(npz_files)} npz file(s) in {directory}\n")
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            if "image" in data:
                image = data["image"]
                if image is not None:
                    min_val = float(np.min(image))
                    max_val = float(np.max(image))
                    print(f"{npz_file.name}: shape={image.shape}, range=[{min_val:.6f}, {max_val:.6f}]")
                else:
                    print(f"{npz_file.name}: image is None")
            else:
                print(f"{npz_file.name}: no 'image' key found")
        except Exception as e:
            print(f"{npz_file.name}: Error loading file - {e}")


if __name__ == "__main__":
    main()
