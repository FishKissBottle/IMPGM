"""Compute per-channel mean and std from training images for a dataset.

Usage:
    python IMPGM_Compute_Mean_Std.py --dataset-yaml configs/datasets/main.yaml
    python IMPGM_Compute_Mean_Std.py --dataset-yaml configs/datasets/main.yaml --update-yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from IMPGM_TifReader import Tif_Read_and_Write


def collect_image_paths(image_roots: list[str]) -> list[str]:
    paths = []
    for root in image_roots:
        root_path = Path(root)
        if not root_path.is_dir():
            print(f"[Warning] Directory not found: {root}")
            continue
        for p in sorted(root_path.iterdir()):
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
                paths.append(str(p))
    return paths


def compute_channel_mean_std(img_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute global per-channel mean and std across all images.

    Args:
        img_paths: List of paths to multi-channel TIF images.

    Returns:
        mean: (C,) array of per-channel means.
        std : (C,) array of per-channel standard deviations.
    """
    ch_sum = None
    ch_sumsq = None
    valid_cnt = None

    skipped_zero = 0
    for path in tqdm(img_paths, desc="Computing stats"):
        img, _, _ = Tif_Read_and_Write().Tif_Read(path)  # (C, H, W)
        img = img.astype(np.float64)

        # Align with IMPGM_Dataset: skip all-zero images
        if np.max(img) == 0.0:
            skipped_zero += 1
            continue

        mask = np.isfinite(img)
        x = np.where(mask, img, 0.0)

        per_ch_sum = x.sum(axis=(1, 2))
        per_ch_sumsq = (x * x).sum(axis=(1, 2))
        per_ch_cnt = mask.sum(axis=(1, 2)).astype(np.float64)

        if ch_sum is None:
            ch_sum = per_ch_sum
            ch_sumsq = per_ch_sumsq
            valid_cnt = per_ch_cnt
        else:
            ch_sum += per_ch_sum
            ch_sumsq += per_ch_sumsq
            valid_cnt += per_ch_cnt

    if skipped_zero:
        print(f"[Info] Skipped {skipped_zero} all-zero image(s) (aligned with dataset filter).")

    if ch_sum is None:
        raise ValueError(
            f"All {len(img_paths)} training image(s) were filtered out "
            f"({skipped_zero} all-zero). Cannot compute statistics."
        )

    valid_cnt = np.maximum(valid_cnt, 1.0)
    mean = ch_sum / valid_cnt
    var = ch_sumsq / valid_cnt - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, std


def update_yaml(yaml_path: Path, mean: np.ndarray, std: np.ndarray, precision: int = 6):
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    fmt = f"{{:.{precision}f}}"
    mean_list = [float(fmt.format(v)) for v in mean]
    std_list = [float(fmt.format(v)) for v in std]

    if "normalization" not in data:
        data["normalization"] = {}

    data["normalization"]["image_mean"] = mean_list
    data["normalization"]["image_std"] = std_list
    data["normalization"]["source"] = "train_split_statistics"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=None)

    print(f"[Updated] {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute dataset channel statistics.")
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        required=True,
        help="Path to the dataset YAML config (e.g. configs/datasets/main.yaml).",
    )
    parser.add_argument(
        "--update-yaml",
        action="store_true",
        help="Write computed mean/std back into the dataset YAML.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for output values (default: 6).",
    )
    args = parser.parse_args()

    yaml_path = PROJECT_ROOT / args.dataset_yaml
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_roots = cfg.get("dataset", {}).get("splits", {}).get("train", {}).get("image_roots", [])
    if not train_roots:
        # Fallback to old-style keys
        train_roots = cfg.get("dataset", {}).get("img_rootdir_list_forTrain", [])

    if not train_roots:
        raise ValueError("No training image roots found in YAML.")

    print(f"Dataset YAML: {yaml_path}")
    print(f"Training roots ({len(train_roots)}):")
    for r in train_roots:
        print(f"  - {r}")

    img_paths = collect_image_paths(train_roots)
    if not img_paths:
        raise ValueError("No training images found.")

    print(f"Total images: {len(img_paths)}")

    mean, std = compute_channel_mean_std(img_paths)

    print(f"\nChannel mean: {mean.tolist()}")
    print(f"Channel std : {std.tolist()}")

    if args.update_yaml:
        update_yaml(yaml_path, mean, std, precision=args.precision)
        print("\nYAML updated successfully.")
    else:
        print("\nTo update the YAML, re-run with --update-yaml")


if __name__ == "__main__":
    main()
