"""Create the publication figure for IMPGM large-image generation results."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image


CONFIGURATIONS = (
    ("ddpm_direct", "DDPM-1000\nDirect-stitch", "ddpm_direct_root"),
    ("ddpm_overlap", "DDPM-1000\nOverlap-aware", "ddpm_overlap_root"),
    ("ddim_direct", "DDIM-250\nDirect-stitch", "ddim_direct_root"),
    ("ddim_overlap", "DDIM-250\nOverlap-aware", "ddim_overlap_root"),
)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _scene_ids(scene_manifest, count, seed, requested_ids):
    payload = _load_json(scene_manifest)
    scenes = payload.get("scenes", [])
    available = [str(item["scene_id"]) for item in scenes]
    if requested_ids:
        missing = [item for item in requested_ids if item not in available]
        if missing:
            raise ValueError(f"Unknown scene IDs: {', '.join(missing)}")
        return list(requested_ids)
    if count <= 0 or count > len(available):
        raise ValueError(f"--num-scenes must be within [1, {len(available)}].")
    return random.Random(seed).sample(available, count)


def _read_mask(path):
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            array = dataset.read(1)
    except ImportError:
        array = np.asarray(Image.open(path))
    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"Mask contains no finite values: {path}")
    threshold = 0.5 if float(finite.max()) <= 1.0 else 127.5
    return (array > threshold).astype(np.float32)


def _read_rgb(scene_dir, percentiles):
    tif_path = scene_dir / "generated.tif"
    try:
        import rasterio

        with rasterio.open(tif_path) as dataset:
            array = dataset.read().astype(np.float32)
        if array.shape[0] < 3:
            raise ValueError(f"Expected at least three bands: {tif_path}")
        rgb = array[[2, 1, 0]]
        output = np.zeros_like(rgb, dtype=np.float32)
        for index, band in enumerate(rgb):
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                continue
            low, high = np.percentile(finite, percentiles)
            if high <= low:
                output[index] = np.clip(band, 0.0, 1.0)
            else:
                output[index] = np.clip((band - low) / (high - low), 0.0, 1.0)
        return np.moveaxis(output, 0, -1)
    except ImportError:
        return np.asarray(Image.open(scene_dir / "generated_rgb.png").convert("RGB")) / 255.0


def _save_rgb(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
    image.save(path)


def _archive_samples(representative_root, scene_ids, roots, percentiles):
    representative_root.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_id in scene_ids:
        scene_output = representative_root / scene_id
        scene_output.mkdir(parents=True, exist_ok=True)
        source_mask = roots["ddpm_direct"] / "scenes" / scene_id / "condition_mask.tif"
        shutil.copy2(source_mask, scene_output / "condition_mask.tif")
        record = {"scene_id": scene_id, "files": {}}
        for key, _, _ in CONFIGURATIONS:
            source_dir = roots[key] / "scenes" / scene_id
            rgb = _read_rgb(source_dir, percentiles)
            full_path = scene_output / f"{key}.png"
            _save_rgb(full_path, rgb)
            record["files"][key] = {"full": str(full_path.resolve())}
        records.append(record)
    metadata = {
        "selection": "fixed_random_sample_from_scene_manifest",
        "scene_ids": scene_ids,
        "stretch_percentiles": list(percentiles),
        "records": records,
    }
    with (representative_root / "representative_samples.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def create_figure(args):
    roots = {key: Path(getattr(args, argument)).expanduser().resolve() for key, _, argument in CONFIGURATIONS}
    for key, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing generation root for {key}: {root}")
    scene_ids = _scene_ids(args.scene_manifest, args.num_scenes, args.seed, args.scene_ids)
    percentiles = tuple(float(value) for value in args.stretch_percentiles)

    figure, axes = plt.subplots(
        len(scene_ids),
        5,
        figsize=(18.0, 3.55 * len(scene_ids)),
        squeeze=False,
    )
    titles = ["Condition\nMask"] + [title for _, title, _ in CONFIGURATIONS]
    header_font = FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf", size=17)
    row_font = FontProperties(fname=r"C:\Windows\Fonts\timesbd.ttf", size=16)
    text_color = "#2b3036"

    for row, scene_id in enumerate(scene_ids):
        mask_path = roots["ddpm_direct"] / "scenes" / scene_id / "condition_mask.tif"
        mask = _read_mask(mask_path)
        panels = [mask]
        for key, _, _ in CONFIGURATIONS:
            panels.append(_read_rgb(roots[key] / "scenes" / scene_id, percentiles))

        for column, (axis, panel) in enumerate(zip(axes[row], panels)):
            if column == 0:
                axis.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            else:
                axis.imshow(panel, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(titles[column], fontproperties=header_font, color=text_color, pad=11)

        axes[row, 0].text(
            -0.16,
            0.5,
            f"Sample\n{row + 1}",
            transform=axes[row, 0].transAxes,
            ha="center",
            va="center",
            fontproperties=row_font,
            color=text_color,
        )

    figure.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.012, wspace=0.025, hspace=0.035)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    print(f"Saved figure: {output}")
    if not args.no_archive:
        representative_root = (
            Path(args.representative_root).expanduser().resolve()
            if args.representative_root
            else output.parent / "representative_samples"
        )
        _archive_samples(representative_root, scene_ids, roots, percentiles)
        print(f"Archived representative samples: {representative_root}")
    print(f"Selected scenes: {', '.join(scene_ids)}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--ddpm-direct-root", required=True)
    parser.add_argument("--ddpm-overlap-root", required=True)
    parser.add_argument("--ddim-direct-root", required=True)
    parser.add_argument("--ddim-overlap-root", required=True)
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--scene-ids", nargs="+")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--stretch-percentiles", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--representative-root")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    low, high = args.stretch_percentiles
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("--stretch-percentiles must satisfy 0 <= low < high <= 100.")
    create_figure(args)


if __name__ == "__main__":
    main()
