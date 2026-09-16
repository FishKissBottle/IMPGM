import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Archive aligned representative samples for the IMPGM background "
            "reference-control experiment and compute paired GMSD/FSIM."
        )
    )
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--uncontrolled-root", required=True)
    parser.add_argument("--background-controlled-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=999)
    return parser.parse_args()


def _save_tensor_image(tensor, path):
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .add(0.5)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


def _output_name(batch_index, position, batch_length, label, suffix):
    stem = f"batch_{batch_index:04d}"
    if batch_length > 1:
        stem += f"_p{position + 1}_{label}"
    return f"{stem}{suffix}"


def _load_tif_tensor(path, tif_reader):
    import numpy as np
    import torch

    array, _, _ = tif_reader.Tif_Read(str(path))
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D TIF array, got {array.shape}: {path}")
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _scalar(value):
    if value is None:
        return None
    return float(value.detach().cpu().item())


def _compute_metrics(prediction, reference, compute_gmsd, compute_fsim):
    import torch

    with torch.no_grad():
        rgb_gmsd, nir_gmsd = compute_gmsd(prediction, reference)
        rgb_fsim, nir_fsim = compute_fsim(prediction, reference)
    return {
        "rgb_gmsd": _scalar(rgb_gmsd),
        "rgb_fsim": _scalar(rgb_fsim),
        "nir_gmsd": _scalar(nir_gmsd),
        "nir_fsim": _scalar(nir_fsim),
    }


def _font(size):
    candidates = (
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_centered(draw, text, center_x, y, font):
    box = draw.textbbox((0, 0), text, font=font)
    x = center_x - (box[2] - box[0]) / 2
    draw.text((x, y), text, fill="black", font=font)


def _save_comparison_figure(
    reference_path,
    uncontrolled_path,
    controlled_path,
    uncontrolled_metrics,
    controlled_metrics,
    output_path,
):
    images = [
        Image.open(reference_path).convert("RGB"),
        Image.open(uncontrolled_path).convert("RGB"),
        Image.open(controlled_path).convert("RGB"),
    ]
    tile_width, tile_height = images[0].size
    images = [
        image.resize((tile_width, tile_height), Image.Resampling.BICUBIC)
        if image.size != (tile_width, tile_height)
        else image
        for image in images
    ]

    margin = 8
    gap = 16
    caption_height = 62
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + tile_width * len(images) + gap * (len(images) - 1),
            margin * 2 + tile_height + caption_height,
        ),
        "white",
    )
    title_font = _font(18)
    metric_font = _font(16)
    titles = ("Reference", "Uncontrolled", "Background-controlled")
    metric_lines = (
        "RGB GMSD / FSIM",
        f"{uncontrolled_metrics['rgb_gmsd']:.4f} / {uncontrolled_metrics['rgb_fsim']:.4f}",
        f"{controlled_metrics['rgb_gmsd']:.4f} / {controlled_metrics['rgb_fsim']:.4f}",
    )
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        x = margin + index * (tile_width + gap)
        canvas.paste(image, (x, margin))
        center_x = x + tile_width / 2
        title_y = margin + tile_height + 4
        _draw_centered(draw, titles[index], center_x, title_y, title_font)
        _draw_centered(draw, metric_lines[index], center_x, title_y + 25, metric_font)
    canvas.save(output_path)


def _metric_row(record):
    uncontrolled = record["metrics"]["uncontrolled"]
    controlled = record["metrics"]["background_controlled"]
    row = {
        "archive_id": record["archive_id"],
        "dataset_index": record["dataset_index"],
        "label": record["label"],
        "sampling_seed": record["sampling_seed"],
    }
    for name, values in (
        ("uncontrolled", uncontrolled),
        ("background_controlled", controlled),
    ):
        for metric, value in values.items():
            row[f"{name}_{metric}"] = value
    for metric in ("rgb_gmsd", "rgb_fsim", "nir_gmsd", "nir_fsim"):
        first = uncontrolled[metric]
        second = controlled[metric]
        row[f"delta_{metric}"] = (
            None if first is None or second is None else second - first
        )
    return row


def _write_metric_outputs(records, output_root):
    rows = [_metric_row(record) for record in records]
    csv_path = output_root / "representative_metrics.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = ("rgb_gmsd", "rgb_fsim", "nir_gmsd", "nir_fsim")
    summary = {
        "num_samples": len(records),
        "metric_reference": "paired_against_the_same_real_reference_image",
        "metric_input": "denormalized_full_band_tif_clamped_to_0_1",
        "metrics": {},
        "background_controlled_improvement_counts": {},
    }
    for branch in ("uncontrolled", "background_controlled"):
        summary["metrics"][branch] = {}
        for metric in metric_names:
            values = [
                record["metrics"][branch][metric]
                for record in records
                if record["metrics"][branch][metric] is not None
            ]
            summary["metrics"][branch][metric] = (
                None if not values else mean(values)
            )
    for metric in metric_names:
        comparable = [
            record
            for record in records
            if record["metrics"]["uncontrolled"][metric] is not None
            and record["metrics"]["background_controlled"][metric] is not None
        ]
        if metric.endswith("gmsd"):
            improved = sum(
                record["metrics"]["background_controlled"][metric]
                < record["metrics"]["uncontrolled"][metric]
                for record in comparable
            )
        else:
            improved = sum(
                record["metrics"]["background_controlled"][metric]
                > record["metrics"]["uncontrolled"][metric]
                for record in comparable
            )
        summary["background_controlled_improvement_counts"][metric] = {
            "improved": improved,
            "comparable": len(comparable),
        }

    summary_path = output_root / "representative_metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, summary_path


def _balanced_selection(candidates, num_samples, seed):
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["label"]].append(candidate)

    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    selected = []
    labels = sorted(groups)
    while len(selected) < num_samples:
        added = False
        for label in labels:
            if groups[label] and len(selected) < num_samples:
                selected.append(groups[label].pop())
                added = True
        if not added:
            break
    if len(selected) != num_samples:
        raise RuntimeError(
            f"Only {len(selected)} aligned samples are available; "
            f"{num_samples} were requested."
        )
    return selected


def main():
    args = _parse_args()
    import torch

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    dataset_yaml = Path(args.dataset_yaml).resolve()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_yaml}")
    os.environ["IMPGM_DATASET_YAML"] = str(dataset_yaml)

    from IMPGM_Config import (
        DATASET_DICT,
        DATASET_NAME,
        DEVICE,
        DIFFUSION_TEST_MICROBATCH_SIZE,
        DRAW_RANDOM_SEED,
        IMGSYN_CONTROLNET_CONFIG,
        TEST_BATCH_SIZE,
    )
    from IMPGM_Dataset import IMPGM_Dataset, _resolve_mask_path
    from Evaluation.Evaluation_Code.IMPGM_Quality_Metrics import (
        compute_FSIM,
        compute_GMSD,
    )
    from IMPGM_TifReader import Tif_Read_and_Write
    from IMPGM_Utils import (
        denormalize_image_tensor,
        extract_high_frequency,
        prepare_high_freq_vis_tensor,
        prepare_mask_vis_tensor,
        prepare_rgb_vis_tensor,
    )

    uncontrolled_root = Path(args.uncontrolled_root).resolve()
    controlled_root = Path(args.background_controlled_root).resolve()
    output_root = Path(args.output_root).resolve()
    for root in (uncontrolled_root, controlled_root):
        if not (root / "rgb").is_dir():
            raise FileNotFoundError(f"Evaluation RGB directory does not exist: {root / 'rgb'}")
        if not (root / "tif").is_dir():
            raise FileNotFoundError(f"Evaluation TIF directory does not exist: {root / 'tif'}")

    metric_device = torch.device(DEVICE)
    if metric_device.type == "cuda" and not torch.cuda.is_available():
        metric_device = torch.device("cpu")
    tif_reader = Tif_Read_and_Write()

    dataset = IMPGM_Dataset(
        img_rootdir_list=list(DATASET_DICT["img_rootdir_list_forTest"]),
        msk_rootdir_list=list(DATASET_DICT["msk_rootdir_list_forTest"]),
        is_train=False,
    )
    batch_size = int(
        TEST_BATCH_SIZE if args.batch_size is None else args.batch_size
    )
    microbatch_size = int(
        DIFFUSION_TEST_MICROBATCH_SIZE
        if args.batch_size is None
        else batch_size
    )
    candidates = []
    for index in range(len(dataset)):
        source_image = Path(dataset.resolve_image_path(index)).resolve()
        label = source_image.stem.split("_")[-2]
        batch_index = index // batch_size
        position = index % batch_size
        batch_length = min(batch_size, len(dataset) - batch_index * batch_size)
        rgb_name = _output_name(batch_index, position, batch_length, label, ".png")
        tif_name = _output_name(batch_index, position, batch_length, label, ".tif")
        uncontrolled_rgb = uncontrolled_root / "rgb" / rgb_name
        controlled_rgb = controlled_root / "rgb" / rgb_name
        uncontrolled_tif = uncontrolled_root / "tif" / tif_name
        controlled_tif = controlled_root / "tif" / tif_name
        if not all(
            path.is_file()
            for path in (
                uncontrolled_rgb,
                controlled_rgb,
                uncontrolled_tif,
                controlled_tif,
            )
        ):
            continue
        candidates.append(
            {
                "dataset_index": index,
                "batch_index": batch_index,
                "batch_position": position,
                "batch_length": batch_length,
                "label": label,
                "source_image": source_image,
                "uncontrolled_rgb": uncontrolled_rgb,
                "background_controlled_rgb": controlled_rgb,
                "uncontrolled_tif": uncontrolled_tif,
                "background_controlled_tif": controlled_tif,
            }
        )

    selected = _balanced_selection(candidates, args.num_samples, args.seed)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for archive_index, candidate in enumerate(selected, start=1):
        sample_dir = output_root / f"sample_{archive_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        dataset_index = candidate["dataset_index"]
        label, _, mask, target, _, _ = dataset[dataset_index]
        if label != candidate["label"]:
            raise RuntimeError(
                f"Label mismatch at dataset index {dataset_index}: "
                f"{label!r} != {candidate['label']!r}."
            )

        reference_path = sample_dir / "reference_rgb.png"
        mask_path = sample_dir / "condition_mask.png"
        high_frequency_path = sample_dir / "background_high_frequency.png"
        uncontrolled_path = sample_dir / "uncontrolled_rgb.png"
        controlled_path = sample_dir / "background_controlled_rgb.png"
        reference_tif_path = sample_dir / "reference.tif"
        uncontrolled_tif_path = sample_dir / "uncontrolled.tif"
        controlled_tif_path = sample_dir / "background_controlled.tif"
        metrics_path = sample_dir / "metrics.json"
        comparison_path = sample_dir / "comparison_rgb.png"

        _save_tensor_image(prepare_rgb_vis_tensor(target), reference_path)
        mask_vis = prepare_mask_vis_tensor(mask).unsqueeze(0)
        _save_tensor_image(mask_vis, mask_path)

        target_batch = target.unsqueeze(0)
        mask_batch = mask.unsqueeze(0)
        high_frequency = extract_high_frequency(
            target_batch,
            IMGSYN_CONTROLNET_CONFIG.CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE,
        )
        background_high_frequency = high_frequency * (1.0 - mask_batch)
        high_frequency_vis = prepare_high_freq_vis_tensor(background_high_frequency)[0]
        _save_tensor_image(high_frequency_vis, high_frequency_path)

        shutil.copy2(candidate["uncontrolled_rgb"], uncontrolled_path)
        shutil.copy2(candidate["background_controlled_rgb"], controlled_path)
        shutil.copy2(candidate["source_image"], reference_tif_path)
        shutil.copy2(candidate["uncontrolled_tif"], uncontrolled_tif_path)
        shutil.copy2(candidate["background_controlled_tif"], controlled_tif_path)

        reference_metric = (
            denormalize_image_tensor(target.unsqueeze(0).float())
            .clamp(0.0, 1.0)
            .to(metric_device)
        )
        uncontrolled_metric = (
            _load_tif_tensor(candidate["uncontrolled_tif"], tif_reader)
            .clamp(0.0, 1.0)
            .to(metric_device)
        )
        controlled_metric = (
            _load_tif_tensor(candidate["background_controlled_tif"], tif_reader)
            .clamp(0.0, 1.0)
            .to(metric_device)
        )
        if uncontrolled_metric.shape != reference_metric.shape:
            raise ValueError(
                "Uncontrolled/reference shape mismatch for "
                f"{candidate['source_image']}: {tuple(uncontrolled_metric.shape)} "
                f"!= {tuple(reference_metric.shape)}"
            )
        if controlled_metric.shape != reference_metric.shape:
            raise ValueError(
                "Background-controlled/reference shape mismatch for "
                f"{candidate['source_image']}: {tuple(controlled_metric.shape)} "
                f"!= {tuple(reference_metric.shape)}"
            )
        metrics = {
            "uncontrolled": _compute_metrics(
                uncontrolled_metric,
                reference_metric,
                compute_GMSD,
                compute_FSIM,
            ),
            "background_controlled": _compute_metrics(
                controlled_metric,
                reference_metric,
                compute_GMSD,
                compute_FSIM,
            ),
        }
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _save_comparison_figure(
            reference_path,
            uncontrolled_path,
            controlled_path,
            metrics["uncontrolled"],
            metrics["background_controlled"],
            comparison_path,
        )

        source_mask = None
        if not label.startswith("No"):
            source_mask = str(
                Path(
                    _resolve_mask_path(
                        candidate["source_image"], dataset._mask_name_to_path
                    )
                ).resolve()
            )
        mb_start = (candidate["batch_position"] // microbatch_size) * microbatch_size
        sampling_seed = int(
            DRAW_RANDOM_SEED + candidate["batch_index"] * batch_size + mb_start
        )
        records.append(
            {
                "archive_id": f"sample_{archive_index:03d}",
                "dataset": DATASET_NAME,
                "dataset_index": dataset_index,
                "label": label,
                "sampling_seed": sampling_seed,
                "source_image": str(candidate["source_image"]),
                "source_mask": source_mask,
                "source_uncontrolled_rgb": str(candidate["uncontrolled_rgb"]),
                "source_background_controlled_rgb": str(
                    candidate["background_controlled_rgb"]
                ),
                "source_uncontrolled_tif": str(candidate["uncontrolled_tif"]),
                "source_background_controlled_tif": str(
                    candidate["background_controlled_tif"]
                ),
                "reference_rgb": str(reference_path),
                "reference_tif": str(reference_tif_path),
                "condition_mask": str(mask_path),
                "background_high_frequency": str(high_frequency_path),
                "uncontrolled_rgb": str(uncontrolled_path),
                "background_controlled_rgb": str(controlled_path),
                "uncontrolled_tif": str(uncontrolled_tif_path),
                "background_controlled_tif": str(controlled_tif_path),
                "metrics": metrics,
                "metrics_file": str(metrics_path),
                "comparison_rgb": str(comparison_path),
            }
        )

    metrics_csv, metrics_summary = _write_metric_outputs(records, output_root)
    manifest = {
        "dataset": DATASET_NAME,
        "dataset_yaml": str(dataset_yaml),
        "selection_seed": int(args.seed),
        "evaluation_batch_size": batch_size,
        "num_samples": len(records),
        "selection": "deterministic_class_balanced_from_aligned_outputs",
        "metric_reference": "paired_against_the_same_real_reference_image",
        "metric_input": "denormalized_full_band_tif_clamped_to_0_1",
        "representative_metrics_csv": str(metrics_csv),
        "representative_metrics_summary": str(metrics_summary),
        "samples": records,
    }
    manifest_path = output_root / "representative_samples_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
