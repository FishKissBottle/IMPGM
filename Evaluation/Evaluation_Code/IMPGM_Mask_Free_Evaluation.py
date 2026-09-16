from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import (
    DATASET_NAME,
    DEVICE,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_CHANNELS,
    PROMPT_DICT,
)
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    GenerationRecord,
    _read_tif,
    _stack_images,
    compute_distribution_metrics,
    compute_diversity_metrics,
    load_generation_manifest,
    validate_generation_protocol,
)
from Evaluation.Evaluation_Code.IMPGM_Quality_Metrics import (
    compute_histogram_W1,
    compute_mean_spectrum_SAM_degrees,
    require_lpips,
)
from IMPGM_Utils import (
    load_model_for_eval,
    prepare_rgb_vis_tensor,
    save_msk_datas,
    save_rgb_datas,
    save_tif_datas,
)
from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_Config import (
    BASE_CHANNELS as EVALUATION_UNET_BASE_CHANNELS,
    CONDITION_CHANNELS as EVALUATION_UNET_CONDITION_CHANNELS,
    MODEL_PATH as EVALUATION_UNET_MODEL_PATH,
    number_of_classes as evaluation_unet_number_of_classes,
)
from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_model import EvaluationUNet


MASK_FREE_METHOD = "Full_IMPGM_Mask_Free"
MASK_FREE_PROTOCOL_VERSION = 1
MASK_FREE_METRICS_FILENAME = "generation_metrics.json"


def _safe_token(value):
    token = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return token.strip("_") or "sample"


def _mask_free_record_id(condition_id, fggen_seed, imgsyn_seed):
    payload = f"{MASK_FREE_METHOD}:{condition_id}:{fggen_seed}:{imgsyn_seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


class MaskFreeGenerationManifestWriter:
    """Save autonomous image-mask pairs and append their JSONL provenance."""

    def __init__(
        self,
        output_root,
        *,
        dataset,
        split,
        checkpoint,
        protocol,
        training_regime,
        preload_sources,
        vae_checkpoint,
        fggen_checkpoint,
        imgsyn_checkpoint,
        fgseg_checkpoint,
        latent_scaling_factor,
        sampler,
        diffusion_steps,
        overwrite=True,
    ):
        self.output_root = Path(output_root).resolve()
        self.dataset = str(dataset)
        self.split = str(split)
        self.checkpoint = str(checkpoint)
        self.protocol = str(protocol)
        self.training_regime = str(training_regime)
        self.preload_sources = preload_sources
        self.vae_checkpoint = str(vae_checkpoint)
        self.fggen_checkpoint = str(fggen_checkpoint)
        self.imgsyn_checkpoint = str(imgsyn_checkpoint)
        self.fgseg_checkpoint = str(fgseg_checkpoint)
        self.latent_scaling_factor = float(latent_scaling_factor)
        self.sampler = str(sampler)
        self.diffusion_steps = int(diffusion_steps)
        self.generated_dir = self.output_root / "generated_tif"
        self.rgb_dir = self.output_root / "generated_rgb"
        self.generated_mask_dir = self.output_root / "generated_masks_tif"
        self.generated_mask_png_dir = self.output_root / "generated_masks_png"
        self.foreground_dir = self.output_root / "foreground_tif"
        self.foreground_rgb_dir = self.output_root / "foreground_rgb"
        self.reference_dir = self.output_root / "reference_tif"
        self.reference_mask_dir = self.output_root / "reference_masks_tif"
        for directory in (
            self.generated_dir,
            self.rgb_dir,
            self.generated_mask_dir,
            self.generated_mask_png_dir,
            self.foreground_dir,
            self.foreground_rgb_dir,
            self.reference_dir,
            self.reference_mask_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_root / MANIFEST_FILENAME
        if overwrite:
            self.manifest_path.write_text("", encoding="utf-8")

    def append_sample(
        self,
        *,
        prediction,
        foreground,
        generated_mask,
        label,
        label_id,
        condition_id,
        fggen_seed,
        imgsyn_seed,
        seed_rank,
        reference=None,
        reference_mask=None,
        reference_projection=None,
        reference_geotransform=None,
        reference_source_image=None,
    ):
        prediction = prediction.detach().float().cpu().clamp(0.0, 1.0)
        foreground = foreground.detach().float().cpu().clamp(0.0, 1.0)
        generated_mask = generated_mask.detach().float().cpu()
        if generated_mask.ndim == 2:
            generated_mask = generated_mask.unsqueeze(0)
        generated_mask = (generated_mask[:1] > 0.5).float()
        if prediction.ndim != 3 or prediction.shape[0] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected one {INPUT_CHANNELS}-band [C,H,W] prediction, got {prediction.shape}."
            )
        if foreground.shape != prediction.shape:
            raise ValueError("Foreground and generated image must have identical shapes.")
        if generated_mask.shape[-2:] != prediction.shape[-2:]:
            raise ValueError("Generated mask and image must have identical spatial shapes.")
        if (reference is None) != (reference_mask is None):
            raise ValueError("Reference image and reference mask must be provided together.")

        rid = _mask_free_record_id(condition_id, fggen_seed, imgsyn_seed)
        stem = (
            f"{_safe_token(condition_id)}_{_safe_token(label)}_"
            f"fg{int(fggen_seed)}_im{int(imgsyn_seed)}_{rid}"
        )
        generated_path = self.generated_dir / f"{stem}.tif"
        rgb_path = self.rgb_dir / f"{stem}.png"
        generated_mask_path = self.generated_mask_dir / f"{stem}.tif"
        generated_mask_png_path = self.generated_mask_png_dir / f"{stem}.png"
        foreground_path = self.foreground_dir / f"{stem}.tif"
        foreground_rgb_path = self.foreground_rgb_dir / f"{stem}.png"

        save_tif_datas(
            prediction.unsqueeze(0),
            savepath=str(generated_path),
            denormalize=False,
        )
        save_rgb_datas(
            prepare_rgb_vis_tensor(prediction.unsqueeze(0), denormalize=False),
            nrow=1,
            savepath=str(rgb_path),
            format="PNG",
            is_makegrid=False,
        )
        save_tif_datas(
            generated_mask.unsqueeze(0),
            savepath=str(generated_mask_path),
            denormalize=False,
            nodata_value=0.0,
        )
        save_msk_datas(
            generated_mask.unsqueeze(0),
            savepath=str(generated_mask_png_path),
            is_makegrid=True,
            nrow=1,
        )
        save_tif_datas(
            foreground.unsqueeze(0),
            savepath=str(foreground_path),
            denormalize=False,
        )
        save_rgb_datas(
            prepare_rgb_vis_tensor(foreground.unsqueeze(0), denormalize=False),
            nrow=1,
            savepath=str(foreground_rgb_path),
            format="PNG",
            is_makegrid=False,
        )

        reference_path = None
        reference_mask_path = None
        if reference is not None:
            reference = reference.detach().float().cpu().clamp(0.0, 1.0)
            reference_mask = reference_mask.detach().float().cpu()
            if reference_mask.ndim == 2:
                reference_mask = reference_mask.unsqueeze(0)
            reference_mask = (reference_mask[:1] > 0.5).float()
            if reference.shape != prediction.shape:
                raise ValueError("Reference and generated image must have identical shapes.")
            projections = (
                None
                if not isinstance(reference_projection, str) or not reference_projection.strip()
                else [reference_projection]
            )
            geotransforms = (
                None
                if reference_geotransform is None
                else [list(reference_geotransform)]
            )
            reference_path = self.reference_dir / f"{stem}.tif"
            reference_mask_path = self.reference_mask_dir / f"{stem}.tif"
            save_tif_datas(
                reference.unsqueeze(0),
                projections=projections,
                geotransforms=geotransforms,
                savepath=str(reference_path),
                denormalize=False,
            )
            save_tif_datas(
                reference_mask.unsqueeze(0),
                projections=projections,
                geotransforms=geotransforms,
                savepath=str(reference_mask_path),
                denormalize=False,
                nodata_value=0.0,
            )

        record = GenerationRecord(
            record_id=rid,
            method=MASK_FREE_METHOD,
            dataset=self.dataset,
            split=self.split,
            label=str(label),
            label_id=int(label_id),
            condition_id=str(condition_id),
            seed=int(fggen_seed),
            seed_rank=int(seed_rank),
            primary=bool(int(seed_rank) == 0),
            deterministic=False,
            protocol=self.protocol,
            checkpoint=self.checkpoint,
            generated_rgb=str(rgb_path),
            generated_tif=str(generated_path),
            reference_tif=None if reference_path is None else str(reference_path),
            condition_mask_tif=str(generated_mask_path),
            source_image=None,
            training_regime=self.training_regime,
            preload_sources=self.preload_sources,
            fggen_seed=int(fggen_seed),
            imgsyn_seed=int(imgsyn_seed),
            external_mask_used=False,
            mask_source="fgseg_generated",
            generated_mask_tif=str(generated_mask_path),
            generated_mask_png=str(generated_mask_png_path),
            reference_mask_tif=(
                None if reference_mask_path is None else str(reference_mask_path)
            ),
            foreground_tif=str(foreground_path),
            foreground_rgb=str(foreground_rgb_path),
            vae_checkpoint=self.vae_checkpoint,
            fggen_checkpoint=self.fggen_checkpoint,
            imgsyn_checkpoint=self.imgsyn_checkpoint,
            fgseg_checkpoint=self.fgseg_checkpoint,
            latent_scaling_factor=self.latent_scaling_factor,
            sampler=self.sampler,
            diffusion_steps=self.diffusion_steps,
            reference_source_image=(
                None if reference_source_image is None else str(reference_source_image)
            ),
        )
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record


def validate_mask_free_generation_protocol(records):
    """Validate autonomous-generation provenance before metric computation."""
    if not records:
        raise ValueError("Mask-free generation manifest is empty.")
    if {record.method for record in records} != {MASK_FREE_METHOD}:
        raise ValueError(
            f"Mask-free evaluation requires method={MASK_FREE_METHOD!r}."
        )
    for record in records:
        if record.external_mask_used is not False:
            raise ValueError(
                f"Mask-free record {record.record_id} must set external_mask_used=false."
            )
        if record.mask_source != "fgseg_generated":
            raise ValueError(
                f"Mask-free record {record.record_id} must use an FgSeg-generated mask."
            )
        if record.fggen_seed is None or record.imgsyn_seed is None:
            raise ValueError(
                f"Mask-free record {record.record_id} must record both diffusion seeds."
            )
        if int(record.seed) != int(record.fggen_seed):
            raise ValueError(
                f"Mask-free record {record.record_id} has inconsistent primary/FgGen seeds."
            )
        if record.generated_mask_tif is None:
            raise ValueError(
                f"Mask-free record {record.record_id} has no generated mask."
            )
        if Path(record.condition_mask_tif).resolve() != Path(record.generated_mask_tif).resolve():
            raise ValueError(
                f"Mask-free record {record.record_id} must expose its generated mask through "
                "condition_mask_tif for common evaluators."
            )
        if record.source_image is not None:
            raise ValueError(
                f"Mask-free record {record.record_id} may not bind a real source as a condition."
            )
    report = validate_generation_protocol(records)
    report.update({
        "mask_free_protocol_version": MASK_FREE_PROTOCOL_VERSION,
        "external_mask_used": False,
        "mask_source": "fgseg_generated",
    })
    return report


def compute_mask_free_spectral_metrics(records, bins=256):
    primary = [record for record in records if record.primary]
    class_results = {}
    for label in sorted({record.label for record in primary}):
        group = [record for record in primary if record.label == label]
        generated_sum = torch.zeros(INPUT_CHANNELS)
        reference_sum = torch.zeros(INPUT_CHANNELS)
        generated_hist = torch.zeros(INPUT_CHANNELS, bins)
        reference_hist = torch.zeros(INPUT_CHANNELS, bins)
        generated_support = 0
        reference_support = 0
        for record in tqdm(
            group,
            desc=f"Mask-free spectral {label}",
            unit="image",
            leave=False,
            dynamic_ncols=True,
        ):
            if record.reference_tif is None or record.reference_mask_tif is None:
                raise ValueError("Mask-free spectral evaluation requires test references.")
            generated = _read_tif(record.generated_tif).float().clamp(0.0, 1.0)
            reference = _read_tif(record.reference_tif).float().clamp(0.0, 1.0)
            generated_mask = _read_tif(record.generated_mask_tif)[0] > 0.5
            reference_mask = _read_tif(record.reference_mask_tif)[0] > 0.5
            if not torch.any(generated_mask):
                generated_mask = torch.ones_like(generated_mask, dtype=torch.bool)
            if not torch.any(reference_mask):
                reference_mask = torch.ones_like(reference_mask, dtype=torch.bool)
            generated_pixels = generated[:, generated_mask]
            reference_pixels = reference[:, reference_mask]
            generated_support += int(generated_mask.sum().item())
            reference_support += int(reference_mask.sum().item())
            generated_sum += generated_pixels.sum(dim=1)
            reference_sum += reference_pixels.sum(dim=1)
            for channel in range(INPUT_CHANNELS):
                generated_hist[channel] += torch.histc(
                    generated_pixels[channel], bins=bins, min=0.0, max=1.0
                )
                reference_hist[channel] += torch.histc(
                    reference_pixels[channel], bins=bins, min=0.0, max=1.0
                )
        generated_mean = generated_sum / max(generated_support, 1)
        reference_mean = reference_sum / max(reference_support, 1)
        w1 = [
            compute_histogram_W1(generated_hist[channel], reference_hist[channel], bins)
            for channel in range(INPUT_CHANNELS)
        ]
        sam_value = compute_mean_spectrum_SAM_degrees(generated_mean, reference_mean)
        class_results[label] = {
            "n": len(group),
            "generated_pixel_support": generated_support,
            "reference_pixel_support": reference_support,
            "mean_spectrum_generated": generated_mean.tolist(),
            "mean_spectrum_reference": reference_mean.tolist(),
            "mean_spectrum_sam_deg": sam_value,
            "mean_spectrum_sam_status": (
                "ok" if sam_value is not None else "undefined_zero_norm"
            ),
            "per_band_w1": w1,
            "mean_w1": float(np.mean(w1)),
        }
    if not class_results:
        return {"per_class": {}, "macro": None, "support_weighted": None}
    valid_sam = [
        value for value in class_results.values()
        if value["mean_spectrum_sam_deg"] is not None
    ]
    sams = np.asarray(
        [value["mean_spectrum_sam_deg"] for value in valid_sam],
        dtype=np.float64,
    )
    w1s = np.asarray(
        [value["mean_w1"] for value in class_results.values()],
        dtype=np.float64,
    )
    weights = np.asarray(
        [value["reference_pixel_support"] for value in class_results.values()],
        dtype=np.float64,
    )
    weights = weights / max(weights.sum(), 1.0)
    valid_sam_weights = np.asarray(
        [value["reference_pixel_support"] for value in valid_sam],
        dtype=np.float64,
    )
    if valid_sam_weights.size:
        valid_sam_weights = valid_sam_weights / max(valid_sam_weights.sum(), 1.0)
    return {
        "per_class": class_results,
        "macro": {
            "mean_spectrum_sam_deg": None if sams.size == 0 else float(sams.mean()),
            "sam_valid_classes": int(sams.size),
            "mean_w1": float(w1s.mean()),
        },
        "support_weighted": {
            "mean_spectrum_sam_deg": (
                None if sams.size == 0 else float(np.sum(sams * valid_sam_weights))
            ),
            "sam_valid_classes": int(sams.size),
            "mean_w1": float(np.sum(w1s * weights)),
        },
        "mask_protocol": "generated_uses_fgseg_mask;reference_uses_real_mask",
    }


def _empty_pair_totals():
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0}


def _update_pair_totals(totals, prediction, target):
    totals["tp"] += float((prediction & target).sum().item())
    totals["fp"] += float((prediction & ~target).sum().item())
    totals["fn"] += float((~prediction & target).sum().item())


def _pair_scores(totals):
    eps = 1.0e-8
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    iou = tp / (tp + fp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(dice),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
    }


@torch.no_grad()
def compute_mask_free_pair_consistency(records, checkpoint_path, batch_size=8):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Evaluation_UNet checkpoint does not exist: {checkpoint_path}")
    model = EvaluationUNet(
        image_channels=INPUT_CHANNELS,
        num_classes=evaluation_unet_number_of_classes(),
        base_channels=EVALUATION_UNET_BASE_CHANNELS,
        condition_channels=EVALUATION_UNET_CONDITION_CHANNELS,
    ).to(DEVICE)
    model, checkpoint_metadata = load_model_for_eval(
        str(checkpoint_path), model, map_location=DEVICE
    )
    if checkpoint_metadata.get("dataset_name") != DATASET_NAME:
        raise ValueError(
            "Evaluation_UNet dataset mismatch: "
            f"checkpoint={checkpoint_metadata.get('dataset_name')!r}, active={DATASET_NAME!r}."
        )
    if int(checkpoint_metadata.get("num_classes") or -1) != len(PROMPT_DICT):
        raise ValueError("Evaluation_UNet class-count mismatch.")
    if checkpoint_metadata.get("input_domain") != "normalized_complete_image":
        raise ValueError("Evaluation_UNet checkpoint has an incompatible input domain.")
    model.eval()
    mean = torch.tensor(IMAGE_MEAN, device=DEVICE).view(1, -1, 1, 1)
    std = torch.tensor(IMAGE_STD, device=DEVICE).view(1, -1, 1, 1)
    generated_totals = _empty_pair_totals()
    reference_totals = _empty_pair_totals()
    per_class_generated = defaultdict(_empty_pair_totals)
    primary = [record for record in records if record.primary]
    step = max(1, int(batch_size))
    starts = range(0, len(primary), step)
    for start in tqdm(
        starts,
        total=math.ceil(len(primary) / step),
        desc="Mask-free pair consistency",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        group = primary[start:start + step]
        generated = _stack_images(group, "generated_tif").to(DEVICE).float().clamp(0.0, 1.0)
        generated_masks = _stack_images(group, "generated_mask_tif")[:, :1].to(DEVICE) > 0.5
        label_ids = torch.tensor(
            [record.label_id for record in group], device=DEVICE, dtype=torch.long
        )
        generated_pred = torch.sigmoid(
            model((generated - mean) / std, label_ids).float()
        ) >= 0.5
        if generated_pred.shape != generated_masks.shape:
            raise ValueError("Evaluation_UNet output and generated masks have different shapes.")
        _update_pair_totals(generated_totals, generated_pred, generated_masks)
        for index, record in enumerate(group):
            _update_pair_totals(
                per_class_generated[record.label],
                generated_pred[index:index + 1],
                generated_masks[index:index + 1],
            )

        if all(record.reference_tif and record.reference_mask_tif for record in group):
            reference = _stack_images(group, "reference_tif").to(DEVICE).float().clamp(0.0, 1.0)
            reference_masks = _stack_images(group, "reference_mask_tif")[:, :1].to(DEVICE) > 0.5
            reference_pred = torch.sigmoid(
                model((reference - mean) / std, label_ids).float()
            ) >= 0.5
            _update_pair_totals(reference_totals, reference_pred, reference_masks)

    return {
        "generated_pair_consistency": _pair_scores(generated_totals),
        "generated_pair_consistency_per_class": {
            label: _pair_scores(per_class_generated[label])
            for label in sorted(per_class_generated)
        },
        "real_image_evaluator_ceiling": _pair_scores(reference_totals),
        "protocol": "evaluation_unet_prediction_vs_fgseg_generated_mask",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_metadata": {
            "dataset_name": checkpoint_metadata.get("dataset_name"),
            "num_classes": checkpoint_metadata.get("num_classes"),
            "input_domain": checkpoint_metadata.get("input_domain"),
            "conditioning": checkpoint_metadata.get("conditioning"),
        },
    }


def evaluate_mask_free_generation_manifest(
    manifest_path,
    *,
    output_path=None,
    seed=0,
    max_swd_samples=128,
    evaluation_unet_checkpoint="auto",
    skip_lpips=False,
):
    """Evaluate one existing autonomous image-mask generation manifest."""
    manifest_path = Path(manifest_path).resolve()
    records = load_generation_manifest(manifest_path)
    datasets = sorted({record.dataset for record in records})
    splits = sorted({record.split for record in records})
    if len(datasets) != 1 or len(splits) != 1:
        raise ValueError("One mask-free manifest must contain one dataset and one split.")
    if datasets[0] != DATASET_NAME:
        raise ValueError(
            f"Manifest dataset {datasets[0]!r} does not match active dataset {DATASET_NAME!r}."
        )
    if any(record.reference_tif is None or record.reference_mask_tif is None for record in records):
        raise ValueError("Mask-free generation quality evaluation requires test references.")
    if not skip_lpips:
        require_lpips()
    protocol_validation = validate_mask_free_generation_protocol(records)
    if evaluation_unet_checkpoint == "auto":
        evaluation_unet_checkpoint = EVALUATION_UNET_MODEL_PATH
    if evaluation_unet_checkpoint is not None:
        evaluation_unet_checkpoint = Path(evaluation_unet_checkpoint).resolve()
        if not evaluation_unet_checkpoint.is_file():
            raise FileNotFoundError(
                "Evaluation_UNet checkpoint does not exist: "
                f"{evaluation_unet_checkpoint}"
            )
    progress = tqdm(total=4, desc="Mask-free evaluation: distribution", unit="stage", dynamic_ncols=True)
    distribution = compute_distribution_metrics(
        records,
        manifest_path.parent / "feature_cache",
        seed=seed,
        max_swd_samples=max_swd_samples,
    )
    progress.update(1)
    progress.set_description("Mask-free evaluation: spectral")
    spectral = compute_mask_free_spectral_metrics(records)
    progress.update(1)
    progress.set_description("Mask-free evaluation: diversity")
    diversity = (
        {"status": "skipped", "reason": "disabled_by_user"}
        if skip_lpips
        else compute_diversity_metrics(records)
    )
    progress.update(1)
    progress.set_description("Mask-free evaluation: pair consistency")
    if evaluation_unet_checkpoint is None:
        pair_consistency = {
            "status": "not_computed",
            "reason": "disabled_by_user",
            "checkpoint": None,
        }
    else:
        pair_consistency = compute_mask_free_pair_consistency(
            records, evaluation_unet_checkpoint
        )
    progress.update(1)
    progress.close()

    result = {
        "manifest_version": MANIFEST_VERSION,
        "mask_free_protocol_version": MASK_FREE_PROTOCOL_VERSION,
        "method": MASK_FREE_METHOD,
        "dataset": datasets[0],
        "split": splits[0],
        "protocols": sorted({record.protocol for record in records}),
        "training_regimes": sorted({record.training_regime for record in records}),
        "preload_sources": sorted({
            record.preload_sources for record in records if record.preload_sources is not None
        }),
        "checkpoints": sorted({record.checkpoint for record in records}),
        "num_primary_samples": sum(record.primary for record in records),
        "num_total_samples": len(records),
        "num_conditions": len({record.condition_id for record in records}),
        "per_class_primary_counts": {
            label: sum(record.primary and record.label == label for record in records)
            for label in sorted({record.label for record in records})
        },
        "fggen_seeds": sorted({int(record.fggen_seed) for record in records}),
        "imgsyn_seeds": sorted({int(record.imgsyn_seed) for record in records}),
        "protocol_validation": protocol_validation,
        "distribution": distribution,
        "spectral_distribution": spectral,
        "diversity": diversity,
        "image_mask_pair_consistency": pair_consistency,
    }
    output_path = Path(output_path or manifest_path.parent / MASK_FREE_METRICS_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_mask_free_generation_evaluation_cli(default_manifest=None, description=None):
    parser = argparse.ArgumentParser(
        description=description or "Evaluate an IMPGM mask-free generation manifest."
    )
    parser.add_argument("--manifest", default=None if default_manifest is None else str(default_manifest))
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-swd-samples", type=int, default=128)
    parser.add_argument("--evaluation-unet-checkpoint", default="auto")
    parser.add_argument("--no-condition-segmentation", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    if args.manifest is None:
        parser.error("--manifest is required.")
    result = evaluate_mask_free_generation_manifest(
        args.manifest,
        output_path=args.output,
        seed=args.seed,
        max_swd_samples=args.max_swd_samples,
        skip_lpips=args.skip_lpips,
        evaluation_unet_checkpoint=(
            None if args.no_condition_segmentation else args.evaluation_unet_checkpoint
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
