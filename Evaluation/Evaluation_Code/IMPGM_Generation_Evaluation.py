from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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
    VIS_BAND_ORDER,
)
from Evaluation.Evaluation_Code.IMPGM_Quality_Metrics import (
    build_inception_feature_extractor,
    compute_FID_from_features,
    compute_KID_from_features,
    compute_LPIPS_diversity,
    compute_histogram_W1,
    compute_mean_spectrum_SAM_degrees,
    compute_multiband_SWD,
    extract_inception_features,
    require_lpips,
)
from IMPGM_TifReader import Tif_Read_and_Write
from IMPGM_Utils import (
    load_model_for_eval,
    prepare_rgb_vis_tensor,
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


MANIFEST_FILENAME = "generation_manifest.jsonl"
METRICS_FILENAME = "generation_metrics.json"
MANIFEST_VERSION = 2
GENERATION_BASE_SEED = 999
GENERATION_SEED_CONDITION_STRIDE = 1009
GENERATION_SEED_RANK_STRIDE = 1000003
DEFAULT_DOWNSTREAM_CONDITION_FRACTION = 0.15
CONDITION_SELECTION_VERSION = 1


def _condition_fraction_token(sample_fraction):
    return f"{float(sample_fraction):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def default_condition_selection_path(dataset, split, sample_fraction, seed=GENERATION_BASE_SEED):
    return (
        Path(__file__).resolve().parent
        / "Evaluation"
        / "Condition_Selections"
        / str(dataset)
        / (
            f"{str(split).lower()}_fraction_{_condition_fraction_token(sample_fraction)}_"
            f"seed_{int(seed)}.json"
        )
    )


def resolve_condition_selection(
    candidates,
    *,
    dataset,
    split,
    sample_fraction=None,
    selection_path=None,
    seed=GENERATION_BASE_SEED,
):
    """Resolve one deterministic class-stratified condition subset.

    Train-split calls default to fifteen percent and share a persisted selection file.
    Other splits keep all conditions unless a fraction is explicitly requested.
    """
    split = str(split).strip().lower()
    effective_fraction = (
        DEFAULT_DOWNSTREAM_CONDITION_FRACTION
        if sample_fraction is None and split == "train"
        else 1.0 if sample_fraction is None else float(sample_fraction)
    )
    if not 0.0 < effective_fraction <= 1.0:
        raise ValueError(
            f"sample_fraction must be in (0, 1], got {effective_fraction}."
        )

    normalized = []
    seen_condition_ids = set()
    for candidate in candidates:
        condition_id = str(candidate["condition_id"]).strip()
        label = str(candidate["label"]).strip()
        if not condition_id or not label:
            raise ValueError("Condition candidates require non-empty condition_id and label.")
        if condition_id in seen_condition_ids:
            raise ValueError(f"Duplicate condition_id in candidate set: {condition_id!r}.")
        seen_condition_ids.add(condition_id)
        normalized.append({
            "index": int(candidate["index"]),
            "condition_id": condition_id,
            "label": label,
            "source_image": str(candidate.get("source_image", "")),
        })
    if not normalized:
        raise RuntimeError(f"The {split!r} split contains no condition candidates.")

    signature_rows = sorted(
        (item["condition_id"], item["label"], item["source_image"])
        for item in normalized
    )
    candidate_signature = hashlib.sha256(
        json.dumps(signature_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    candidate_by_id = {item["condition_id"]: item for item in normalized}

    resolved_selection_path = None
    if selection_path is not None:
        resolved_selection_path = Path(selection_path).resolve()
    elif split == "train":
        resolved_selection_path = default_condition_selection_path(
            dataset,
            split,
            effective_fraction,
            seed,
        ).resolve()

    if resolved_selection_path is not None and resolved_selection_path.is_file():
        payload = json.loads(resolved_selection_path.read_text(encoding="utf-8"))
        expected = {
            "version": CONDITION_SELECTION_VERSION,
            "dataset": str(dataset),
            "split": split,
            "seed": int(seed),
            "candidate_signature": candidate_signature,
        }
        mismatches = [
            f"{key}: file={payload.get(key)!r}, expected={value!r}"
            for key, value in expected.items()
            if payload.get(key) != value
        ]
        stored_fraction = float(payload.get("sample_fraction", -1.0))
        if not math.isclose(stored_fraction, effective_fraction, rel_tol=0.0, abs_tol=1.0e-12):
            mismatches.append(
                f"sample_fraction: file={stored_fraction!r}, expected={effective_fraction!r}"
            )
        if mismatches:
            raise ValueError(
                f"Condition selection is incompatible with the active protocol: "
                f"{resolved_selection_path}\n  - " + "\n  - ".join(mismatches)
            )
        selected_ids = [str(value) for value in payload.get("selected_condition_ids", [])]
        if not selected_ids:
            raise ValueError(f"Condition selection contains no samples: {resolved_selection_path}")
        missing_ids = sorted(set(selected_ids) - set(candidate_by_id))
        if missing_ids:
            raise ValueError(
                "Condition selection references samples absent from the active dataset: "
                f"{missing_ids[:10]}."
            )
    else:
        grouped = defaultdict(list)
        for item in normalized:
            grouped[item["label"]].append(item)
        target_total = max(
            len(grouped),
            min(len(normalized), int(round(len(normalized) * effective_fraction))),
        )
        raw_quotas = {
            label: len(items) * effective_fraction
            for label, items in grouped.items()
        }
        quotas = {
            label: max(1, int(math.floor(raw_quotas[label])))
            for label in grouped
        }
        remaining = target_total - sum(quotas.values())
        allocation_order = sorted(
            grouped,
            key=lambda label: (
                -(raw_quotas[label] - math.floor(raw_quotas[label])),
                label,
            ),
        )
        while remaining > 0:
            allocated = False
            for label in allocation_order:
                if remaining == 0:
                    break
                if quotas[label] < len(grouped[label]):
                    quotas[label] += 1
                    remaining -= 1
                    allocated = True
            if not allocated:
                break
        deallocation_order = list(reversed(allocation_order))
        while remaining < 0:
            deallocated = False
            for label in deallocation_order:
                if remaining == 0:
                    break
                if quotas[label] > 1:
                    quotas[label] -= 1
                    remaining += 1
                    deallocated = True
            if not deallocated:
                break

        selected_ids = []
        for label in sorted(grouped):
            ranked = sorted(
                grouped[label],
                key=lambda item: (
                    hashlib.sha256(
                        f"{int(seed)}:{label}:{item['condition_id']}".encode("utf-8")
                    ).digest(),
                    item["condition_id"],
                ),
            )
            selected_ids.extend(
                item["condition_id"] for item in ranked[:quotas[label]]
            )
        selected_ids = sorted(selected_ids)

        if resolved_selection_path is not None:
            per_class_selected = {
                label: sum(candidate_by_id[value]["label"] == label for value in selected_ids)
                for label in sorted(grouped)
            }
            payload = {
                "version": CONDITION_SELECTION_VERSION,
                "dataset": str(dataset),
                "split": split,
                "sample_fraction": effective_fraction,
                "seed": int(seed),
                "candidate_signature": candidate_signature,
                "num_candidates": len(normalized),
                "num_selected": len(selected_ids),
                "per_class_candidates": {
                    label: len(grouped[label]) for label in sorted(grouped)
                },
                "per_class_selected": per_class_selected,
                "selected_condition_ids": selected_ids,
            }
            resolved_selection_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = resolved_selection_path.with_suffix(
                resolved_selection_path.suffix + ".tmp"
            )
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(resolved_selection_path)

    selected_id_set = set(selected_ids)
    selected_indices = [
        item["index"] for item in normalized if item["condition_id"] in selected_id_set
    ]
    if len(selected_indices) != len(selected_ids):
        raise RuntimeError("Condition selection did not resolve to unique dataset indices.")
    return selected_indices, resolved_selection_path, effective_fraction


def build_generation_seed(condition_index, seed_rank):
    condition_index = int(condition_index)
    seed_rank = int(seed_rank)
    if condition_index < 0 or seed_rank < 0:
        raise ValueError("condition_index and seed_rank must be non-negative.")
    return int(
        GENERATION_BASE_SEED
        + condition_index * GENERATION_SEED_CONDITION_STRIDE
        + seed_rank * GENERATION_SEED_RANK_STRIDE
    )


@dataclass(frozen=True)
class GenerationRecord:
    """One generated sample and all information needed for fair evaluation."""

    record_id: str
    method: str
    dataset: str
    split: str
    label: str
    label_id: int
    condition_id: str
    seed: int
    seed_rank: int
    primary: bool
    deterministic: bool
    protocol: str
    checkpoint: str
    generated_rgb: str
    generated_tif: str
    reference_tif: str | None
    condition_mask_tif: str
    source_image: str | None = None
    training_regime: str = "unspecified"
    preload_sources: str | None = None
    fggen_seed: int | None = None
    imgsyn_seed: int | None = None
    external_mask_used: bool | None = None
    mask_source: str | None = None
    generated_mask_tif: str | None = None
    generated_mask_png: str | None = None
    reference_mask_tif: str | None = None
    foreground_tif: str | None = None
    foreground_rgb: str | None = None
    vae_checkpoint: str | None = None
    fggen_checkpoint: str | None = None
    imgsyn_checkpoint: str | None = None
    fgseg_checkpoint: str | None = None
    latent_scaling_factor: float | None = None
    sampler: str | None = None
    diffusion_steps: int | None = None
    reference_source_image: str | None = None
    requested_steps_per_stage: int | None = None
    actual_fggen_steps: int | None = None
    actual_imgsyn_steps: int | None = None
    actual_total_steps: int | None = None
    ddim_eta: float | None = None
    scheduler_type: str | None = None


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return token.strip("_") or "sample"


def _record_id(method: str, condition_id: str, seed: int) -> str:
    payload = f"{method}:{condition_id}:{int(seed)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


class GenerationManifestWriter:
    """Save physical-space evaluation samples and append a JSONL manifest."""

    def __init__(
        self,
        output_root,
        *,
        method,
        dataset,
        split,
        checkpoint,
        protocol="mask_label_to_image",
        training_regime="unspecified",
        preload_sources=None,
        sampler=None,
        diffusion_steps=None,
        requested_steps_per_stage=None,
        actual_fggen_steps=None,
        actual_imgsyn_steps=None,
        actual_total_steps=None,
        ddim_eta=None,
        scheduler_type=None,
        overwrite=True,
    ):
        self.output_root = Path(output_root).resolve()
        self.method = str(method)
        self.dataset = str(dataset)
        self.split = str(split)
        self.checkpoint = str(checkpoint)
        self.protocol = str(protocol)
        self.training_regime = str(training_regime)
        self.preload_sources = (
            None if preload_sources is None else str(preload_sources)
        )
        self.sampler = None if sampler is None else str(sampler)
        self.diffusion_steps = (
            None if diffusion_steps is None else int(diffusion_steps)
        )
        self.requested_steps_per_stage = (
            None
            if requested_steps_per_stage is None
            else int(requested_steps_per_stage)
        )
        self.actual_fggen_steps = (
            None if actual_fggen_steps is None else int(actual_fggen_steps)
        )
        self.actual_imgsyn_steps = (
            None if actual_imgsyn_steps is None else int(actual_imgsyn_steps)
        )
        self.actual_total_steps = (
            None if actual_total_steps is None else int(actual_total_steps)
        )
        self.ddim_eta = None if ddim_eta is None else float(ddim_eta)
        self.scheduler_type = (
            None if scheduler_type is None else str(scheduler_type)
        )
        self.generated_dir = self.output_root / "generated_tif"
        self.rgb_dir = self.output_root / "generated_rgb"
        self.reference_dir = self.output_root / "reference_tif"
        self.mask_dir = self.output_root / "condition_masks"
        for directory in (
            self.generated_dir,
            self.rgb_dir,
            self.reference_dir,
            self.mask_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_root / MANIFEST_FILENAME
        if overwrite:
            self.manifest_path.write_text("", encoding="utf-8")

    def append_sample(
        self,
        *,
        prediction,
        reference,
        condition_mask,
        label,
        label_id,
        condition_id,
        seed,
        seed_rank=0,
        deterministic=False,
        projection=None,
        geotransform=None,
        source_image=None,
    ) -> GenerationRecord:
        prediction = prediction.detach().float().cpu().clamp(0.0, 1.0)
        reference = reference.detach().float().cpu().clamp(0.0, 1.0)
        condition_mask = condition_mask.detach().float().cpu()
        if condition_mask.ndim == 2:
            condition_mask = condition_mask.unsqueeze(0)
        condition_mask = (condition_mask[:1] > 0.5).float()
        if prediction.shape != reference.shape:
            raise ValueError(
                f"Prediction/reference shape mismatch: {prediction.shape} vs {reference.shape}."
            )
        if prediction.ndim != 3 or prediction.shape[0] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected one {INPUT_CHANNELS}-band [C,H,W] prediction, got {prediction.shape}."
            )
        if condition_mask.shape[-2:] != prediction.shape[-2:]:
            raise ValueError("Condition mask and prediction must have the same spatial size.")

        rid = _record_id(self.method, condition_id, seed)
        stem = (
            f"{_safe_token(condition_id)}_{_safe_token(label)}_"
            f"s{int(seed)}_{rid}"
        )
        generated_path = self.generated_dir / f"{stem}.tif"
        rgb_path = self.rgb_dir / f"{stem}.png"
        reference_path = self.reference_dir / f"{stem}.tif"
        mask_path = self.mask_dir / f"{stem}.tif"
        projections = None if projection is None else [projection]
        geotransforms = None if geotransform is None else [list(geotransform)]
        save_tif_datas(
            prediction.unsqueeze(0),
            projections=projections,
            geotransforms=geotransforms,
            savepath=str(generated_path),
            denormalize=False,
        )
        save_rgb_datas(
            prepare_rgb_vis_tensor(
                prediction.unsqueeze(0),
                denormalize=False,
            ),
            nrow=1,
            savepath=str(rgb_path),
            format="PNG",
            is_makegrid=False,
        )
        save_tif_datas(
            reference.unsqueeze(0),
            projections=projections,
            geotransforms=geotransforms,
            savepath=str(reference_path),
            denormalize=False,
        )
        save_tif_datas(
            condition_mask,
            projections=projections,
            geotransforms=geotransforms,
            savepath=str(mask_path),
            denormalize=False,
            nodata_value=0.0,
        )

        record = GenerationRecord(
            record_id=rid,
            method=self.method,
            dataset=self.dataset,
            split=self.split,
            label=str(label),
            label_id=int(label_id),
            condition_id=str(condition_id),
            seed=int(seed),
            seed_rank=int(seed_rank),
            primary=bool(int(seed_rank) == 0),
            deterministic=bool(deterministic),
            protocol=self.protocol,
            checkpoint=self.checkpoint,
            generated_rgb=str(rgb_path),
            generated_tif=str(generated_path),
            reference_tif=str(reference_path),
            condition_mask_tif=str(mask_path),
            source_image=None if source_image is None else str(source_image),
            training_regime=self.training_regime,
            preload_sources=self.preload_sources,
            sampler=self.sampler,
            diffusion_steps=self.diffusion_steps,
            requested_steps_per_stage=self.requested_steps_per_stage,
            actual_fggen_steps=self.actual_fggen_steps,
            actual_imgsyn_steps=self.actual_imgsyn_steps,
            actual_total_steps=self.actual_total_steps,
            ddim_eta=self.ddim_eta,
            scheduler_type=self.scheduler_type,
        )
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record


def _batch_geotransforms(geotransforms, batch_size):
    if torch.is_tensor(geotransforms):
        values = geotransforms.detach().cpu()
        if values.shape != (batch_size, 6):
            raise ValueError(f"Expected geotransforms [B,6], got {values.shape}.")
        return values.tolist()
    if (
        isinstance(geotransforms, (list, tuple))
        and len(geotransforms) == 6
        and all(torch.is_tensor(value) for value in geotransforms)
    ):
        return torch.stack([value.detach().cpu() for value in geotransforms], dim=1).tolist()
    values = [list(value) for value in geotransforms]
    if len(values) != batch_size or any(len(value) != 6 for value in values):
        raise ValueError("Every sample must provide one six-value geotransform.")
    return values


def append_generation_batch(
    writer,
    *,
    predictions,
    references,
    masks,
    labels,
    label_ids,
    condition_ids,
    seeds,
    projections,
    geotransforms,
    seed_ranks=None,
    deterministic=False,
    source_images=None,
):
    batch_size = int(predictions.shape[0])
    if references.shape[0] != batch_size or masks.shape[0] != batch_size:
        raise ValueError("Predictions, references, and masks must share a batch size.")
    seed_ranks = [0] * batch_size if seed_ranks is None else list(seed_ranks)
    source_images = [None] * batch_size if source_images is None else list(source_images)
    geotransforms = _batch_geotransforms(geotransforms, batch_size)
    records = []
    for index in range(batch_size):
        records.append(writer.append_sample(
            prediction=predictions[index],
            reference=references[index],
            condition_mask=masks[index],
            label=labels[index],
            label_id=int(label_ids[index]),
            condition_id=condition_ids[index],
            seed=int(seeds[index]),
            seed_rank=int(seed_ranks[index]),
            deterministic=deterministic,
            projection=projections[index],
            geotransform=geotransforms[index],
            source_image=source_images[index],
        ))
    return records


def load_generation_manifest(manifest_path) -> list[GenerationRecord]:
    manifest_path = Path(manifest_path).resolve()
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                records.append(GenerationRecord(**payload))
            except TypeError as exc:
                raise ValueError(
                    f"Invalid generation manifest record at line {line_number}."
                ) from exc
    if not records:
        raise ValueError(f"Generation manifest is empty: {manifest_path}")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Generation manifest contains duplicate record_id values.")
    for record in records:
        allow_missing_reference = (
            record.method == "Full_IMPGM_Mask_Free" and record.split == "train"
        )
        if record.reference_tif is None and not allow_missing_reference:
            raise ValueError(
                "reference_tif may be null only for Full_IMPGM_Mask_Free train records."
            )
        required_fields = ("generated_rgb", "generated_tif", "condition_mask_tif")
        optional_fields = (
            "reference_tif",
            "generated_mask_tif",
            "generated_mask_png",
            "reference_mask_tif",
            "foreground_tif",
            "foreground_rgb",
        )
        for field_name in (*required_fields, *optional_fields):
            field_value = getattr(record, field_name)
            if field_value is None:
                if field_name in required_fields:
                    raise ValueError(
                        f"Manifest field {field_name!r} may not be null."
                    )
                continue
            path = Path(field_value)
            if not path.is_file():
                raise FileNotFoundError(f"Manifest path does not exist: {path}")
    return records


def _read_tif(path) -> torch.Tensor:
    array, _, _ = Tif_Read_and_Write().Tif_Read(str(path))
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    return torch.from_numpy(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0))


def _stable_subset(records: Iterable[GenerationRecord], count: int, salt: str):
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{salt}:{record.record_id}".encode("utf-8")
        ).digest(),
    )
    return ranked[: min(int(count), len(ranked))]


def _stack_images(records, field_name):
    return torch.stack([_read_tif(getattr(record, field_name)) for record in records])


def _feature_cache_key(record, field_name, band_mode="rgb"):
    path = Path(getattr(record, field_name))
    stat = path.stat()
    return (
        f"{path}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{band_mode}:{VIS_BAND_ORDER}:{INPUT_CHANNELS}"
    )


@torch.no_grad()
def _inception_features(
    records,
    field_name,
    cache_path,
    batch_size=16,
    band_mode="rgb",
):
    cache_path = Path(cache_path)
    cache = {"features": {}}
    if cache_path.is_file():
        loaded = torch.load(cache_path, map_location="cpu")
        if isinstance(loaded, dict) and isinstance(loaded.get("features"), dict):
            cache = loaded
    features = cache["features"]
    missing = [
        record for record in records
        if _feature_cache_key(record, field_name, band_mode) not in features
    ]
    if missing:
        feature_extractor = build_inception_feature_extractor(DEVICE)
        batch_starts = range(0, len(missing), max(1, int(batch_size)))
        for start in tqdm(
            batch_starts,
            total=math.ceil(len(missing) / max(1, int(batch_size))),
            desc=f"Inception {band_mode}/{field_name}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            batch_records = missing[start:start + max(1, int(batch_size))]
            images = _stack_images(batch_records, field_name).float().to(DEVICE)
            batch_features = extract_inception_features(
                images,
                band_mode=band_mode,
                vis_band_order=VIS_BAND_ORDER,
                feature_extractor=feature_extractor,
            ).cpu()
            for record, feature in zip(batch_records, batch_features):
                features[_feature_cache_key(record, field_name, band_mode)] = feature
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    return torch.stack([
        features[_feature_cache_key(record, field_name, band_mode)] for record in records
    ]).float()


def _distribution_group_metrics(records, cache_dir, seed, max_swd_samples):
    generated_rgb_features = _inception_features(
        records,
        "generated_tif",
        Path(cache_dir) / "generated_rgb_inception.pt",
    )
    reference_rgb_features = _inception_features(
        records,
        "reference_tif",
        Path(cache_dir) / "reference_rgb_inception.pt",
    )
    generated_nir_features = None
    reference_nir_features = None
    if INPUT_CHANNELS > 3:
        generated_nir_features = _inception_features(
            records,
            "generated_tif",
            Path(cache_dir) / "generated_nir_inception.pt",
            band_mode="nir",
        )
        reference_nir_features = _inception_features(
            records,
            "reference_tif",
            Path(cache_dir) / "reference_nir_inception.pt",
            band_mode="nir",
        )
    swd_records = _stable_subset(records, max_swd_samples, f"swd:{seed}")
    generated = _stack_images(swd_records, "generated_tif").float().clamp(0.0, 1.0)
    reference = _stack_images(swd_records, "reference_tif").float().clamp(0.0, 1.0)
    swd_value = compute_multiband_SWD(generated, reference, seed=seed)
    fid_rgb = compute_FID_from_features(
        generated_rgb_features, reference_rgb_features, device=DEVICE
    )
    fid_nir = None
    if generated_nir_features is not None:
        fid_nir = compute_FID_from_features(
            generated_nir_features, reference_nir_features, device=DEVICE
        )
    kid_rgb = compute_KID_from_features(
        generated_rgb_features, reference_rgb_features
    )
    kid_nir = None
    if generated_nir_features is not None:
        kid_nir = compute_KID_from_features(
            generated_nir_features, reference_nir_features
        )
    return {
        "n": len(records),
        "fid_rgb": fid_rgb,
        "fid_rgb_status": "ok" if fid_rgb is not None else "unavailable",
        "kid_rgb": kid_rgb,
        "kid_rgb_status": "ok" if kid_rgb is not None else "unavailable",
        "fid_nir": fid_nir,
        "fid_nir_status": "ok" if fid_nir is not None else "unavailable",
        "kid_nir": kid_nir,
        "kid_nir_status": "ok" if kid_nir is not None else "unavailable",
        "swd_all_bands": None if swd_value is None else float(swd_value.item()),
        "swd_status": "unavailable" if swd_value is None else "ok",
        "swd_n": len(swd_records),
    }


def compute_distribution_metrics(records, cache_dir, seed=0, max_swd_samples=128):
    """Compute global/per-class RGB and NIR feature metrics plus all-band SWD."""
    primary = [record for record in records if record.primary]
    reference_rgb_features = _inception_features(
        primary,
        "reference_tif",
        Path(cache_dir) / "real_baseline_reference_rgb_inception.pt",
    )
    reference_nir_features = None
    if INPUT_CHANNELS > 3:
        reference_nir_features = _inception_features(
            primary,
            "reference_tif",
            Path(cache_dir) / "real_baseline_reference_nir_inception.pt",
            band_mode="nir",
        )
    ranked_indices = sorted(
        range(len(primary)),
        key=lambda index: hashlib.sha256(
            f"real-baseline:{seed}:{primary[index].record_id}".encode("utf-8")
        ).digest(),
    )
    half = len(ranked_indices) // 2
    left_indices = ranked_indices[:half]
    right_indices = ranked_indices[half:half * 2]
    if half >= 2:
        left_records = [primary[index] for index in left_indices]
        right_records = [primary[index] for index in right_indices]
        swd_count = min(half, int(max_swd_samples))
        left_images = _stack_images(left_records[:swd_count], "reference_tif").float().clamp(0.0, 1.0)
        right_images = _stack_images(right_records[:swd_count], "reference_tif").float().clamp(0.0, 1.0)
        baseline_swd_value = compute_multiband_SWD(
            left_images, right_images, seed=seed
        )
        baseline_swd = (
            None
            if baseline_swd_value is None
            else float(baseline_swd_value.item())
        )
        baseline_fid_rgb = compute_FID_from_features(
            reference_rgb_features[left_indices],
            reference_rgb_features[right_indices],
            device=DEVICE,
        )
        baseline_fid_nir = None
        if reference_nir_features is not None:
            baseline_fid_nir = compute_FID_from_features(
                reference_nir_features[left_indices],
                reference_nir_features[right_indices],
                device=DEVICE,
            )
        baseline_kid_rgb = compute_KID_from_features(
            reference_rgb_features[left_indices], reference_rgb_features[right_indices]
        )
        baseline_kid_nir = None
        if reference_nir_features is not None:
            baseline_kid_nir = compute_KID_from_features(
                reference_nir_features[left_indices],
                reference_nir_features[right_indices],
            )
        real_baseline = {
            "n_per_half": half,
            "fid_rgb": baseline_fid_rgb,
            "fid_rgb_status": "ok" if baseline_fid_rgb is not None else "unavailable",
            "kid_rgb": baseline_kid_rgb,
            "kid_rgb_status": (
                "ok" if baseline_kid_rgb is not None else "unavailable"
            ),
            "fid_nir": baseline_fid_nir,
            "fid_nir_status": (
                "ok" if baseline_fid_nir is not None else "unavailable"
            ),
            "kid_nir": baseline_kid_nir,
            "kid_nir_status": (
                "ok" if baseline_kid_nir is not None else "unavailable"
            ),
            "swd_all_bands": baseline_swd,
            "swd_status": "unavailable" if baseline_swd is None else "ok",
            "swd_n_per_half": swd_count,
        }
    else:
        real_baseline = {"status": "insufficient_samples", "n_per_half": half}

    result = {
        "global": _distribution_group_metrics(
            primary,
            Path(cache_dir) / "global",
            seed,
            max_swd_samples,
        ),
        "per_class": {},
        "real_vs_real_finite_sample_baseline": real_baseline,
    }
    labels = sorted({record.label for record in primary})
    for label in labels:
        group = [record for record in primary if record.label == label]
        if len(group) < 2:
            result["per_class"][label] = {"n": len(group), "status": "insufficient_samples"}
            continue
        result["per_class"][label] = _distribution_group_metrics(
            group,
            Path(cache_dir) / f"class_{_safe_token(label)}",
            seed,
            max_swd_samples,
        )
    return result


def compute_spectral_distribution_metrics(records, bins=256):
    """Compare generated and real class-region spectra without pixel pairing."""
    primary = [record for record in records if record.primary]
    class_results = {}
    for label in sorted({record.label for record in primary}):
        group = [record for record in primary if record.label == label]
        generated_sum = torch.zeros(INPUT_CHANNELS)
        reference_sum = torch.zeros(INPUT_CHANNELS)
        generated_hist = torch.zeros(INPUT_CHANNELS, bins)
        reference_hist = torch.zeros(INPUT_CHANNELS, bins)
        support = 0
        for record in tqdm(
            group,
            desc=f"Spectral {label}",
            unit="image",
            leave=False,
            dynamic_ncols=True,
        ):
            generated = _read_tif(record.generated_tif).float().clamp(0.0, 1.0)
            reference = _read_tif(record.reference_tif).float().clamp(0.0, 1.0)
            mask = _read_tif(record.condition_mask_tif)[0] > 0.5
            if not torch.any(mask):
                mask = torch.ones_like(mask, dtype=torch.bool)
            pixels_generated = generated[:, mask]
            pixels_reference = reference[:, mask]
            support += int(mask.sum().item())
            generated_sum += pixels_generated.sum(dim=1)
            reference_sum += pixels_reference.sum(dim=1)
            for channel in range(INPUT_CHANNELS):
                generated_hist[channel] += torch.histc(
                    pixels_generated[channel], bins=bins, min=0.0, max=1.0
                )
                reference_hist[channel] += torch.histc(
                    pixels_reference[channel], bins=bins, min=0.0, max=1.0
                )
        if support == 0:
            continue
        generated_mean = generated_sum / support
        reference_mean = reference_sum / support
        w1 = [
            compute_histogram_W1(
                generated_hist[channel], reference_hist[channel], bins
            )
            for channel in range(INPUT_CHANNELS)
        ]
        sam_value = compute_mean_spectrum_SAM_degrees(
            generated_mean, reference_mean
        )
        class_results[label] = {
            "n": len(group),
            "pixel_support": support,
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
    supports = np.asarray(
        [result["pixel_support"] for result in class_results.values()], dtype=np.float64
    )
    weights = supports / supports.sum()
    valid_sam_results = [
        result for result in class_results.values()
        if result["mean_spectrum_sam_deg"] is not None
    ]
    sams = np.asarray(
        [result["mean_spectrum_sam_deg"] for result in valid_sam_results],
        dtype=np.float64,
    )
    w1s = np.asarray([result["mean_w1"] for result in class_results.values()])
    return {
        "per_class": class_results,
        "macro": {
            "mean_spectrum_sam_deg": None if sams.size == 0 else float(sams.mean()),
            "sam_valid_classes": int(sams.size),
            "mean_w1": float(w1s.mean()),
        },
        "support_weighted": {
            "mean_spectrum_sam_deg": (
                None
                if sams.size == 0
                else float(np.average(
                    sams,
                    weights=np.asarray(
                        [result["pixel_support"] for result in valid_sam_results],
                        dtype=np.float64,
                    ),
                ))
            ),
            "sam_valid_classes": int(sams.size),
            "mean_w1": float(np.sum(w1s * weights)),
        },
    }


def validate_generation_protocol(records, diversity_seed_count=5):
    if int(diversity_seed_count) < 2:
        raise ValueError("diversity_seed_count must be at least 2.")
    expected_diversity_ranks = list(range(int(diversity_seed_count)))
    groups = defaultdict(list)
    for record in records:
        groups[(record.label, record.condition_id)].append(record)

    diversity_groups = 0
    deterministic_groups = 0
    for (label, condition_id), group in groups.items():
        primary_records = [record for record in group if record.primary]
        if len(primary_records) != 1:
            raise ValueError(
                f"Condition {label}:{condition_id} must contain exactly one primary sample; "
                f"found {len(primary_records)}."
            )
        ranks = sorted(record.seed_rank for record in group)
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"Condition {label}:{condition_id} contains duplicate seed_rank values.")
        if len({record.seed for record in group}) != len(group):
            raise ValueError(f"Condition {label}:{condition_id} contains duplicate seeds.")
        if any(record.primary != (record.seed_rank == 0) for record in group):
            raise ValueError(
                f"Condition {label}:{condition_id} must mark seed_rank=0 as its only primary sample."
            )
        if len({record.label_id for record in group}) != 1:
            raise ValueError(f"Condition {label}:{condition_id} contains inconsistent label_id values.")
        source_images = {
            str(Path(record.source_image).resolve()).casefold()
            for record in group
            if record.source_image is not None
        }
        if len(source_images) > 1:
            raise ValueError(f"Condition {label}:{condition_id} contains inconsistent source images.")

        deterministic_flags = {record.deterministic for record in group}
        if len(deterministic_flags) != 1:
            raise ValueError(
                f"Condition {label}:{condition_id} mixes deterministic and stochastic records."
            )
        if True in deterministic_flags:
            deterministic_groups += 1
            if ranks != [0]:
                raise ValueError(
                    f"Deterministic condition {label}:{condition_id} must contain seed_rank [0]."
                )
        elif ranks == expected_diversity_ranks:
            diversity_groups += 1
        elif ranks != [0]:
            raise ValueError(
                f"Stochastic condition {label}:{condition_id} must contain seed ranks [0] "
                f"or {expected_diversity_ranks}; found {ranks}."
            )

    return {
        "status": "valid",
        "conditions": len(groups),
        "diversity_seed_count": int(diversity_seed_count),
        "diversity_conditions": diversity_groups,
        "deterministic_conditions": deterministic_groups,
    }


def compute_diversity_metrics(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record.label, record.condition_id)].append(record)
    per_condition = {}
    rgb_lpips_values = []
    nir_lpips_values = []
    pixel_values = []
    grouped_items = sorted(groups.items())
    for (label, condition_id), group in tqdm(
        grouped_items,
        desc="Diversity",
        unit="condition",
        leave=False,
        dynamic_ncols=True,
    ):
        group = sorted(group, key=lambda record: record.seed_rank)
        if len(group) < 2:
            continue
        samples = _stack_images(group, "generated_tif").to(DEVICE).float().clamp(0.0, 1.0)
        rgb_lpips_value, nir_lpips_value, pixel_value = compute_LPIPS_diversity(samples)
        key = f"{label}:{condition_id}"
        per_condition[key] = {
            "n": len(group),
            "rgb_lpips": (
                None if rgb_lpips_value is None else float(rgb_lpips_value.cpu().item())
            ),
            "nir_lpips": (
                None if nir_lpips_value is None else float(nir_lpips_value.cpu().item())
            ),
            "all_band_pixel_rmse": float(pixel_value.cpu().item()),
        }
        if rgb_lpips_value is not None:
            rgb_lpips_values.append(float(rgb_lpips_value.cpu().item()))
        if nir_lpips_value is not None:
            nir_lpips_values.append(float(nir_lpips_value.cpu().item()))
        pixel_values.append(float(pixel_value.cpu().item()))
    if not per_condition:
        return {"status": "not_available", "reason": "no repeated condition groups"}
    return {
        "conditions": len(per_condition),
        "mean_rgb_lpips": (
            None if not rgb_lpips_values else float(np.mean(rgb_lpips_values))
        ),
        "mean_nir_lpips": (
            None if not nir_lpips_values else float(np.mean(nir_lpips_values))
        ),
        "mean_all_band_pixel_rmse": float(np.mean(pixel_values)),
        "per_condition": per_condition,
    }


@torch.no_grad()
def compute_condition_segmentation_diagnostics(records, checkpoint_path, batch_size=8):
    """Evaluate condition alignment with a frozen complete-image evaluator."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Evaluation_UNet checkpoint does not exist: {checkpoint_path}"
        )
    model = EvaluationUNet(
        image_channels=INPUT_CHANNELS,
        num_classes=evaluation_unet_number_of_classes(),
        base_channels=EVALUATION_UNET_BASE_CHANNELS,
        condition_channels=EVALUATION_UNET_CONDITION_CHANNELS,
    ).to(DEVICE)
    model, checkpoint_metadata = load_model_for_eval(
        str(checkpoint_path), model, map_location=DEVICE
    )
    checkpoint_dataset = checkpoint_metadata.get("dataset_name")
    checkpoint_classes = checkpoint_metadata.get("num_classes")
    checkpoint_domain = checkpoint_metadata.get("input_domain")
    if checkpoint_dataset != DATASET_NAME:
        raise ValueError(
            f"Evaluation_UNet dataset mismatch: checkpoint={checkpoint_dataset!r}, "
            f"active={DATASET_NAME!r}."
        )
    if int(checkpoint_classes or -1) != len(PROMPT_DICT):
        raise ValueError(
            "Evaluation_UNet class-count mismatch: "
            f"checkpoint={checkpoint_classes}, active={len(PROMPT_DICT)}."
        )
    if checkpoint_domain != "normalized_complete_image":
        raise ValueError(
            "The supplied checkpoint was not trained in the normalized complete-image domain."
        )
    model.eval()
    mean = torch.tensor(IMAGE_MEAN, device=DEVICE).view(1, -1, 1, 1)
    std = torch.tensor(IMAGE_STD, device=DEVICE).view(1, -1, 1, 1)

    totals = {"generated_intersection": 0.0, "generated_union": 0.0,
              "generated_sum": 0.0, "real_intersection": 0.0,
              "real_union": 0.0, "real_sum": 0.0, "mask_sum": 0.0}
    primary = [record for record in records if record.primary]
    batch_starts = range(0, len(primary), max(1, int(batch_size)))
    for start in tqdm(
        batch_starts,
        total=math.ceil(len(primary) / max(1, int(batch_size))),
        desc="Condition consistency",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        group = primary[start:start + max(1, int(batch_size))]
        generated = _stack_images(group, "generated_tif").to(DEVICE).float().clamp(0.0, 1.0)
        reference = _stack_images(group, "reference_tif").to(DEVICE).float().clamp(0.0, 1.0)
        masks = _stack_images(group, "condition_mask_tif")[:, :1].to(DEVICE) > 0.5
        label_ids = torch.tensor(
            [record.label_id for record in group],
            device=DEVICE,
            dtype=torch.long,
        )
        generated_logits = model((generated - mean) / std, label_ids)
        real_logits = model((reference - mean) / std, label_ids)
        if generated_logits.shape != masks.shape or real_logits.shape != masks.shape:
            raise ValueError(
                "Evaluation_UNet diagnostics require one binary output channel with the same spatial "
                f"shape as the condition mask; generated={tuple(generated_logits.shape)}, "
                f"real={tuple(real_logits.shape)}, mask={tuple(masks.shape)}."
            )
        generated_pred = torch.sigmoid(generated_logits.float()) >= 0.5
        real_pred = torch.sigmoid(real_logits.float()) >= 0.5
        for prefix, prediction in (("generated", generated_pred), ("real", real_pred)):
            intersection = (prediction & masks).sum().item()
            totals[f"{prefix}_intersection"] += intersection
            totals[f"{prefix}_union"] += (prediction | masks).sum().item()
            totals[f"{prefix}_sum"] += prediction.sum().item()
        totals["mask_sum"] += masks.sum().item()

    def scores(prefix):
        intersection = totals[f"{prefix}_intersection"]
        union = totals[f"{prefix}_union"]
        denominator = totals[f"{prefix}_sum"] + totals["mask_sum"]
        return {
            "iou": float(intersection / max(union, 1.0)),
            "dice": float(2.0 * intersection / max(denominator, 1.0)),
        }

    return {
        "generated_consistency": scores("generated"),
        "real_image_evaluator_ceiling": scores("real"),
        "protocol": "frozen_class_conditioned_unet_trained_on_real_complete_images",
        "checkpoint_metadata": {
            "dataset_name": checkpoint_dataset,
            "num_classes": checkpoint_classes,
            "input_domain": checkpoint_domain,
            "conditioning": checkpoint_metadata.get("conditioning"),
        },
    }


def validate_replacement_manifest(records, heldout_source_paths):
    if any(record.split != "train" for record in records):
        raise ValueError("Downstream replacement manifests may contain train split records only.")
    if any(not record.primary or int(record.seed_rank) != 0 for record in records):
        raise ValueError(
            "Downstream replacement requires exactly one primary seed-rank-0 sample "
            "per selected train condition; regenerate with --no-diversity."
        )
    condition_ids = [record.condition_id for record in records]
    condition_counts = Counter(condition_ids)
    duplicate_condition_ids = sorted({
        condition_id
        for condition_id, count in condition_counts.items()
        if count > 1
    })
    if duplicate_condition_ids:
        raise ValueError(
            "Downstream replacement contains duplicate train conditions: "
            f"{duplicate_condition_ids[:10]}."
        )
    missing_sources = [record.condition_id for record in records if not record.source_image]
    if missing_sources:
        raise ValueError(
            "Downstream replacement records require source_image provenance for matched "
            f"real samples: {missing_sources[:10]}."
        )
    mismatched_sources = [
        record.condition_id
        for record in records
        if Path(str(record.source_image)).stem != str(record.condition_id)
    ]
    if mismatched_sources:
        raise ValueError(
            "Downstream replacement condition_id/source_image bindings are inconsistent: "
            f"{mismatched_sources[:10]}."
        )
    heldout = {str(Path(path).resolve()).casefold() for path in heldout_source_paths}
    leaked = [
        record.source_image for record in records
        if record.source_image is not None
        and str(Path(record.source_image).resolve()).casefold() in heldout
    ]
    if leaked:
        raise ValueError(f"Synthetic replacement manifest leaks {len(leaked)} held-out sources.")


def summarize_replacement_reports(real_only_report, synthetic_only_report):
    """Summarize one paired real-only versus synthetic-only replacement experiment."""
    real_payload = json.loads(Path(real_only_report).read_text(encoding="utf-8"))
    synthetic_payload = json.loads(Path(synthetic_only_report).read_text(encoding="utf-8"))
    if int(real_payload["seed"]) != int(synthetic_payload["seed"]):
        raise ValueError("Real-only and synthetic-only reports must use the same seed.")
    if real_payload.get("condition_signature") != synthetic_payload.get("condition_signature"):
        raise ValueError("Real-only and synthetic-only reports must use the same conditions.")
    if int(real_payload.get("num_train_samples", -1)) != int(
        synthetic_payload.get("num_train_samples", -2)
    ):
        raise ValueError("Real-only and synthetic-only reports must use equal sample counts.")

    metric_names = ("miou", "dice", "f1", "precision", "recall")
    real_metrics = {
        metric: float(real_payload["metrics"][metric]) for metric in metric_names
    }
    synthetic_metrics = {
        metric: float(synthetic_payload["metrics"][metric]) for metric in metric_names
    }
    return {
        "seed": int(real_payload["seed"]),
        "num_train_samples_per_protocol": int(real_payload["num_train_samples"]),
        "real_only": real_metrics,
        "synthetic_only": synthetic_metrics,
        "synthetic_minus_real": {
            metric: synthetic_metrics[metric] - real_metrics[metric]
            for metric in metric_names
        },
        "real_only_report": str(Path(real_only_report).resolve()),
        "synthetic_only_report": str(Path(synthetic_only_report).resolve()),
    }


def evaluate_generation_manifest(
    manifest_path,
    *,
    output_path=None,
    seed=0,
    max_swd_samples=128,
    evaluation_unet_checkpoint="auto",
    skip_lpips=False,
):
    manifest_path = Path(manifest_path).resolve()
    records = load_generation_manifest(manifest_path)
    methods = sorted({record.method for record in records})
    datasets = sorted({record.dataset for record in records})
    splits = sorted({record.split for record in records})
    protocols = sorted({record.protocol for record in records})
    if len(methods) != 1 or len(datasets) != 1 or len(splits) != 1:
        raise ValueError("One manifest must contain exactly one method, dataset, and split.")
    if not skip_lpips:
        require_lpips()
    protocol_validation = validate_generation_protocol(records)
    if evaluation_unet_checkpoint == "auto":
        evaluation_unet_checkpoint = EVALUATION_UNET_MODEL_PATH
    if evaluation_unet_checkpoint is not None:
        evaluation_unet_checkpoint = Path(evaluation_unet_checkpoint).resolve()
        if not evaluation_unet_checkpoint.is_file():
            raise FileNotFoundError(
                "Evaluation_UNet checkpoint does not exist: "
                f"{evaluation_unet_checkpoint}"
            )
    evaluation_progress = tqdm(
        total=4,
        desc="Evaluation: distribution metrics",
        unit="stage",
        dynamic_ncols=True,
    )
    distribution_metrics = compute_distribution_metrics(
        records,
        manifest_path.parent / "feature_cache",
        seed=seed,
        max_swd_samples=max_swd_samples,
    )
    evaluation_progress.update(1)
    evaluation_progress.set_description("Evaluation: spectral metrics")
    spectral_metrics = compute_spectral_distribution_metrics(records)
    evaluation_progress.update(1)
    evaluation_progress.set_description("Evaluation: diversity metrics")
    diversity_metrics = (
        {"status": "skipped", "reason": "disabled_by_user"}
        if skip_lpips
        else compute_diversity_metrics(records)
    )
    evaluation_progress.update(1)
    evaluation_progress.set_description("Evaluation: condition consistency")
    result = {
        "manifest_version": MANIFEST_VERSION,
        "method": methods[0],
        "dataset": datasets[0],
        "split": splits[0],
        "protocols": protocols,
        "training_regimes": sorted({record.training_regime for record in records}),
        "preload_sources": sorted({
            record.preload_sources
            for record in records
            if record.preload_sources is not None
        }),
        "checkpoints": sorted({record.checkpoint for record in records}),
        "num_primary_samples": sum(record.primary for record in records),
        "num_total_samples": len(records),
        "num_conditions": len({record.condition_id for record in records}),
        "per_class_primary_counts": {
            label: sum(record.primary and record.label == label for record in records)
            for label in sorted({record.label for record in records})
        },
        "seeds": sorted({record.seed for record in records}),
        "protocol_validation": protocol_validation,
        "distribution": distribution_metrics,
        "spectral_distribution": spectral_metrics,
        "diversity": diversity_metrics,
    }
    resolved_evaluation_unet_checkpoint = (
        None
        if evaluation_unet_checkpoint is None
        else str(evaluation_unet_checkpoint)
    )
    result["evaluation_dependencies"] = {
        "evaluation_unet_checkpoint": resolved_evaluation_unet_checkpoint,
        "lpips": "skipped" if skip_lpips else "required",
    }
    if evaluation_unet_checkpoint is not None:
        result["condition_segmentation_diagnostics"] = (
            compute_condition_segmentation_diagnostics(
                records,
                evaluation_unet_checkpoint,
            )
        )
        result["condition_segmentation_diagnostics"]["checkpoint"] = (
            resolved_evaluation_unet_checkpoint
        )
    else:
        result["condition_segmentation_diagnostics"] = {
            "status": "not_computed",
            "reason": "disabled_by_user",
            "checkpoint": None,
        }
    evaluation_progress.update(1)
    evaluation_progress.set_description("Evaluation complete")
    evaluation_progress.close()
    output_path = Path(output_path or manifest_path.with_name(METRICS_FILENAME))
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def run_generation_evaluation_cli(default_manifest=None, description=None):
    """Run manifest-only evaluation without loading a generation model."""
    parser = argparse.ArgumentParser(
        description=description or "Evaluate one IMPGM generation manifest."
    )
    parser.add_argument(
        "--manifest",
        default=None if default_manifest is None else str(default_manifest),
        required=default_manifest is None,
        help="Path to generation_manifest.jsonl produced by the inference step.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-swd-samples", type=int, default=128)
    parser.add_argument("--evaluation-unet-checkpoint", default="auto")
    parser.add_argument("--no-condition-segmentation", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    result = evaluate_generation_manifest(
        args.manifest,
        output_path=args.output,
        seed=args.seed,
        max_swd_samples=args.max_swd_samples,
        skip_lpips=args.skip_lpips,
        evaluation_unet_checkpoint=(
            None
            if args.no_condition_segmentation
            else args.evaluation_unet_checkpoint
        ),
    )
    output_path = Path(args.output or Path(args.manifest).with_name(METRICS_FILENAME))
    print(f"Generation metrics: {output_path.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate one IMPGM generation manifest.")
    parser.add_argument("manifest")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-swd-samples", type=int, default=128)
    parser.add_argument("--evaluation-unet-checkpoint", default="auto")
    parser.add_argument("--no-condition-segmentation", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    result = evaluate_generation_manifest(
        args.manifest,
        output_path=args.output,
        seed=args.seed,
        max_swd_samples=args.max_swd_samples,
        skip_lpips=args.skip_lpips,
        evaluation_unet_checkpoint=(
            None
            if args.no_condition_segmentation
            else args.evaluation_unet_checkpoint
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
