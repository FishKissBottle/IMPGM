from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Multi_Condition_Generation.Diffusion_Sampler_CFG import (
    DDIMSampler_CFG,
    DDPMSampler_CFG,
)
from FgGen.FgGen_Code.FgGen_ControlNet_Denoise import FgGen_ControlNet_Denoise
from FgGen.FgGen_Code.FgGen_Diffusion_Denoise import FgGen_Diffusion_Denoise
from FgSeg_UNet.FgSeg_Code.FgSeg_UNet_Infer import _build_model as _build_fgseg_model
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion_Denoise import ImgSyn_Diffusion_Denoise
from ImgSyn.ImgSyn_Code.IMPGM_Full_Pipeline_Inference import (
    _build_dataset,
    _build_models,
    _condition_label_from_path,
    _resolve_full_pipeline_scheduler_identity,
    _resolve_full_pipeline_training_provenance,
    _resolved_evaluation_entries,
    resolve_full_pipeline_sampling_config,
)
from ImgSyn.ImgSyn_Code.IMPGM_Mask_Free_Pipeline_Inference import (
    _build_models as _build_autonomous_models,
    _resolve_scheduler_identity as _resolve_autonomous_scheduler_identity,
    _resolve_training_provenance as _resolve_autonomous_training_provenance,
)
from IMPGM_Config import (
    DATASET_NAME,
    DATASET_YAML_PATH,
    DEVICE,
    FGGEN_BASE_DIFFUSION_MODEL_SAVEPATH,
    FGGEN_CONTROLNET_CONFIG,
    FGGEN_DIFFUSION_CONFIG,
    FGSEG_UNET_MODEL_SAVEPATH,
    IMG_SIZE,
    IMGSYN_DIFFUSION_CONFIG,
    LATENT_HIDDENCHANNEL,
    PROMPT_DICT,
    SIDELENGTH_SCALE_FACTOR,
    VAE_MODEL_SAVEPATH,
)
from IMPGM_Dataset import _resolve_mask_path
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import (
    GenerationManifestWriter,
)
from Evaluation.Evaluation_Code.IMPGM_Mask_Free_Evaluation import (
    MaskFreeGenerationManifestWriter,
)
from IMPGM_Scheduler import build_beta_schedule_from_config
from IMPGM_Utils import (
    build_torch_generator,
    build_train_autocast,
    decode_from_scaled_latent,
    denormalize_image_tensor,
    descale_latent,
    normalize_latent_scaling_factor,
    randn,
    save_tif_datas,
    scale_latent,
    set_random_seed,
)


CONDITION_FILE_VERSION = 1
MULTI_CONDITION_MANIFEST_VERSION = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_LAMBDA_CLOUD = 1.2
DEFAULT_LAMBDA_OBJECT = 0.8
SEED_MODULUS = 2**31 - 1


def _stable_digest(*values) -> bytes:
    payload = ":".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _stable_seed(base_seed, role, identity) -> int:
    offset = int.from_bytes(_stable_digest(role, identity)[:8], "big")
    return int((int(base_seed) + offset) % SEED_MODULUS)


def _rank_entries(entries, *, seed, label):
    return sorted(
        entries,
        key=lambda entry: _stable_digest(
            seed,
            label,
            Path(entry[1]).stem,
        ),
    )


def _entry_payload(dataset, entry):
    index, source_path = entry
    source_path = str(Path(source_path).resolve())
    mask_path = _resolve_mask_path(source_path, dataset._mask_name_to_path)
    return {
        "dataset_index": int(index),
        "condition_id": Path(source_path).stem,
        "label": _condition_label_from_path(source_path),
        "label_id": int(PROMPT_DICT[_condition_label_from_path(source_path)]),
        "source_image": source_path,
        "source_mask": str(Path(mask_path).resolve()),
    }


def _validate_labels(object_label, object_reference_label, cloud_labels, cloud_reference_label):
    requested = [object_label, object_reference_label, *cloud_labels, cloud_reference_label]
    missing = sorted({label for label in requested if label not in PROMPT_DICT})
    if missing:
        raise ValueError(
            f"Dataset {DATASET_NAME!r} does not define required labels: {missing}."
        )
    if not cloud_labels:
        raise ValueError("At least one cloud label is required.")
    if len(set(cloud_labels)) != len(cloud_labels):
        raise ValueError("Cloud labels must be unique.")


def prepare_conditions(
    *,
    split,
    object_label,
    object_reference_label,
    cloud_labels,
    cloud_reference_label,
    conditions_per_combination,
    seed,
    output,
    cloud_generation="controlled",
):
    """Create a deterministic object/cloud condition-pair file."""
    conditions_per_combination = int(conditions_per_combination)
    if conditions_per_combination <= 0:
        raise ValueError("conditions_per_combination must be positive.")
    if cloud_generation not in {"controlled", "autonomous"}:
        raise ValueError(f"Unsupported cloud generation path: {cloud_generation!r}.")
    _validate_labels(
        object_label,
        object_reference_label,
        cloud_labels,
        cloud_reference_label,
    )

    dataset = _build_dataset(split)
    grouped = {}
    for entry in _resolved_evaluation_entries(dataset):
        grouped.setdefault(_condition_label_from_path(entry[1]), []).append(entry)
    if not grouped.get(object_label):
        raise RuntimeError(f"No valid {object_label!r} conditions were found in split {split!r}.")
    missing_cloud_data = [label for label in cloud_labels if not grouped.get(label)]
    if cloud_generation == "controlled" and missing_cloud_data:
        raise RuntimeError(
            f"No valid conditions were found for cloud labels: {missing_cloud_data}."
        )

    ranked_objects = _rank_entries(grouped[object_label], seed=seed, label=object_label)
    ranked_clouds = {
        label: _rank_entries(grouped[label], seed=seed, label=label)
        for label in cloud_labels
    } if cloud_generation == "controlled" else {}
    pairs = []
    for combination_index, cloud_label in enumerate(cloud_labels):
        actual_count = min(conditions_per_combination, len(ranked_objects))
        if cloud_generation == "controlled":
            actual_count = min(actual_count, len(ranked_clouds[cloud_label]))
        if actual_count < conditions_per_combination:
            print(
                f"[Multi-condition] {cloud_label}: using {actual_count} paired "
                f"conditions instead of the requested {conditions_per_combination}."
            )
        for item_index in range(actual_count):
            # Reuse the same ranked Water conditions across cloud combinations.
            object_entry = ranked_objects[item_index]
            object_payload = _entry_payload(dataset, object_entry)
            if cloud_generation == "controlled":
                cloud_payload = _entry_payload(dataset, ranked_clouds[cloud_label][item_index])
            else:
                cloud_payload = {
                    "condition_id": f"autonomous_{cloud_label}_{item_index + 1:03d}",
                    "label": cloud_label,
                    "label_id": int(PROMPT_DICT[cloud_label]),
                    "source_image": None,
                    "source_mask": None,
                }
            pair_digest = hashlib.sha256(
                (
                    f"{object_payload['condition_id']}:{cloud_payload['condition_id']}:"
                    f"{cloud_label}"
                ).encode("utf-8")
            ).hexdigest()[:16]
            pairs.append(
                {
                    "pair_id": f"{cloud_label}_{item_index + 1:03d}_{pair_digest}",
                    "combination_index": int(combination_index),
                    "combination_sample_index": int(item_index),
                    "object_condition": object_payload,
                    "cloud_condition": cloud_payload,
                }
            )
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "condition_file_version": CONDITION_FILE_VERSION,
        "dataset": DATASET_NAME,
        "dataset_yaml": str(Path(DATASET_YAML_PATH).resolve()),
        "split": str(split),
        "selection_seed": int(seed),
        "object_label": str(object_label),
        "object_reference_label": str(object_reference_label),
        "cloud_labels": list(cloud_labels),
        "cloud_reference_label": str(cloud_reference_label),
        "cloud_generation": cloud_generation,
        "requested_conditions_per_combination": conditions_per_combination,
        "actual_conditions_per_combination": {
            label: sum(
                pair["cloud_condition"]["label"] == label for pair in pairs
            )
            for label in cloud_labels
        },
        "shared_object_conditions_across_combinations": True,
        "num_unique_object_conditions": len(
            {pair["object_condition"]["condition_id"] for pair in pairs}
        ),
        "num_pairs": len(pairs),
        "pairs": pairs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Condition pairs: {output_path}")
    print(f"Prepared {len(pairs)} object/cloud condition pairs.")
    return output_path


def _load_condition_file(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("condition_file_version", -1)) != CONDITION_FILE_VERSION:
        raise ValueError(f"Unsupported condition file version in {path}.")
    if payload.get("dataset") != DATASET_NAME:
        raise ValueError(
            f"Condition-file dataset mismatch: file={payload.get('dataset')!r}, "
            f"active={DATASET_NAME!r}."
        )
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Condition file contains no pairs: {path}")
    pair_ids = [pair.get("pair_id") for pair in pairs]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Condition file contains missing or duplicate pair_id values.")
    return path, payload


def _current_dataset_index(dataset):
    by_condition_id = {}
    for index, source_path in _resolved_evaluation_entries(dataset):
        condition_id = Path(source_path).stem
        if condition_id in by_condition_id:
            raise ValueError(f"Duplicate condition identifier in active dataset: {condition_id}")
        by_condition_id[condition_id] = (index, str(Path(source_path).resolve()))
    return by_condition_id


def _resolve_condition_entry(condition, current_index):
    condition_id = str(condition["condition_id"])
    if condition_id not in current_index:
        raise FileNotFoundError(
            f"Condition {condition_id!r} from the pair file is absent from the active dataset."
        )
    index, source_path = current_index[condition_id]
    label = _condition_label_from_path(source_path)
    if label != condition["label"]:
        raise ValueError(
            f"Condition label changed for {condition_id}: file={condition['label']!r}, "
            f"active={label!r}."
        )
    return index, source_path


def _randn_batch(shape, seeds):
    tensors = []
    for seed in seeds:
        generator = build_torch_generator(int(seed), DEVICE)
        tensors.append(randn((1, *shape[1:]), device=DEVICE, generator=generator))
    return torch.cat(tensors, dim=0)


def _build_cfg_sampler(
    sampler_mode,
    model,
    beta_t,
    generators,
    lambda_cloud,
    lambda_object,
):
    sampler_class = DDPMSampler_CFG if sampler_mode == "ddpm" else DDIMSampler_CFG
    return sampler_class(
        model,
        None,
        beta_t,
        is_Inference=True,
        generator=generators,
        lambda_cloud=lambda_cloud,
        lambda_object=lambda_object,
    ).to(DEVICE)


@torch.no_grad()
def _generate_multi_condition_batch(
    *,
    vae,
    img_model,
    sampler_mode,
    object_labels,
    object_reference_labels,
    cloud_labels,
    cloud_reference_labels,
    object_latents,
    cloud_latents,
    zero_latents,
    cloud_masks,
    initial_noise,
    imgsyn_seeds,
    lambda_cloud,
    lambda_object,
    ddim_steps,
    ddim_eta,
):
    latent_scaling_factor = normalize_latent_scaling_factor()
    foreground_latents = {
        "fg_imgs_NoObj_e": scale_latent(zero_latents, latent_scaling_factor),
        "fg_imgs_Obj_e": scale_latent(object_latents, latent_scaling_factor),
        "fg_imgs_NoCloud_e": scale_latent(zero_latents, latent_scaling_factor),
        "fg_imgs_Cloud_e": scale_latent(cloud_latents, latent_scaling_factor),
    }
    prompt_strings = {
        "prompt_str_NoObj": list(object_reference_labels),
        "prompt_str_Obj": list(object_labels),
        "prompt_str_NoCloud": list(cloud_reference_labels),
        "prompt_str_Cloud": list(cloud_labels),
    }
    latent_height, latent_width = initial_noise.shape[-2:]
    cloud_masks_latent = F.interpolate(
        cloud_masks.float(),
        size=(latent_height, latent_width),
        mode="bilinear",
        align_corners=False,
    )
    generators = [build_torch_generator(seed, DEVICE) for seed in imgsyn_seeds]
    sampler = _build_cfg_sampler(
        sampler_mode,
        img_model,
        build_beta_schedule_from_config(IMGSYN_DIFFUSION_CONFIG),
        generators,
        lambda_cloud,
        lambda_object,
    )
    sampler_kwargs = {
        "seed": int(imgsyn_seeds[0]),
        "generator": generators,
    }
    if sampler_mode == "ddim":
        sampler_kwargs.update({"steps": int(ddim_steps), "eta": float(ddim_eta)})
    generated_latent = sampler(
        initial_noise.clone(),
        prompt_strings,
        foreground_latents,
        cloud_masks_latent,
        is_record_process=False,
        **sampler_kwargs,
    )
    return decode_from_scaled_latent(
        vae,
        generated_latent,
        latent_scaling_factor,
    ).float()


def _save_cloud_evidence(writer, record, sample):
    mask_dir = writer.output_root / (
        "cloud_condition_masks" if sample["cloud_reference"] is not None else "cloud_generated_masks_tif"
    )
    reference_dir = writer.output_root / "cloud_reference_tif"
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record.generated_tif).stem
    mask_path = mask_dir / f"{stem}.tif"
    reference_path = reference_dir / f"{stem}.tif"
    generated_projections = [sample["object_projection"]]
    generated_geotransforms = [list(sample["object_geotransform"])]
    cloud_projections = [sample["cloud_projection"]]
    cloud_geotransforms = [list(sample["cloud_geotransform"])]
    save_tif_datas(
        sample["cloud_mask"],
        projections=generated_projections,
        geotransforms=generated_geotransforms,
        savepath=str(mask_path),
        denormalize=False,
        nodata_value=0.0,
    )
    if sample["cloud_reference"] is None:
        return str(mask_path.resolve()), None
    reference_dir.mkdir(parents=True, exist_ok=True)
    save_tif_datas(
        sample["cloud_reference"].unsqueeze(0),
        projections=cloud_projections,
        geotransforms=cloud_geotransforms,
        savepath=str(reference_path),
        denormalize=False,
    )
    return str(mask_path.resolve()), str(reference_path.resolve())


def _checkpoint_description(sampling_config, scheduler_description):
    eta_token = (
        "not_applicable"
        if sampling_config["ddim_eta"] is None
        else f"{sampling_config['ddim_eta']:.12g}"
    )
    return (
        f"fggen_diffusion={FGGEN_BASE_DIFFUSION_MODEL_SAVEPATH};"
        f"fggen_controlnet={FGGEN_CONTROLNET_CONFIG.MODEL_SAVEPATH};"
        f"imgsyn_diffusion={IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH};"
        f"vae={VAE_MODEL_SAVEPATH};scheduler={scheduler_description};"
        f"sampler={sampling_config['sampler']};"
        f"requested_steps_per_stage={sampling_config['requested_steps_per_stage']};"
        f"actual_fggen_steps_per_branch={sampling_config['actual_fggen_steps']};"
        f"actual_imgsyn_steps={sampling_config['actual_imgsyn_steps']};"
        f"ddim_eta={eta_token}"
    )


def _make_writer(
    output_root,
    *,
    variant,
    split,
    checkpoint,
    training_regime,
    preload_sources,
    scheduler_tag,
    sampling_config,
    condition_file,
    lambda_cloud,
    lambda_object,
    cloud_generation="controlled",
):
    method = (
        "Full_IMPGM_Object_Only"
        if variant == "object_only"
        else "Full_IMPGM_Multi_Condition"
    )
    if variant == "multi_condition" and cloud_generation == "autonomous":
        method = "Full_IMPGM_Multi_Condition_Autonomous_Cloud"
    fggen_branch_count = 1 if variant == "object_only" else 2
    actual_fggen_steps = (
        fggen_branch_count * sampling_config["actual_fggen_steps"]
    )
    actual_total_steps = actual_fggen_steps + sampling_config["actual_imgsyn_steps"]
    return GenerationManifestWriter(
        Path(output_root) / variant,
        method=method,
        dataset=DATASET_NAME,
        split=split,
        checkpoint=checkpoint,
        protocol=(
            f"paired_multi_condition_inference;variant={variant};"
            f"scheduler={scheduler_tag};condition_file={condition_file};"
            f"lambda_cloud={float(lambda_cloud):.12g};"
            f"lambda_object={float(lambda_object):.12g};"
            f"cloud_generation={cloud_generation}"
        ),
        training_regime=training_regime,
        preload_sources=preload_sources,
        sampler=sampling_config["sampler"],
        diffusion_steps=sampling_config["requested_steps_per_stage"],
        requested_steps_per_stage=sampling_config["requested_steps_per_stage"],
        actual_fggen_steps=actual_fggen_steps,
        actual_imgsyn_steps=sampling_config["actual_imgsyn_steps"],
        actual_total_steps=actual_total_steps,
        ddim_eta=sampling_config["ddim_eta"],
        scheduler_type=str(FGGEN_CONTROLNET_CONFIG.SCHEDULER_TYPE).lower(),
        overwrite=True,
    )


def _manifest_extras(
    *,
    variant,
    pair,
    sample,
    record,
    cloud_mask_path,
    cloud_reference_path,
    condition_file,
    object_reference_label,
    cloud_reference_label,
    lambda_cloud,
    lambda_object,
    object_fg_seed,
    cloud_fg_seed,
    imgsyn_seed,
):
    return {
        "multi_condition_manifest_version": MULTI_CONDITION_MANIFEST_VERSION,
        "variant": variant,
        "pair_id": pair["pair_id"],
        "cloud_label": sample["cloud_label"],
        "cloud_label_id": int(PROMPT_DICT[sample["cloud_label"]]),
        "cloud_reference_label": cloud_reference_label,
        "cloud_reference_label_id": int(PROMPT_DICT[cloud_reference_label]),
        "cloud_condition_id": sample["cloud_condition_id"],
        "cloud_source_image": sample["cloud_source_path"],
        "cloud_condition_mask_tif": cloud_mask_path,
        "cloud_reference_tif": cloud_reference_path,
        "object_label": sample["object_label"],
        "object_label_id": int(PROMPT_DICT[sample["object_label"]]),
        "object_reference_label": object_reference_label,
        "object_reference_label_id": int(PROMPT_DICT[object_reference_label]),
        "object_condition_id": sample["object_condition_id"],
        "object_source_image": sample["object_source_path"],
        "object_condition_mask_tif": record.condition_mask_tif,
        "object_reference_tif": record.reference_tif,
        "reference_role": "object_source_real_reference",
        "lambda_cloud": float(lambda_cloud),
        "lambda_object": float(lambda_object),
        "combination_coefficients_applied": bool(variant == "multi_condition"),
        "object_fg_seed": int(object_fg_seed),
        "cloud_fg_seed": (
            int(cloud_fg_seed) if variant == "multi_condition" else None
        ),
        "cloud_foreground_applied": bool(variant == "multi_condition"),
        "imgsyn_seed": int(imgsyn_seed),
        "shared_initial_noise_id": hashlib.sha256(
            f"imgsyn:{int(imgsyn_seed)}".encode("utf-8")
        ).hexdigest()[:16],
        "actual_network_forward_count": int(
            record.actual_fggen_steps
            + (1 if variant == "object_only" else 4) * record.actual_imgsyn_steps
        ),
        "condition_file": str(Path(condition_file).resolve()),
    }


@torch.no_grad()
def generate(
    *,
    condition_file,
    sampler,
    variants,
    lambda_cloud,
    lambda_object,
    batch_size,
    seed,
    output_root,
    ddim_steps=None,
    ddim_eta=None,
    cloud_generation=None,
):
    if not math.isfinite(lambda_cloud) or lambda_cloud < 0.0:
        raise ValueError("lambda_cloud must be finite and non-negative.")
    if not math.isfinite(lambda_object) or lambda_object < 0.0:
        raise ValueError("lambda_object must be finite and non-negative.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    variants = list(dict.fromkeys(variants))
    unsupported = sorted(set(variants) - {"object_only", "multi_condition"})
    if unsupported or not variants:
        raise ValueError(f"Unsupported or empty variants: {unsupported}")

    condition_path, condition_payload = _load_condition_file(condition_file)
    if cloud_generation is None:
        # Older condition files contain only mask-controlled cloud conditions.
        cloud_generation = condition_payload.get("cloud_generation", "controlled")
    if cloud_generation not in {"controlled", "autonomous"}:
        raise ValueError(f"Unsupported cloud generation path: {cloud_generation!r}.")
    if cloud_generation == "controlled" and condition_payload.get("cloud_generation") == "autonomous":
        raise ValueError("This condition file has no external cloud masks; use cloud_generation='autonomous'.")
    _validate_labels(
        condition_payload["object_label"], condition_payload["object_reference_label"],
        list(dict.fromkeys(pair["cloud_condition"]["label"] for pair in condition_payload["pairs"])),
        condition_payload["cloud_reference_label"],
    )
    split = str(condition_payload["split"])
    object_reference_label = str(condition_payload["object_reference_label"])
    cloud_reference_label = str(condition_payload["cloud_reference_label"])
    sampling_config = resolve_full_pipeline_sampling_config(
        sampler,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
    )
    set_random_seed(seed, deterministic=False)
    dataset = _build_dataset(split)
    current_index = _current_dataset_index(dataset)
    method_name, scheduler_tag, scheduler_description = (
        _resolve_full_pipeline_scheduler_identity()
    )
    del method_name
    training_regime, preload_sources = _resolve_full_pipeline_training_provenance()
    vae, fg_model, img_model = _build_models()
    fgseg_model = (
        _build_fgseg_model() if cloud_generation == "autonomous" and "multi_condition" in variants else None
    )
    checkpoint = _checkpoint_description(sampling_config, scheduler_description)
    if fgseg_model is not None:
        checkpoint += f";fgseg={FGSEG_UNET_MODEL_SAVEPATH};cloud_generation=autonomous"
    output_root = Path(output_root).resolve()
    writers = {
        variant: _make_writer(
            output_root,
            variant=variant,
            split=split,
            checkpoint=checkpoint,
            training_regime=training_regime,
            preload_sources=preload_sources,
            scheduler_tag=scheduler_tag,
            sampling_config=sampling_config,
            condition_file=condition_path,
            lambda_cloud=lambda_cloud,
            lambda_object=lambda_object,
            cloud_generation=cloud_generation,
        )
        for variant in variants
    }
    manifest_payloads = {variant: [] for variant in variants}
    pairs = condition_payload["pairs"]
    latent_size = IMG_SIZE // SIDELENGTH_SCALE_FACTOR

    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]
        samples = []
        for pair in batch_pairs:
            object_index, object_source_path = _resolve_condition_entry(
                pair["object_condition"], current_index
            )
            object_label, _, object_mask, object_reference_norm, object_proj, object_geo = (
                dataset[object_index]
            )
            cloud_label = pair["cloud_condition"]["label"]
            cloud_source_path = cloud_mask = cloud_reference = None
            cloud_proj, cloud_geo = object_proj, object_geo
            if cloud_generation == "controlled":
                cloud_index, cloud_source_path = _resolve_condition_entry(pair["cloud_condition"], current_index)
                cloud_label, _, cloud_mask, cloud_reference_norm, cloud_proj, cloud_geo = dataset[cloud_index]
                cloud_mask = cloud_mask.float()
                cloud_reference = denormalize_image_tensor(
                    cloud_reference_norm.unsqueeze(0)
                )[0].float().clamp(0.0, 1.0)
            if object_label != pair["object_condition"]["label"]:
                raise ValueError(f"Object label mismatch for pair {pair['pair_id']}.")
            if cloud_label != pair["cloud_condition"]["label"]:
                raise ValueError(f"Cloud label mismatch for pair {pair['pair_id']}.")
            samples.append(
                {
                    "pair": pair,
                    "object_label": object_label,
                    "cloud_label": cloud_label,
                    "object_mask": object_mask.float(),
                    "cloud_mask": cloud_mask,
                    "object_reference": denormalize_image_tensor(
                        object_reference_norm.unsqueeze(0)
                    )[0].float().clamp(0.0, 1.0),
                    "cloud_reference": cloud_reference,
                    "object_projection": object_proj,
                    "object_geotransform": object_geo,
                    "cloud_projection": cloud_proj,
                    "cloud_geotransform": cloud_geo,
                    "object_source_path": object_source_path,
                    "cloud_source_path": cloud_source_path,
                    "object_condition_id": pair["object_condition"]["condition_id"],
                    "cloud_condition_id": pair["cloud_condition"]["condition_id"],
                }
            )

        object_labels = [sample["object_label"] for sample in samples]
        cloud_labels = [sample["cloud_label"] for sample in samples]
        object_masks = torch.stack([sample["object_mask"] for sample in samples]).to(DEVICE)
        cloud_masks = (
            torch.stack([sample["cloud_mask"] for sample in samples]).to(DEVICE)
            if cloud_generation == "controlled" else None
        )
        object_fg_seeds = [
            _stable_seed(seed, "object_fg", sample["object_condition_id"])
            for sample in samples
        ]
        cloud_fg_seeds = [
            _stable_seed(seed, "cloud_fg", sample["cloud_condition_id"])
            for sample in samples
        ]
        imgsyn_seeds = [
            _stable_seed(seed, "imgsyn", sample["object_condition_id"])
            for sample in samples
        ]

        object_foregrounds = FgGen_ControlNet_Denoise(
            sampler_mode=sampler,
            prompt_str=object_labels,
            conditional_element=object_masks,
            need_to_decode=True,
            seed=object_fg_seeds[0],
            controlnet_model=fg_model,
            vae_model=vae,
            generators=[build_torch_generator(value, DEVICE) for value in object_fg_seeds],
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        cloud_foregrounds = None
        cloud_latents = None
        if "multi_condition" in variants and cloud_generation == "autonomous":
            cloud_foregrounds, cloud_latents, cloud_masks = _generate_autonomous_foregrounds(
                vae=vae, fg_model=fg_model.FgGen_Diffusion_model, fgseg_model=fgseg_model,
                labels=cloud_labels, seeds=cloud_fg_seeds, sampler_mode=sampler,
                latent_scaling_factor=normalize_latent_scaling_factor(),
                ddim_steps=ddim_steps, ddim_eta=ddim_eta,
            )
            for index, sample in enumerate(samples):
                sample["cloud_mask"] = cloud_masks[index]
        elif "multi_condition" in variants:
            cloud_foregrounds = FgGen_ControlNet_Denoise(
                sampler_mode=sampler,
                prompt_str=cloud_labels,
                conditional_element=cloud_masks,
                need_to_decode=True,
                seed=cloud_fg_seeds[0],
                controlnet_model=fg_model,
                vae_model=vae,
                generators=[build_torch_generator(value, DEVICE) for value in cloud_fg_seeds],
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
            )

        with build_train_autocast():
            object_latents, _, _ = vae.reparameterize(vae.encode(object_foregrounds))
            if cloud_foregrounds is not None and cloud_generation == "controlled":
                cloud_latents, _, _ = vae.reparameterize(vae.encode(cloud_foregrounds))
            zero_foregrounds = torch.zeros_like(object_foregrounds)
            zero_latents, _, _ = vae.reparameterize(vae.encode(zero_foregrounds))

        initial_noise = _randn_batch(
            (
                len(samples),
                LATENT_HIDDENCHANNEL,
                latent_size,
                latent_size,
            ),
            imgsyn_seeds,
        )
        predictions_norm = {}
        if "object_only" in variants:
            predictions_norm["object_only"] = ImgSyn_Diffusion_Denoise(
                sampler_mode=sampler,
                prompt_str=object_labels,
                fg_imgs_e=object_latents,
                need_to_decode=True,
                seed=imgsyn_seeds[0],
                diffusion_model=img_model,
                vae_model=vae,
                initial_noise=initial_noise.clone(),
                generators=[build_torch_generator(value, DEVICE) for value in imgsyn_seeds],
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
            ).float()
        if "multi_condition" in variants:
            predictions_norm["multi_condition"] = _generate_multi_condition_batch(
                vae=vae,
                img_model=img_model,
                sampler_mode=sampler,
                object_labels=object_labels,
                object_reference_labels=[object_reference_label] * len(samples),
                cloud_labels=cloud_labels,
                cloud_reference_labels=[cloud_reference_label] * len(samples),
                object_latents=object_latents,
                cloud_latents=cloud_latents,
                zero_latents=zero_latents,
                cloud_masks=cloud_masks,
                initial_noise=initial_noise,
                imgsyn_seeds=imgsyn_seeds,
                lambda_cloud=lambda_cloud,
                lambda_object=lambda_object,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
            )

        for variant, prediction_norm in predictions_norm.items():
            predictions = denormalize_image_tensor(prediction_norm).float().clamp(0.0, 1.0)
            writer = writers[variant]
            for index, (prediction, sample) in enumerate(zip(predictions, samples)):
                record = writer.append_sample(
                    prediction=prediction,
                    reference=sample["object_reference"],
                    condition_mask=sample["object_mask"],
                    label=sample["object_label"],
                    label_id=PROMPT_DICT[sample["object_label"]],
                    condition_id=sample["pair"]["pair_id"],
                    seed=imgsyn_seeds[index],
                    seed_rank=0,
                    deterministic=False,
                    projection=sample["object_projection"],
                    geotransform=sample["object_geotransform"],
                    source_image=sample["object_source_path"],
                )
                cloud_mask_path = cloud_reference_path = None
                if sample["cloud_mask"] is not None:
                    cloud_mask_path, cloud_reference_path = _save_cloud_evidence(writer, record, sample)
                payload = asdict(record)
                payload.update(
                    _manifest_extras(
                        variant=variant,
                        pair=sample["pair"],
                        sample=sample,
                        record=record,
                        cloud_mask_path=cloud_mask_path,
                        cloud_reference_path=cloud_reference_path,
                        condition_file=condition_path,
                        object_reference_label=object_reference_label,
                        cloud_reference_label=cloud_reference_label,
                        lambda_cloud=lambda_cloud,
                        lambda_object=lambda_object,
                        object_fg_seed=object_fg_seeds[index],
                        cloud_fg_seed=cloud_fg_seeds[index],
                        imgsyn_seed=imgsyn_seeds[index],
                    )
                )
                payload.update({
                    "object_generation": "controlled", "cloud_generation": cloud_generation,
                    "object_external_mask_used": True,
                    "cloud_external_mask_used": cloud_generation == "controlled",
                    "cloud_mask_source": "external" if cloud_generation == "controlled" else "fgseg_generated",
                    "fgseg_checkpoint": str(FGSEG_UNET_MODEL_SAVEPATH) if fgseg_model is not None else None,
                })
                manifest_payloads[variant].append(payload)

        completed = min(batch_start + len(batch_pairs), len(pairs))
        print(
            f"[Multi-condition] {completed}/{len(pairs)} paired conditions generated "
            f"with batch_size={len(batch_pairs)}."
        )

    for variant, writer in writers.items():
        manifest_text = "".join(
            json.dumps(payload, ensure_ascii=False) + "\n"
            for payload in manifest_payloads[variant]
        )
        writer.manifest_path.write_text(manifest_text, encoding="utf-8")
        print(f"{variant} manifest: {writer.manifest_path.resolve()}")
    return {variant: writer.manifest_path for variant, writer in writers.items()}


@torch.no_grad()
def _generate_autonomous_foregrounds(
    *, vae, fg_model, fgseg_model, labels, seeds, sampler_mode,
    latent_scaling_factor, ddim_steps, ddim_eta,
):
    _, scaled_latents = FgGen_Diffusion_Denoise(
        sampler_mode=sampler_mode,
        prompt_str=list(labels),
        need_to_decode=False,
        seed=int(seeds[0]),
        diffusion_model=fg_model,
        vae_model=vae,
        generators=[build_torch_generator(seed, DEVICE) for seed in seeds],
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
    )
    with build_train_autocast():
        foregrounds = decode_from_scaled_latent(
            vae, scaled_latents, latent_scaling_factor
        )
        mask_logits = fgseg_model(foregrounds)
    masks = (torch.sigmoid(mask_logits.float()) >= 0.5).float()
    return (
        foregrounds.float(),
        descale_latent(scaled_latents, latent_scaling_factor),
        masks,
    )


@torch.no_grad()
def generate_autonomous(
    *, object_label, object_reference_label, cloud_labels, cloud_reference_label,
    samples_per_combination, sampler, lambda_cloud, lambda_object, batch_size,
    seed, output_root, ddim_steps=None, ddim_eta=None,
):
    """Compose independently sampled foregrounds without external image/mask inputs."""
    _validate_labels(object_label, object_reference_label, cloud_labels, cloud_reference_label)
    samples_per_combination = int(samples_per_combination)
    batch_size = int(batch_size)
    if samples_per_combination <= 0 or batch_size <= 0:
        raise ValueError("samples_per_combination and batch_size must be positive.")
    for name, value in (("lambda_cloud", lambda_cloud), ("lambda_object", lambda_object)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if ddim_eta is not None and not math.isfinite(float(ddim_eta)):
        raise ValueError("ddim_eta must be finite.")
    sampling_config = resolve_full_pipeline_sampling_config(
        sampler, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
        fggen_config=FGGEN_DIFFUSION_CONFIG,
    )
    sampler = sampling_config["sampler"]
    scheduler_tag, scheduler_description = _resolve_autonomous_scheduler_identity()
    training_regime, preload_sources = _resolve_autonomous_training_provenance()
    set_random_seed(seed, deterministic=False)
    vae, fg_model, img_model, fgseg_model, latent_scaling_factor = _build_autonomous_models()
    writer = MaskFreeGenerationManifestWriter(
        Path(output_root) / "multi_condition",
        dataset=DATASET_NAME,
        split="autonomous",
        checkpoint=f"autonomous_multi_condition;scheduler={scheduler_description}",
        protocol=f"autonomous_multi_condition_inference;scheduler={scheduler_tag}",
        training_regime=training_regime,
        preload_sources=preload_sources,
        vae_checkpoint=VAE_MODEL_SAVEPATH,
        fggen_checkpoint=FGGEN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
        imgsyn_checkpoint=IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
        fgseg_checkpoint=FGSEG_UNET_MODEL_SAVEPATH,
        latent_scaling_factor=latent_scaling_factor,
        sampler=sampler,
        diffusion_steps=sampling_config["requested_steps_per_stage"],
    )
    cloud_mask_dir = writer.output_root / "cloud_generated_masks_tif"
    cloud_foreground_dir = writer.output_root / "cloud_foreground_tif"
    cloud_mask_dir.mkdir(parents=True, exist_ok=True)
    cloud_foreground_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (cloud_label, index)
        for cloud_label in cloud_labels
        for index in range(samples_per_combination)
    ]
    manifest_payloads = []
    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]
        count = len(batch_pairs)
        object_labels = [object_label] * count
        batch_cloud_labels = [label for label, _ in batch_pairs]
        # Keep the object and initial scene noise matched across cloud labels.
        object_seeds = [_stable_seed(seed, "object_fg", f"{object_label}:{index}")
                        for _, index in batch_pairs]
        cloud_seeds = [_stable_seed(seed, "cloud_fg", f"{label}:{index}")
                       for label, index in batch_pairs]
        imgsyn_seeds = [_stable_seed(seed, "imgsyn", f"{object_label}:{index}")
                       for _, index in batch_pairs]
        foreground_kwargs = dict(
            vae=vae, fg_model=fg_model, fgseg_model=fgseg_model,
            sampler_mode=sampler, latent_scaling_factor=latent_scaling_factor,
            ddim_steps=ddim_steps, ddim_eta=ddim_eta,
        )
        object_foregrounds, object_latents, object_masks = _generate_autonomous_foregrounds(
            labels=object_labels, seeds=object_seeds, **foreground_kwargs
        )
        cloud_foregrounds, cloud_latents, cloud_masks = _generate_autonomous_foregrounds(
            labels=batch_cloud_labels, seeds=cloud_seeds, **foreground_kwargs
        )
        with build_train_autocast():
            zero_latents, _, _ = vae.reparameterize(vae.encode(torch.zeros_like(object_foregrounds)))
        initial_noise = _randn_batch(object_latents.shape, imgsyn_seeds)
        predictions = _generate_multi_condition_batch(
            vae=vae, img_model=img_model, sampler_mode=sampler,
            object_labels=object_labels, object_reference_labels=[object_reference_label] * count,
            cloud_labels=batch_cloud_labels, cloud_reference_labels=[cloud_reference_label] * count,
            object_latents=object_latents, cloud_latents=cloud_latents, zero_latents=zero_latents,
            cloud_masks=cloud_masks, initial_noise=initial_noise, imgsyn_seeds=imgsyn_seeds,
            lambda_cloud=lambda_cloud, lambda_object=lambda_object,
            ddim_steps=ddim_steps, ddim_eta=ddim_eta,
        )
        predictions = denormalize_image_tensor(predictions).float().clamp(0.0, 1.0)
        object_foregrounds = denormalize_image_tensor(object_foregrounds).float().clamp(0.0, 1.0)
        cloud_foregrounds = denormalize_image_tensor(cloud_foregrounds).float().clamp(0.0, 1.0)
        for index, (cloud_label, sample_index) in enumerate(batch_pairs):
            condition_id = "autonomous_" + _stable_digest(object_label, cloud_label, sample_index).hex()[:16]
            record = writer.append_sample(
                prediction=predictions[index], foreground=object_foregrounds[index],
                generated_mask=object_masks[index], label=object_label,
                label_id=int(PROMPT_DICT[object_label]), condition_id=condition_id,
                fggen_seed=object_seeds[index], imgsyn_seed=imgsyn_seeds[index], seed_rank=0,
            )
            stem = Path(record.generated_tif).stem
            cloud_mask_path = cloud_mask_dir / f"{stem}.tif"
            cloud_foreground_path = cloud_foreground_dir / f"{stem}.tif"
            save_tif_datas(cloud_masks[index].unsqueeze(0), savepath=str(cloud_mask_path),
                           denormalize=False, nodata_value=0.0)
            save_tif_datas(cloud_foregrounds[index].unsqueeze(0), savepath=str(cloud_foreground_path),
                           denormalize=False)
            fg_steps = 2 * sampling_config["actual_fggen_steps"]
            img_steps = sampling_config["actual_imgsyn_steps"]
            record = replace(
                record, method="IMPGM_Multi_Condition_Autonomous",
                requested_steps_per_stage=sampling_config["requested_steps_per_stage"],
                actual_fggen_steps=fg_steps, actual_imgsyn_steps=img_steps,
                actual_total_steps=fg_steps + img_steps, ddim_eta=sampling_config["ddim_eta"],
                scheduler_type=str(FGGEN_DIFFUSION_CONFIG.SCHEDULER_TYPE).lower(),
            )
            payload = asdict(record)
            payload.update({
                "multi_condition_manifest_version": MULTI_CONDITION_MANIFEST_VERSION,
                "variant": "multi_condition", "generation_path": "autonomous",
                "dataset_yaml": str(DATASET_YAML_PATH), "sample_index": sample_index,
                "object_label": object_label, "object_reference_label": object_reference_label,
                "cloud_label": cloud_label, "cloud_label_id": int(PROMPT_DICT[cloud_label]),
                "cloud_reference_label": cloud_reference_label, "reference_role": "none",
                "object_fg_seed": object_seeds[index], "cloud_fg_seed": cloud_seeds[index],
                "cloud_generated_mask_tif": str(cloud_mask_path),
                "cloud_foreground_tif": str(cloud_foreground_path),
                "lambda_cloud": float(lambda_cloud), "lambda_object": float(lambda_object),
                "actual_network_forward_count": fg_steps + 4 * img_steps,
            })
            manifest_payloads.append(payload)
        print(f"[Autonomous composition] {batch_start + count}/{len(pairs)} samples generated.")
    writer.manifest_path.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in manifest_payloads),
        encoding="utf-8",
    )
    print(f"Autonomous multi-condition manifest: {writer.manifest_path}")
    return writer.manifest_path


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Generate mask-conditioned or autonomous IMPGM multi-condition samples."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-conditions",
        help="Create a deterministic object/cloud condition-pair file.",
    )
    prepare_parser.add_argument(
        "--split", choices=("train", "valid", "test", "draw"), default="test"
    )
    prepare_parser.add_argument("--object-label", default="Water")
    prepare_parser.add_argument("--object-reference-label", default="NoObj")
    prepare_parser.add_argument(
        "--cloud-labels",
        nargs="+",
        default=("FewCloud", "LessCloud", "MoreCloud", "ManyCloud"),
    )
    prepare_parser.add_argument("--cloud-reference-label", default="NoCloud")
    prepare_parser.add_argument("--cloud-generation", choices=("controlled", "autonomous"), default="controlled")
    prepare_parser.add_argument("--conditions-per-combination", type=int, default=20)
    prepare_parser.add_argument("--seed", type=int, default=999)
    prepare_parser.add_argument("--output", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate paired object-only and multi-condition samples.",
    )
    generate_parser.add_argument("--condition-file", required=True)
    generate_parser.add_argument(
        "--cloud-generation", choices=("controlled", "autonomous"), default=None,
        help="Cloud foreground path; defaults to the path recorded in the condition file.",
    )
    generate_parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    generate_parser.add_argument(
        "--variants",
        nargs="+",
        choices=("object_only", "multi_condition"),
        default=("object_only", "multi_condition"),
    )
    generate_parser.add_argument("--lambda-cloud", type=float, default=DEFAULT_LAMBDA_CLOUD)
    generate_parser.add_argument("--lambda-object", type=float, default=DEFAULT_LAMBDA_OBJECT)
    generate_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    generate_parser.add_argument("--seed", type=int, default=999)
    generate_parser.add_argument("--output-root", required=True)
    generate_parser.add_argument("--ddim-steps", type=int, default=None)
    generate_parser.add_argument("--ddim-eta", type=float, default=None)

    autonomous_parser = subparsers.add_parser(
        "generate-autonomous", help="Compose autonomous object/cloud foregrounds without input masks.",
    )
    autonomous_parser.add_argument("--object-label", default="Water")
    autonomous_parser.add_argument("--object-reference-label", default="NoObj")
    autonomous_parser.add_argument(
        "--cloud-labels", nargs="+", default=("FewCloud", "LessCloud", "MoreCloud", "ManyCloud"),
    )
    autonomous_parser.add_argument("--cloud-reference-label", default="NoCloud")
    autonomous_parser.add_argument("--samples-per-combination", type=int, default=20)
    autonomous_parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    autonomous_parser.add_argument("--lambda-cloud", type=float, default=DEFAULT_LAMBDA_CLOUD)
    autonomous_parser.add_argument("--lambda-object", type=float, default=DEFAULT_LAMBDA_OBJECT)
    autonomous_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    autonomous_parser.add_argument("--seed", type=int, default=999)
    autonomous_parser.add_argument("--output-root", required=True)
    autonomous_parser.add_argument("--ddim-steps", type=int, default=None)
    autonomous_parser.add_argument("--ddim-eta", type=float, default=None)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "prepare-conditions":
        prepare_conditions(
            split=args.split,
            object_label=args.object_label,
            object_reference_label=args.object_reference_label,
            cloud_labels=args.cloud_labels,
            cloud_reference_label=args.cloud_reference_label,
            conditions_per_combination=args.conditions_per_combination,
            seed=args.seed,
            output=args.output,
            cloud_generation=args.cloud_generation,
        )
        return

    if args.sampler == "ddim" and (args.ddim_steps is None or args.ddim_eta is None):
        parser.error("DDIM generation requires both --ddim-steps and --ddim-eta.")
    if args.sampler != "ddim" and (args.ddim_steps is not None or args.ddim_eta is not None):
        parser.error("--ddim-steps and --ddim-eta may only be used with --sampler ddim.")
    if args.command == "generate-autonomous":
        generate_autonomous(
            object_label=args.object_label, object_reference_label=args.object_reference_label,
            cloud_labels=args.cloud_labels, cloud_reference_label=args.cloud_reference_label,
            samples_per_combination=args.samples_per_combination, sampler=args.sampler,
            lambda_cloud=args.lambda_cloud, lambda_object=args.lambda_object,
            batch_size=args.batch_size, seed=args.seed, output_root=args.output_root,
            ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
        )
        return
    generate(
        condition_file=args.condition_file,
        sampler=args.sampler,
        variants=args.variants,
        lambda_cloud=args.lambda_cloud,
        lambda_object=args.lambda_object,
        batch_size=args.batch_size,
        seed=args.seed,
        output_root=args.output_root,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        cloud_generation=args.cloud_generation,
    )


if __name__ == "__main__":
    main()
