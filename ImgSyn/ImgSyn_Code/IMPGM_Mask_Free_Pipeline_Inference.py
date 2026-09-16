import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FgGen.FgGen_Code.FgGen_Diffusion import FgGen_Diffusion_UNet
from FgGen.FgGen_Code.FgGen_Diffusion_Denoise import FgGen_Diffusion_Denoise
from FgSeg_UNet.FgSeg_Code.FgSeg_UNet_model import FgSeg_UNet
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion import ImgSyn_Diffusion_UNet
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion_Denoise import ImgSyn_Diffusion_Denoise
from IMPGM_Config import (
    DATASET_DICT,
    DATASET_NAME,
    DEVICE,
    DRAW_SAMPLER_MODE,
    FGGEN_DIFFUSION_CONFIG,
    FGSEG_UNET_MODEL_SAVEPATH,
    IMGSYN_DIFFUSION_CONFIG,
    INPUT_CHANNELS,
    PROMPT_DICT,
    UNET_OUTPUT_CHANNELS,
    VAE_MODEL_SAVEPATH,
    VAE_PRELOAD_SOURCE_DATASET,
    VAE_TRAINING_REGIME,
)
from IMPGM_Dataset import IMPGM_Dataset, build_unique_resolved_entries
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import (
    GENERATION_BASE_SEED,
    build_generation_seed,
    resolve_condition_selection,
)
from Evaluation.Evaluation_Code.IMPGM_Mask_Free_Evaluation import (
    MaskFreeGenerationManifestWriter,
)
from IMPGM_Scheduler import build_scheduler_tag
from IMPGM_Utils import (
    build_torch_generator,
    build_train_autocast,
    decode_from_scaled_latent,
    denormalize_image_tensor,
    descale_latent,
    load_model_for_eval,
    load_standard_vae,
    normalize_latent_scaling_factor,
    set_random_seed,
)


DEFAULT_INFERENCE_BATCH_SIZE = 32
DEFAULT_DIVERSITY_CONDITIONS_PER_CLASS = 20
DEFAULT_DIVERSITY_SEEDS = 5


def _split_keys(split_name):
    suffix = {"train": "Train", "valid": "Valid", "test": "Test", "draw": "Draw"}[split_name]
    return f"img_rootdir_list_for{suffix}", f"msk_rootdir_list_for{suffix}"


def _build_dataset(split_name):
    image_key, mask_key = _split_keys(split_name)
    if image_key not in DATASET_DICT or mask_key not in DATASET_DICT:
        raise KeyError(f"Dataset {DATASET_NAME!r} does not define split {split_name!r}.")
    return IMPGM_Dataset(
        img_rootdir_list=list(DATASET_DICT[image_key]),
        msk_rootdir_list=list(DATASET_DICT[mask_key]),
        is_train=False,
    )


def _label_from_path(path):
    tokens = Path(path).stem.split("_")
    if len(tokens) < 2:
        raise ValueError(f"Cannot derive a class label from {path}.")
    label = tokens[-2]
    if label not in PROMPT_DICT:
        raise KeyError(f"Unknown class label {label!r} in {path}.")
    return label


def _resolve_training_provenance():
    components = {
        "vae": (VAE_TRAINING_REGIME, VAE_PRELOAD_SOURCE_DATASET),
        "fggen_diffusion": (
            FGGEN_DIFFUSION_CONFIG.TRAINING_REGIME,
            FGGEN_DIFFUSION_CONFIG.PRELOAD_SOURCE_DATASET,
        ),
        "imgsyn_diffusion": (
            IMGSYN_DIFFUSION_CONFIG.TRAINING_REGIME,
            IMGSYN_DIFFUSION_CONFIG.PRELOAD_SOURCE_DATASET,
        ),
    }
    regimes = {regime for regime, _ in components.values()}
    training_regime = regimes.pop() if len(regimes) == 1 else "mixed"
    preload_sources = ";".join(
        f"{name}={source}"
        for name, (regime, source) in components.items()
        if regime != "from_scratch"
    )
    return training_regime, preload_sources or None


def _resolve_scheduler_identity():
    fg = FGGEN_DIFFUSION_CONFIG
    img = IMGSYN_DIFFUSION_CONFIG
    scheduler_type = str(fg.SCHEDULER_TYPE).strip().lower()
    attributes = [
        "SCHEDULER_TYPE",
        "STEPS",
        "SCHEDULER_MIN_BETA",
        "SCHEDULER_MAX_BETA",
    ]
    if scheduler_type == "cosine":
        attributes.append("SCHEDULER_COSINE_S")
    elif scheduler_type == "power":
        attributes.append("SCHEDULER_POWER_VAL")
    elif scheduler_type == "sigmoid":
        attributes.extend(("SCHEDULER_SIGMOID_START", "SCHEDULER_SIGMOID_END"))
    mismatches = [name for name in attributes if getattr(fg, name) != getattr(img, name)]
    if mismatches:
        raise ValueError(
            "Mask-free FgGen/ImgSyn scheduler settings differ: " + ", ".join(mismatches)
        )
    scheduler_tag = build_scheduler_tag(scheduler_type, int(fg.SCHEDULER_POWER_VAL))
    description = (
        f"type={scheduler_type},steps={int(fg.STEPS)},"
        f"min_beta={float(fg.SCHEDULER_MIN_BETA):.12g},"
        f"max_beta={float(fg.SCHEDULER_MAX_BETA):.12g}"
    )
    return scheduler_tag, description


def _build_models():
    checkpoint_paths = (
        VAE_MODEL_SAVEPATH,
        FGGEN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
        IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
        FGSEG_UNET_MODEL_SAVEPATH,
    )
    for checkpoint_path in checkpoint_paths:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Mask-free inference checkpoint does not exist: {checkpoint_path}"
            )

    vae, latent_scaling_factor, _ = load_standard_vae(
        VAE_MODEL_SAVEPATH, device=DEVICE, load_ema=True
    )
    fg_model = FgGen_Diffusion_UNet(
        ch=FGGEN_DIFFUSION_CONFIG.MODEL_CH,
        out_ch=FGGEN_DIFFUSION_CONFIG.MODEL_OUT_CH,
        ch_mult=FGGEN_DIFFUSION_CONFIG.MODEL_CH_MULT,
        attn_resolutions=FGGEN_DIFFUSION_CONFIG.MODEL_ATTN_RESOLUTIONS,
        dropout=FGGEN_DIFFUSION_CONFIG.MODEL_DROPOUT,
        resamp_with_conv=FGGEN_DIFFUSION_CONFIG.MODEL_RESAMP_WITH_CONV,
        in_channels=FGGEN_DIFFUSION_CONFIG.MODEL_IN_CHANNELS,
        resolution=FGGEN_DIFFUSION_CONFIG.MODEL_RESOLUTION,
        prompt_dict=PROMPT_DICT,
    ).to(DEVICE)
    fg_model, _ = load_model_for_eval(
        FGGEN_DIFFUSION_CONFIG.MODEL_SAVEPATH, fg_model, map_location=DEVICE
    )
    img_model = ImgSyn_Diffusion_UNet(
        ch=IMGSYN_DIFFUSION_CONFIG.MODEL_CH,
        out_ch=IMGSYN_DIFFUSION_CONFIG.MODEL_OUT_CH,
        ch_mult=IMGSYN_DIFFUSION_CONFIG.MODEL_CH_MULT,
        attn_resolutions=IMGSYN_DIFFUSION_CONFIG.MODEL_ATTN_RESOLUTIONS,
        dropout=IMGSYN_DIFFUSION_CONFIG.MODEL_DROPOUT,
        resamp_with_conv=IMGSYN_DIFFUSION_CONFIG.MODEL_RESAMP_WITH_CONV,
        in_channels=IMGSYN_DIFFUSION_CONFIG.MODEL_IN_CHANNELS,
        resolution=IMGSYN_DIFFUSION_CONFIG.MODEL_RESOLUTION,
        prompt_dict=PROMPT_DICT,
    ).to(DEVICE)
    img_model, _ = load_model_for_eval(
        IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH, img_model, map_location=DEVICE
    )
    fgseg_model = FgSeg_UNet(
        in_channels=INPUT_CHANNELS,
        out_channels=UNET_OUTPUT_CHANNELS,
    ).to(DEVICE)
    fgseg_model, _ = load_model_for_eval(
        FGSEG_UNET_MODEL_SAVEPATH, fgseg_model, map_location=DEVICE
    )
    for model in (vae, fg_model, img_model, fgseg_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return vae, fg_model, img_model, fgseg_model, latent_scaling_factor


def _rank_entries(entries, seed):
    return sorted(
        entries,
        key=lambda item: (
            hashlib.sha256(f"{seed}:{item[1]}".encode("utf-8")).digest(),
            item[1],
        ),
    )


def _build_generation_conditions(
    dataset,
    split_name,
    sample_fraction,
    sample_seed,
    max_samples_per_class,
    match_reference_counts,
):
    resolved_entries = build_unique_resolved_entries(dataset)
    candidates = [
        {
            "index": index,
            "condition_id": Path(source_path).stem,
            "label": _label_from_path(source_path),
            "source_image": source_path,
        }
        for index, source_path in resolved_entries
    ]
    selected_indices, selection_path, effective_fraction = resolve_condition_selection(
        candidates,
        dataset=DATASET_NAME,
        split=split_name,
        sample_fraction=sample_fraction,
        seed=sample_seed,
    )
    selected_index_set = set(selected_indices)
    selected_entries = [
        entry for entry in resolved_entries if entry[0] in selected_index_set
    ]
    grouped = defaultdict(list)
    for entry in selected_entries:
        grouped[_label_from_path(entry[1])].append(entry)
    if max_samples_per_class is not None:
        limit = int(max_samples_per_class)
        if limit <= 0:
            raise ValueError("max_samples_per_class must be positive.")
        grouped = {
            label: _rank_entries(entries, sample_seed)[:limit]
            for label, entries in grouped.items()
        }
    if match_reference_counts and (
        not math_is_one(effective_fraction) or max_samples_per_class is not None
    ):
        raise ValueError(
            "--match-reference-counts cannot be combined with fractional or capped sampling."
        )

    conditions = []
    for label in sorted(grouped):
        for class_index, (dataset_index, source_path) in enumerate(grouped[label]):
            conditions.append({
                "label": label,
                "label_id": int(PROMPT_DICT[label]),
                "condition_id": (
                    f"mask_free_{_safe_dataset_token(DATASET_NAME)}_"
                    f"{_safe_dataset_token(label)}_{class_index:06d}"
                ),
                "dataset_index": dataset_index,
                "reference_source_image": source_path,
            })
    if not conditions:
        raise RuntimeError("Mask-free inference selected zero generation conditions.")
    return conditions, selection_path, effective_fraction, len(resolved_entries)


def math_is_one(value):
    return abs(float(value) - 1.0) <= 1.0e-12


def _safe_dataset_token(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def _select_diversity_condition_ids(conditions, per_class):
    grouped = defaultdict(list)
    for condition in conditions:
        grouped[condition["label"]].append(condition)
    selected = set()
    for label, group in grouped.items():
        ranked = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"{label}:{item['condition_id']}".encode("utf-8")
            ).digest(),
        )
        selected.update(
            item["condition_id"] for item in ranked[:int(per_class)]
        )
    return selected


@torch.no_grad()
def _generate_batch(
    vae,
    fg_model,
    img_model,
    fgseg_model,
    labels,
    fggen_seeds,
    imgsyn_seeds,
    sampler_mode,
    latent_scaling_factor,
):
    fg_generators = [build_torch_generator(seed, DEVICE) for seed in fggen_seeds]
    _, foreground_scaled_latent = FgGen_Diffusion_Denoise(
        sampler_mode=sampler_mode,
        prompt_str=list(labels),
        need_to_decode=False,
        seed=int(fggen_seeds[0]),
        diffusion_model=fg_model,
        vae_model=vae,
        generators=fg_generators,
    )
    with build_train_autocast():
        foreground_norm = decode_from_scaled_latent(
            vae, foreground_scaled_latent, latent_scaling_factor
        )
        foreground_latent = descale_latent(
            foreground_scaled_latent, latent_scaling_factor
        )
        imgsyn_generators = [
            build_torch_generator(seed, DEVICE) for seed in imgsyn_seeds
        ]
        prediction_norm = ImgSyn_Diffusion_Denoise(
            sampler_mode=sampler_mode,
            prompt_str=list(labels),
            fg_imgs_e=foreground_latent,
            need_to_decode=True,
            seed=int(imgsyn_seeds[0]),
            diffusion_model=img_model,
            vae_model=vae,
            generators=imgsyn_generators,
        )
        mask_logits = fgseg_model(foreground_norm)
    generated_masks = (torch.sigmoid(mask_logits.float()) >= 0.5).float()
    return prediction_norm.float(), foreground_norm.float(), generated_masks


@torch.no_grad()
def evaluate_mask_free_impgm_pipeline(
    split_name="test",
    sampler_mode=None,
    output_root=None,
    match_reference_counts=False,
    max_samples_per_class=None,
    sample_fraction=None,
    sample_seed=GENERATION_BASE_SEED,
    generate_diversity=True,
    diversity_conditions_per_class=DEFAULT_DIVERSITY_CONDITIONS_PER_CLASS,
    diversity_seeds=DEFAULT_DIVERSITY_SEEDS,
    batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
):
    """Generate autonomous image-mask pairs without using real masks as conditions."""
    split_name = str(split_name).lower()
    sampler_mode = str(sampler_mode or DRAW_SAMPLER_MODE).lower()
    if split_name not in {"train", "valid", "test", "draw"}:
        raise ValueError(f"Unsupported split: {split_name!r}")
    if sampler_mode not in {"ddpm", "ddim"}:
        raise ValueError(f"Unsupported sampler: {sampler_mode!r}")
    if int(diversity_seeds) != DEFAULT_DIVERSITY_SEEDS:
        raise ValueError("The mask-free protocol requires exactly five diversity seeds.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    set_random_seed(int(sample_seed), deterministic=False)
    dataset = _build_dataset(split_name)
    conditions, selection_path, effective_fraction, full_count = _build_generation_conditions(
        dataset,
        split_name,
        sample_fraction,
        int(sample_seed),
        max_samples_per_class,
        bool(match_reference_counts),
    )
    class_counts = Counter(item["label"] for item in conditions)
    print(
        f"[Mask-Free IMPGM] selected {len(conditions)}/{full_count} class-count references; "
        f"per_class={dict(sorted(class_counts.items()))}."
    )

    training_regime, preload_sources = _resolve_training_provenance()
    scheduler_tag, scheduler_description = _resolve_scheduler_identity()
    vae, fg_model, img_model, fgseg_model, latent_scaling_factor = _build_models()
    latent_scaling_factor = normalize_latent_scaling_factor(latent_scaling_factor)
    default_root = PROJECT_ROOT / "Evaluation" / "Full_IMPGM_Mask_Free" / DATASET_NAME
    if training_regime != "from_scratch":
        default_root /= training_regime
    output_root = Path(output_root or default_root / split_name)
    checkpoint = (
        f"vae={VAE_MODEL_SAVEPATH};"
        f"fggen_diffusion={FGGEN_DIFFUSION_CONFIG.MODEL_SAVEPATH};"
        f"imgsyn_diffusion={IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH};"
        f"fgseg={FGSEG_UNET_MODEL_SAVEPATH};scheduler={scheduler_description}"
    )
    writer = MaskFreeGenerationManifestWriter(
        output_root,
        dataset=DATASET_NAME,
        split=split_name,
        checkpoint=checkpoint,
        protocol=(
            "label_noise_to_fggen_diffusion_to_imgsyn_diffusion_and_fgseg;"
            f"scheduler={scheduler_tag};reference_fraction={effective_fraction:.12g};"
            f"reference_selection={selection_path or 'deterministic_inline'};"
            "external_mask_used=false"
        ),
        training_regime=training_regime,
        preload_sources=preload_sources,
        vae_checkpoint=VAE_MODEL_SAVEPATH,
        fggen_checkpoint=FGGEN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
        imgsyn_checkpoint=IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH,
        fgseg_checkpoint=FGSEG_UNET_MODEL_SAVEPATH,
        latent_scaling_factor=latent_scaling_factor,
        sampler=sampler_mode,
        diffusion_steps=int(FGGEN_DIFFUSION_CONFIG.STEPS),
        overwrite=True,
    )

    diversity_ids = (
        _select_diversity_condition_ids(conditions, diversity_conditions_per_class)
        if generate_diversity else set()
    )
    jobs = []
    for condition_index, condition in enumerate(conditions):
        ranks = (
            range(DEFAULT_DIVERSITY_SEEDS)
            if condition["condition_id"] in diversity_ids
            else range(1)
        )
        for seed_rank in ranks:
            fggen_seed = build_generation_seed(condition_index, seed_rank)
            jobs.append({
                **condition,
                "seed_rank": seed_rank,
                "fggen_seed": fggen_seed,
                "imgsyn_seed": fggen_seed + 1,
            })

    progress = tqdm(total=len(jobs), desc="Mask-free IMPGM inference", unit="sample", dynamic_ncols=True)
    for batch_start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[batch_start:batch_start + batch_size]
        prediction_norm, foreground_norm, generated_masks = _generate_batch(
            vae,
            fg_model,
            img_model,
            fgseg_model,
            [job["label"] for job in batch_jobs],
            [job["fggen_seed"] for job in batch_jobs],
            [job["imgsyn_seed"] for job in batch_jobs],
            sampler_mode,
            latent_scaling_factor,
        )
        predictions = denormalize_image_tensor(prediction_norm).float().clamp(0.0, 1.0)
        foregrounds = denormalize_image_tensor(foreground_norm).float().clamp(0.0, 1.0)
        foregrounds = foregrounds * generated_masks

        reference_cache = {}
        if split_name != "train":
            for job in batch_jobs:
                dataset_index = int(job["dataset_index"])
                if dataset_index in reference_cache:
                    continue
                label, _, mask, image_norm, projection, geotransform = dataset[dataset_index]
                if label != job["label"]:
                    raise ValueError(
                        f"Reference label mismatch: expected {job['label']!r}, got {label!r}."
                    )
                reference_cache[dataset_index] = (
                    denormalize_image_tensor(image_norm.unsqueeze(0))[0].float().clamp(0.0, 1.0),
                    mask[:1].float(),
                    projection,
                    geotransform,
                )

        for index, job in enumerate(batch_jobs):
            reference_values = reference_cache.get(int(job["dataset_index"]))
            if reference_values is None:
                reference = reference_mask = projection = geotransform = None
            else:
                reference, reference_mask, projection, geotransform = reference_values
            writer.append_sample(
                prediction=predictions[index],
                foreground=foregrounds[index],
                generated_mask=generated_masks[index],
                label=job["label"],
                label_id=job["label_id"],
                condition_id=job["condition_id"],
                fggen_seed=job["fggen_seed"],
                imgsyn_seed=job["imgsyn_seed"],
                seed_rank=job["seed_rank"],
                reference=reference,
                reference_mask=reference_mask,
                reference_projection=projection,
                reference_geotransform=geotransform,
                reference_source_image=(
                    job["reference_source_image"] if reference is not None else None
                ),
            )
        progress.update(len(batch_jobs))
    progress.close()
    return writer.manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate autonomous IMPGM image-mask pairs without real mask conditions."
    )
    parser.add_argument("--split", choices=("train", "valid", "test", "draw"), default="test")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--match-reference-counts", action="store_true")
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--sample-seed", type=int, default=GENERATION_BASE_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--no-diversity", action="store_true")
    args = parser.parse_args()
    manifest = evaluate_mask_free_impgm_pipeline(
        split_name=args.split,
        sampler_mode=args.sampler,
        output_root=args.output_root,
        match_reference_counts=args.match_reference_counts,
        max_samples_per_class=args.max_samples_per_class,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        generate_diversity=not args.no_diversity,
        batch_size=args.batch_size,
    )
    print(f"Generation manifest: {manifest}")


if __name__ == "__main__":
    main()
