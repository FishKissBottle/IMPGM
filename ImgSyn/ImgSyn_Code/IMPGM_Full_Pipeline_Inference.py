import argparse
import hashlib
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FgGen.FgGen_Code.FgGen_ControlNet import ControlNet_on_FgGen_Diffusion
from FgGen.FgGen_Code.FgGen_ControlNet_Denoise import FgGen_ControlNet_Denoise
from Diffusion_Sampler import build_ddim_time_steps
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion import ImgSyn_Diffusion_UNet
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion_Denoise import ImgSyn_Diffusion_Denoise
from IMPGM_Config import (
    DATASET_DICT,
    DATASET_NAME,
    DEVICE,
    DRAW_SAMPLER_MODE,
    FGGEN_BASE_DIFFUSION_MODEL_SAVEPATH,
    FGGEN_CONTROLNET_CONFIG,
    FGGEN_DIFFUSION_CONFIG,
    IMGSYN_DIFFUSION_CONFIG,
    PROMPT_DICT,
    VAE_MODEL_SAVEPATH,
    VAE_PRELOAD_SOURCE_DATASET,
    VAE_TRAINING_REGIME,
)
from IMPGM_Scheduler import build_beta_schedule_from_config, build_scheduler_tag
from IMPGM_Dataset import IMPGM_Dataset
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import (
    GENERATION_BASE_SEED,
    GenerationManifestWriter,
    build_generation_seed,
    resolve_condition_selection,
)
from IMPGM_Utils import (
    build_torch_generator,
    build_train_autocast,
    denormalize_image_tensor,
    load_controlnet_model_for_eval,
    load_model_for_eval,
    load_standard_vae,
    set_random_seed,
)


DEFAULT_INFERENCE_BATCH_SIZE = 32


def _resolve_full_pipeline_training_provenance():
    components = {
        "vae": (VAE_TRAINING_REGIME, VAE_PRELOAD_SOURCE_DATASET),
        "fggen_diffusion": (
            FGGEN_DIFFUSION_CONFIG.TRAINING_REGIME,
            FGGEN_DIFFUSION_CONFIG.PRELOAD_SOURCE_DATASET,
        ),
        "fggen_controlnet": (
            FGGEN_CONTROLNET_CONFIG.TRAINING_REGIME,
            FGGEN_CONTROLNET_CONFIG.PRELOAD_SOURCE_DATASET,
        ),
        "imgsyn_diffusion": (
            IMGSYN_DIFFUSION_CONFIG.TRAINING_REGIME,
            IMGSYN_DIFFUSION_CONFIG.PRELOAD_SOURCE_DATASET,
        ),
    }
    regimes = {regime for regime, _ in components.values()}
    training_regime = regimes.pop() if len(regimes) == 1 else "mixed"
    preload_sources = ";".join(
        f"{name}={source_dataset}"
        for name, (regime, source_dataset) in components.items()
        if regime != "from_scratch"
    )
    return training_regime, preload_sources or None


def _resolve_full_pipeline_scheduler_identity():
    """Validate the shared scheduler and return its manifest identity."""
    scheduler_configs = (
        FGGEN_DIFFUSION_CONFIG,
        FGGEN_CONTROLNET_CONFIG,
        IMGSYN_DIFFUSION_CONFIG,
    )
    scheduler_types = {
        str(config.SCHEDULER_TYPE).strip().lower()
        for config in scheduler_configs
    }
    if len(scheduler_types) != 1:
        raise ValueError(
            "Full IMPGM scheduler mismatch across FgGen Diffusion, FgGen "
            f"ControlNet, and ImgSyn Diffusion: {sorted(scheduler_types)}."
        )

    scheduler_type = scheduler_types.pop()
    shared_attributes = ["STEPS", "SCHEDULER_MIN_BETA", "SCHEDULER_MAX_BETA"]
    if scheduler_type == "cosine":
        shared_attributes.append("SCHEDULER_COSINE_S")
    elif scheduler_type == "power":
        shared_attributes.append("SCHEDULER_POWER_VAL")
    elif scheduler_type == "sigmoid":
        shared_attributes.extend(
            ["SCHEDULER_SIGMOID_START", "SCHEDULER_SIGMOID_END"]
        )
    scheduler_values = {}
    for attribute in shared_attributes:
        values = {getattr(config, attribute) for config in scheduler_configs}
        if len(values) != 1:
            raise ValueError(
                f"Full IMPGM scheduler parameter {attribute} differs across "
                f"the generation chain: {sorted(values)}."
            )
        scheduler_values[attribute] = values.pop()

    power_value = int(scheduler_values.get(
        "SCHEDULER_POWER_VAL",
        FGGEN_DIFFUSION_CONFIG.SCHEDULER_POWER_VAL,
    ))
    scheduler_tag = build_scheduler_tag(scheduler_type, power_value)
    scheduler_description_parts = [
        f"type={scheduler_type}",
        f"steps={int(scheduler_values['STEPS'])}",
        f"min_beta={float(scheduler_values['SCHEDULER_MIN_BETA']):.12g}",
        f"max_beta={float(scheduler_values['SCHEDULER_MAX_BETA']):.12g}",
    ]
    if scheduler_type == "cosine":
        scheduler_description_parts.append(
            f"cosine_s={float(scheduler_values['SCHEDULER_COSINE_S']):.12g}"
        )
    elif scheduler_type == "power":
        scheduler_description_parts.append(f"exponent={power_value}")
    elif scheduler_type == "sigmoid":
        scheduler_description_parts.extend([
            f"sigmoid_start={float(scheduler_values['SCHEDULER_SIGMOID_START']):.12g}",
            f"sigmoid_end={float(scheduler_values['SCHEDULER_SIGMOID_END']):.12g}",
        ])
    scheduler_description = ",".join(scheduler_description_parts)
    return f"Full_IMPGM_{scheduler_tag}", scheduler_tag, scheduler_description


def _actual_ddim_steps(task_config, requested_steps):
    beta_t = build_beta_schedule_from_config(task_config)
    alpha_t_bar = torch.cumprod(1.0 - beta_t, dim=0)
    time_steps, _ = build_ddim_time_steps(
        alpha_t_bar,
        steps=requested_steps,
        sample_method="alpha_space",
    )
    return int(len(time_steps))


def resolve_full_pipeline_sampling_config(
    sampler_mode,
    *,
    ddim_steps=None,
    ddim_eta=None,
    fggen_config=FGGEN_CONTROLNET_CONFIG,
):
    sampler_mode = str(sampler_mode).strip().lower()
    if sampler_mode not in {"ddpm", "ddim"}:
        raise ValueError(f"Unsupported sampler: {sampler_mode!r}")

    fg_training_steps = int(fggen_config.STEPS)
    img_training_steps = int(IMGSYN_DIFFUSION_CONFIG.STEPS)
    if sampler_mode == "ddpm":
        if ddim_steps is not None or ddim_eta is not None:
            raise ValueError(
                "--ddim-steps and --ddim-eta may only be used with sampler='ddim'."
            )
        return {
            "sampler": "ddpm",
            "requested_steps_per_stage": fg_training_steps,
            "actual_fggen_steps": fg_training_steps,
            "actual_imgsyn_steps": img_training_steps,
            "actual_total_steps": fg_training_steps + img_training_steps,
            "ddim_eta": None,
            "deterministic_given_initial_noise": False,
        }

    if ddim_steps is None or ddim_eta is None:
        raise ValueError(
            "DDIM inference requires explicit ddim_steps and ddim_eta values."
        )
    requested_steps = int(ddim_steps)
    eta = float(ddim_eta)
    maximum_steps = min(fg_training_steps, img_training_steps)
    if requested_steps <= 0 or requested_steps > maximum_steps:
        raise ValueError(
            f"DDIM steps must be in [1, {maximum_steps}], got {requested_steps}."
        )
    if eta < 0.0:
        raise ValueError(f"DDIM eta must be non-negative, got {eta}.")

    actual_fggen_steps = _actual_ddim_steps(
        fggen_config,
        requested_steps,
    )
    actual_imgsyn_steps = _actual_ddim_steps(
        IMGSYN_DIFFUSION_CONFIG,
        requested_steps,
    )
    return {
        "sampler": "ddim",
        "requested_steps_per_stage": requested_steps,
        "actual_fggen_steps": actual_fggen_steps,
        "actual_imgsyn_steps": actual_imgsyn_steps,
        "actual_total_steps": actual_fggen_steps + actual_imgsyn_steps,
        "ddim_eta": eta,
        "deterministic_given_initial_noise": bool(eta == 0.0),
    }


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


def _build_models():
    fg_config = FGGEN_CONTROLNET_CONFIG
    img_config = IMGSYN_DIFFUSION_CONFIG
    for path in (
        VAE_MODEL_SAVEPATH,
        fg_config.MODEL_SAVEPATH,
        img_config.MODEL_SAVEPATH,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Full IMPGM evaluation checkpoint does not exist: {path}")

    vae, _, _ = load_standard_vae(VAE_MODEL_SAVEPATH, device=DEVICE, load_ema=True)
    vae.eval()
    fg_model = ControlNet_on_FgGen_Diffusion(
        FgGen_Diffusion_model_savepath=FGGEN_BASE_DIFFUSION_MODEL_SAVEPATH,
        ch=fg_config.MODEL_CH,
        out_ch=fg_config.MODEL_OUT_CH,
        ch_mult=fg_config.MODEL_CH_MULT,
        attn_resolutions=fg_config.MODEL_ATTN_RESOLUTIONS,
        dropout=fg_config.MODEL_DROPOUT,
        resamp_with_conv=fg_config.MODEL_RESAMP_WITH_CONV,
        in_channels=fg_config.MODEL_IN_CHANNELS,
        resolution=fg_config.MODEL_RESOLUTION,
        conditional_ch=fg_config.MODEL_CONDITIONAL_CH,
        ControlNet_weight=fg_config.MODEL_CONTROLNET_WEIGHT,
        prompt_dict=PROMPT_DICT,
    ).to(DEVICE)
    fg_model, _ = load_controlnet_model_for_eval(
        fg_config.MODEL_SAVEPATH,
        fg_model,
        map_location=DEVICE,
    )
    fg_model.eval()

    img_model = ImgSyn_Diffusion_UNet(
        ch=img_config.MODEL_CH,
        out_ch=img_config.MODEL_OUT_CH,
        ch_mult=img_config.MODEL_CH_MULT,
        attn_resolutions=img_config.MODEL_ATTN_RESOLUTIONS,
        dropout=img_config.MODEL_DROPOUT,
        resamp_with_conv=img_config.MODEL_RESAMP_WITH_CONV,
        in_channels=img_config.MODEL_IN_CHANNELS,
        resolution=img_config.MODEL_RESOLUTION,
        prompt_dict=PROMPT_DICT,
    ).to(DEVICE)
    img_model, _ = load_model_for_eval(
        img_config.MODEL_SAVEPATH,
        img_model,
        map_location=DEVICE,
    )
    img_model.eval()
    return vae, fg_model, img_model


def _condition_label_from_path(path):
    tokens = Path(path).stem.split("_")
    if len(tokens) < 2:
        raise ValueError(f"Cannot derive IMPGM label from {path}")
    return tokens[-2]


def _resolved_evaluation_entries(dataset):
    """Return unique catalog indices paired with the source path actually loaded."""
    entries = []
    seen_paths = set()
    for index in range(len(dataset)):
        source_path = dataset.resolve_image_path(index)
        source_key = str(Path(source_path).resolve()).casefold()
        if source_key in seen_paths:
            continue
        seen_paths.add(source_key)
        entries.append((index, source_path))
    return entries


def _select_diversity_indices(entries, per_class=20):
    grouped = {}
    entry_paths = dict(entries)
    for index, path in entries:
        grouped.setdefault(_condition_label_from_path(path), []).append(index)
    selected = set()
    for label, indices in grouped.items():
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{label}:{Path(entry_paths[index]).stem}".encode("utf-8")
            ).digest(),
        )
        selected.update(ranked[: int(per_class)])
    return selected


@torch.no_grad()
def _generate_batch(
    vae,
    fg_model,
    img_model,
    labels,
    masks,
    sampler_mode,
    seeds,
    ddim_steps=None,
    ddim_eta=None,
):
    if not labels or len(labels) != len(masks) or len(labels) != len(seeds):
        raise ValueError("labels, masks, and seeds must be non-empty and have equal lengths.")
    masks = torch.stack(masks, dim=0).to(DEVICE).float()
    fg_generators = [build_torch_generator(int(seed), DEVICE) for seed in seeds]
    fg_prediction = FgGen_ControlNet_Denoise(
        sampler_mode=sampler_mode,
        prompt_str=list(labels),
        conditional_element=masks,
        need_to_decode=True,
        seed=int(seeds[0]),
        controlnet_model=fg_model,
        vae_model=vae,
        generators=fg_generators,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
    )
    with build_train_autocast():
        fg_parameters = vae.encode(fg_prediction)
        fg_latent, _, _ = vae.reparameterize(fg_parameters)
        img_generators = [
            build_torch_generator(int(seed) + 1, DEVICE)
            for seed in seeds
        ]
        prediction = ImgSyn_Diffusion_Denoise(
            sampler_mode=sampler_mode,
            prompt_str=list(labels),
            fg_imgs_e=fg_latent,
            need_to_decode=True,
            seed=int(seeds[0]) + 1,
            diffusion_model=img_model,
            vae_model=vae,
            generators=img_generators,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
    return prediction.float()


@torch.no_grad()
def evaluate_full_impgm_pipeline(
    split_name="test",
    sampler_mode=None,
    output_root=None,
    max_samples=None,
    generate_diversity=True,
    diversity_conditions_per_class=20,
    diversity_seeds=5,
    batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    sample_fraction=None,
    condition_selection_path=None,
    ddim_steps=None,
    ddim_eta=None,
):
    """Generate mask-only Full IMPGM samples and write the unified manifest."""
    split_name = str(split_name).lower()
    sampler_mode = str(sampler_mode or DRAW_SAMPLER_MODE).lower()
    if split_name not in {"train", "valid", "test", "draw"}:
        raise ValueError(f"Unsupported split: {split_name!r}")
    if sampler_mode not in {"ddpm", "ddim"}:
        raise ValueError(f"Unsupported sampler: {sampler_mode!r}")
    sampling_config = resolve_full_pipeline_sampling_config(
        sampler_mode,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
    )
    if int(diversity_seeds) != 5:
        raise ValueError("The unified protocol requires exactly five diversity seeds.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    set_random_seed(GENERATION_BASE_SEED, deterministic=False)
    dataset = _build_dataset(split_name)
    evaluation_entries = _resolved_evaluation_entries(dataset)
    if not evaluation_entries:
        raise RuntimeError(f"Full IMPGM inference split {split_name!r} is empty.")
    full_condition_count = len(evaluation_entries)
    candidates = [
        {
            "index": index,
            "condition_id": Path(source_path).stem,
            "label": _condition_label_from_path(source_path),
            "source_image": source_path,
        }
        for index, source_path in evaluation_entries
    ]
    selected_indices, selection_path, effective_fraction = resolve_condition_selection(
        candidates,
        dataset=DATASET_NAME,
        split=split_name,
        sample_fraction=sample_fraction,
        selection_path=condition_selection_path,
    )
    selected_index_set = set(selected_indices)
    evaluation_entries = [
        entry for entry in evaluation_entries if entry[0] in selected_index_set
    ]
    if max_samples is not None:
        evaluation_entries = evaluation_entries[:min(len(evaluation_entries), int(max_samples))]
    if not evaluation_entries:
        raise RuntimeError("Full IMPGM inference selected zero conditions.")
    print(
        f"[Full IMPGM Inference] selected {len(evaluation_entries)}/{full_condition_count} "
        f"conditions with sample_fraction={effective_fraction:.4f}, "
        f"subset_seed={GENERATION_BASE_SEED}."
    )
    method_name, scheduler_tag, scheduler_description = (
        _resolve_full_pipeline_scheduler_identity()
    )
    training_regime, preload_sources = _resolve_full_pipeline_training_provenance()
    vae, fg_model, img_model = _build_models()
    if sampler_mode == "ddim":
        default_output_root = (
            PROJECT_ROOT
            / "Evaluation"
            / "Sampling_Strategy_Efficiency"
            / DATASET_NAME
            / f"ddim_{sampling_config['requested_steps_per_stage']}"
        )
    else:
        default_output_root = PROJECT_ROOT / "Evaluation" / "Full_IMPGM" / DATASET_NAME
        if training_regime != "from_scratch":
            default_output_root /= training_regime
    output_root = Path(output_root or (default_output_root / split_name))
    eta_token = (
        "not_applicable"
        if sampling_config["ddim_eta"] is None
        else f"{sampling_config['ddim_eta']:.12g}"
    )
    sampling_description = (
        f"sampler={sampler_mode};"
        f"requested_steps_per_stage={sampling_config['requested_steps_per_stage']};"
        f"actual_fggen_steps={sampling_config['actual_fggen_steps']};"
        f"actual_imgsyn_steps={sampling_config['actual_imgsyn_steps']};"
        f"actual_total_steps={sampling_config['actual_total_steps']};"
        f"ddim_eta={eta_token}"
    )
    checkpoint = (
        f"fggen_diffusion={FGGEN_BASE_DIFFUSION_MODEL_SAVEPATH};"
        f"fggen_controlnet={FGGEN_CONTROLNET_CONFIG.MODEL_SAVEPATH};"
        f"imgsyn_diffusion={IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH};"
        f"vae={VAE_MODEL_SAVEPATH};"
        f"scheduler={scheduler_description};{sampling_description}"
    )
    writer = GenerationManifestWriter(
        output_root,
        method=method_name,
        dataset=DATASET_NAME,
        split=split_name,
        checkpoint=checkpoint,
        training_regime=training_regime,
        preload_sources=preload_sources,
        protocol=(
            "mask_label_to_fggen_controlnet_to_imgsyn_diffusion;"
            f"scheduler={scheduler_tag};{sampling_description};"
            f"condition_fraction={effective_fraction:.12g};"
            f"condition_selection={selection_path or 'deterministic_inline'}"
        ),
        sampler=sampler_mode,
        diffusion_steps=sampling_config["requested_steps_per_stage"],
        requested_steps_per_stage=sampling_config["requested_steps_per_stage"],
        actual_fggen_steps=sampling_config["actual_fggen_steps"],
        actual_imgsyn_steps=sampling_config["actual_imgsyn_steps"],
        actual_total_steps=sampling_config["actual_total_steps"],
        ddim_eta=sampling_config["ddim_eta"],
        scheduler_type=str(FGGEN_CONTROLNET_CONFIG.SCHEDULER_TYPE).lower(),
        overwrite=True,
    )
    diversity_indices = (
        _select_diversity_indices(evaluation_entries, diversity_conditions_per_class)
        if generate_diversity else set()
    )
    generation_jobs = []
    for evaluation_index, (index, source_path) in enumerate(evaluation_entries):
        seed_ranks = range(diversity_seeds) if index in diversity_indices else range(1)
        for seed_rank in seed_ranks:
            seed = build_generation_seed(evaluation_index, seed_rank)
            generation_jobs.append(
                {
                    "index": index,
                    "source_path": source_path,
                    "seed": seed,
                    "seed_rank": seed_rank,
                }
            )

    for batch_start in range(0, len(generation_jobs), batch_size):
        batch_jobs = generation_jobs[batch_start:batch_start + batch_size]
        batch_samples = []
        for job in batch_jobs:
            label, _, mask, reference_norm, projection, geotransform = dataset[job["index"]]
            if not isinstance(mask, torch.Tensor):
                raise TypeError("Full IMPGM evaluation requires tensor condition masks.")
            batch_samples.append(
                {
                    **job,
                    "label": label,
                    "mask": mask,
                    "reference": denormalize_image_tensor(
                        reference_norm.unsqueeze(0)
                    )[0].float().clamp(0.0, 1.0),
                    "projection": projection,
                    "geotransform": geotransform,
                    "condition_id": Path(job["source_path"]).stem,
                }
            )

        prediction_norm = _generate_batch(
            vae,
            fg_model,
            img_model,
            [sample["label"] for sample in batch_samples],
            [sample["mask"] for sample in batch_samples],
            sampler_mode,
            [sample["seed"] for sample in batch_samples],
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        predictions = denormalize_image_tensor(prediction_norm).float().clamp(0.0, 1.0)
        for prediction, sample in zip(predictions, batch_samples):
            writer.append_sample(
                prediction=prediction,
                reference=sample["reference"],
                condition_mask=sample["mask"],
                label=sample["label"],
                label_id=PROMPT_DICT[sample["label"]],
                condition_id=sample["condition_id"],
                seed=sample["seed"],
                seed_rank=sample["seed_rank"],
                # Different initial seeds still define a stochastic sample set,
                # even when DDIM eta=0 makes each fixed-seed trajectory deterministic.
                deterministic=False,
                projection=sample["projection"],
                geotransform=sample["geotransform"],
                source_image=sample["source_path"],
            )
        print(
            f"[Full IMPGM Inference] {min(batch_start + len(batch_jobs), len(generation_jobs))}/"
            f"{len(generation_jobs)} generated samples, batch_size={len(batch_jobs)}"
        )
    return writer.manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate the mask-only Full IMPGM benchmark set.")
    parser.add_argument("--split", choices=("train", "valid", "test", "draw"), default="test")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default=None)
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=None,
        help="Requested DDIM steps per generation stage; required for DDIM.",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=None,
        help="DDIM stochasticity coefficient; required for DDIM (use 0.0 for deterministic sampling).",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help=(
            "Deterministic class-stratified condition fraction in (0, 1]. "
            "Defaults to 0.15 for train and 1.0 for other splits."
        ),
    )
    parser.add_argument(
        "--condition-selection",
        default=None,
        help="Shared condition-selection JSON; train uses the common default when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help=f"Number of generated samples per inference batch (default: {DEFAULT_INFERENCE_BATCH_SIZE}).",
    )
    parser.add_argument("--no-diversity", action="store_true")
    args = parser.parse_args()
    effective_sampler = str(args.sampler or DRAW_SAMPLER_MODE).lower()
    if effective_sampler == "ddim" and (
        args.ddim_steps is None or args.ddim_eta is None
    ):
        parser.error("DDIM requires both --ddim-steps and --ddim-eta.")
    if effective_sampler != "ddim" and (
        args.ddim_steps is not None or args.ddim_eta is not None
    ):
        parser.error("--ddim-steps and --ddim-eta may only be used with --sampler ddim.")
    manifest = evaluate_full_impgm_pipeline(
        split_name=args.split,
        sampler_mode=args.sampler,
        output_root=args.output_root,
        max_samples=args.max_samples,
        generate_diversity=not args.no_diversity,
        batch_size=args.batch_size,
        sample_fraction=args.sample_fraction,
        condition_selection_path=args.condition_selection,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
    )
    print(f"Generation manifest: {manifest}")


if __name__ == "__main__":
    main()
