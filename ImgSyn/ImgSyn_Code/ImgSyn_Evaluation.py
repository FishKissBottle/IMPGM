import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import (
    DATASET_DICT,
    DATASET_NAME,
    DATA_TRANSFER_NON_BLOCKING,
    DEVICE,
    DIFFUSION_DRAW_MICROBATCH_SIZE,
    DIFFUSION_TEST_MICROBATCH_SIZE,
    DIFFUSION_VALID_MICROBATCH_SIZE,
    DRAW_BATCH_SIZE,
    DRAW_RANDOM_SEED,
    DRAW_SAMPLER_MODE,
    IMG_SIZE,
    IMGSYN_BASE_DIFFUSION_MODEL_SAVEPATH,
    IMGSYN_CONTROLNET_CONFIG,
    IMGSYN_DIFFUSION_CONFIG,
    LATENT_HIDDENCHANNEL,
    NUM_WORKERS,
    PIN_MEMORY,
    PROMPT_DICT,
    PROJECT_ROOT,
    RANDOM_SEED,
    SIDELENGTH_SCALE_FACTOR,
    TEST_BATCH_SIZE,
    VALID_BATCH_SIZE,
    VAE_MODEL_SAVEPATH,
)
from IMPGM_Dataset import IMPGM_Dataset
from Evaluation.Evaluation_Code.IMPGM_Quality_Metrics import (
    compute_FSIM,
    compute_GMSD,
)
from IMPGM_Utils import (
    build_dataloader_generator,
    build_torch_generator,
    build_train_autocast,
    denormalize_image_tensor,
    extract_high_frequency,
    iter_microbatch_slices,
    load_controlnet_model_for_eval,
    load_model_for_eval,
    load_standard_vae,
    prepare_rgb_vis_tensor,
    randn,
    save_rgb_datas,
    save_tif_datas,
    seed_dataloader_worker,
    set_random_seed,
    slice_microbatch,
)
from ImgSyn.ImgSyn_Code.ImgSyn_ControlNet import (
    ControlNet_on_ImgSyn_Diffusion,
)
from ImgSyn.ImgSyn_Code.ImgSyn_ControlNet_Denoise import (
    ImgSyn_ControlNet_Denoise,
)
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion import ImgSyn_Diffusion_UNet
from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion_Denoise import (
    ImgSyn_Diffusion_Denoise,
)


_SPLIT_RUNTIME = {
    "valid": (VALID_BATCH_SIZE, DIFFUSION_VALID_MICROBATCH_SIZE),
    "test": (TEST_BATCH_SIZE, DIFFUSION_TEST_MICROBATCH_SIZE),
    "draw": (DRAW_BATCH_SIZE, DIFFUSION_DRAW_MICROBATCH_SIZE),
}


def _split_dataset_keys(split_name):
    suffix = {"valid": "Valid", "test": "Test", "draw": "Draw"}[split_name]
    return f"img_rootdir_list_for{suffix}", f"msk_rootdir_list_for{suffix}"


def _build_data_loader(split_name, batch_size=None):
    if DATASET_DICT is None:
        raise RuntimeError("DATASET_DICT must not be None.")
    if split_name not in _SPLIT_RUNTIME:
        raise ValueError(f"Unsupported evaluation split: {split_name!r}")

    image_key, mask_key = _split_dataset_keys(split_name)
    if image_key not in DATASET_DICT or mask_key not in DATASET_DICT:
        raise KeyError(
            f"Dataset {DATASET_NAME!r} does not define the {split_name!r} split."
        )

    configured_batch_size, configured_microbatch_size = _SPLIT_RUNTIME[split_name]
    if batch_size is None:
        batch_size = configured_batch_size
        microbatch_size = configured_microbatch_size
    else:
        batch_size = int(batch_size)
        microbatch_size = batch_size
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    dataset = IMPGM_Dataset(
        img_rootdir_list=list(DATASET_DICT[image_key]),
        msk_rootdir_list=list(DATASET_DICT[mask_key]),
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(RANDOM_SEED),
    )
    return loader, int(microbatch_size)


def _build_diffusion_model(config):
    model = ImgSyn_Diffusion_UNet(
        ch=config.MODEL_CH,
        out_ch=config.MODEL_OUT_CH,
        ch_mult=config.MODEL_CH_MULT,
        attn_resolutions=config.MODEL_ATTN_RESOLUTIONS,
        dropout=config.MODEL_DROPOUT,
        resamp_with_conv=config.MODEL_RESAMP_WITH_CONV,
        in_channels=config.MODEL_IN_CHANNELS,
        resolution=config.MODEL_RESOLUTION,
        prompt_dict=PROMPT_DICT,
    ).to(DEVICE)
    model, _ = load_model_for_eval(
        config.MODEL_SAVEPATH,
        model,
        map_location=DEVICE,
    )
    return model.eval()


def _build_controlnet_model(config):
    model = ControlNet_on_ImgSyn_Diffusion(
        ImgSyn_Diffusion_model_savepath=IMGSYN_BASE_DIFFUSION_MODEL_SAVEPATH,
        ch=config.MODEL_CH,
        out_ch=config.MODEL_OUT_CH,
        ch_mult=config.MODEL_CH_MULT,
        attn_resolutions=config.MODEL_ATTN_RESOLUTIONS,
        dropout=config.MODEL_DROPOUT,
        resamp_with_conv=config.MODEL_RESAMP_WITH_CONV,
        in_channels=config.MODEL_IN_CHANNELS,
        resolution=config.MODEL_RESOLUTION,
        conditional_ch=config.MODEL_CONDITIONAL_CH,
        ControlNet_weight=config.MODEL_CONTROLNET_WEIGHT,
        prompt_dict=PROMPT_DICT,
    ).to(DEVICE)
    model, _ = load_controlnet_model_for_eval(
        config.MODEL_SAVEPATH,
        model,
        map_location=DEVICE,
    )
    return model.eval()


def _metric_values(prediction, target, include_paired_spatial_metrics):
    prediction = denormalize_image_tensor(prediction.float()).clamp(0.0, 1.0)
    target = denormalize_image_tensor(target.float()).clamp(0.0, 1.0)
    if not include_paired_spatial_metrics:
        return {}

    rgb_gmsd, nir_gmsd = compute_GMSD(prediction, target)
    rgb_fsim, nir_fsim = compute_FSIM(prediction, target)
    metrics = {
        "rgb_gmsd": float(rgb_gmsd.detach().item()),
        "rgb_fsim": float(rgb_fsim.detach().item()),
    }
    if nir_gmsd is not None:
        metrics["nir_gmsd"] = float(nir_gmsd.detach().item())
    if nir_fsim is not None:
        metrics["nir_fsim"] = float(nir_fsim.detach().item())
    return metrics


def _collated_geotransforms_to_list(geotransforms):
    if torch.is_tensor(geotransforms):
        values = geotransforms.detach().cpu()
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(
                f"Expected geotransforms with shape [B, 6], got {tuple(values.shape)}."
            )
        return values.tolist()

    if (
        isinstance(geotransforms, (list, tuple))
        and len(geotransforms) == 6
        and all(torch.is_tensor(value) for value in geotransforms)
    ):
        return torch.stack(
            [value.detach().cpu() for value in geotransforms], dim=1
        ).tolist()

    values = [list(value) for value in geotransforms]
    if any(len(value) != 6 for value in values):
        raise ValueError("Every geotransform must contain six values.")
    return values


def _save_outputs(
    prediction,
    labels,
    projections,
    geotransforms,
    output_root,
    batch_idx,
):
    rgb_dir = output_root / "rgb"
    tif_dir = output_root / "tif"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    tif_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = rgb_dir / f"batch_{batch_idx:04d}.png"
    tif_path = tif_dir / f"batch_{batch_idx:04d}.tif"
    save_rgb_datas(
        prepare_rgb_vis_tensor(prediction).cpu(),
        nrow=min(3, int(prediction.shape[0])),
        savepath=str(rgb_path),
        is_makegrid=False,
        prompt_strs=labels,
    )
    save_tif_datas(
        prediction.cpu(),
        projections=list(projections),
        geotransforms=_collated_geotransforms_to_list(geotransforms),
        savepath=str(tif_path),
        prompt_strs=labels,
    )


def _evaluate_batch(
    task_name,
    config,
    model,
    vae_model,
    batch,
    microbatch_size,
    sampler_mode,
    first_sample_index,
):
    labels, fg_imgs, masks, target_imgs, _, _ = batch
    batch_size = int(target_imgs.shape[0])
    prediction_parts = []

    for mb_start, mb_end in iter_microbatch_slices(batch_size, microbatch_size):
        labels_mb = slice_microbatch(labels, mb_start, mb_end)
        fg_imgs_mb = slice_microbatch(fg_imgs, mb_start, mb_end).to(
            DEVICE,
            non_blocking=DATA_TRANSFER_NON_BLOCKING,
        )
        sample_seed = int(DRAW_RANDOM_SEED + first_sample_index + mb_start)
        generator = build_torch_generator(sample_seed, DEVICE)
        latent_size = IMG_SIZE // SIDELENGTH_SCALE_FACTOR
        initial_noise = randn(
            (
                int(mb_end - mb_start),
                LATENT_HIDDENCHANNEL,
                latent_size,
                latent_size,
            ),
            device=DEVICE,
            generator=generator,
        )

        with build_train_autocast():
            fg_params = vae_model.encode(fg_imgs_mb)
            fg_latents, _, _ = vae_model.reparameterize(fg_params)

            if task_name == "imgsyn_diffusion":
                prediction_mb = ImgSyn_Diffusion_Denoise(
                    sampler_mode=sampler_mode,
                    prompt_str=labels_mb,
                    fg_imgs_e=fg_latents,
                    seed=sample_seed,
                    diffusion_model=model,
                    vae_model=vae_model,
                    initial_noise=initial_noise,
                )
            else:
                target_mb = slice_microbatch(target_imgs, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                masks_mb = slice_microbatch(masks, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                high_frequency = extract_high_frequency(
                    target_mb,
                    config.CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE,
                )
                condition = high_frequency * (1.0 - masks_mb)
                prediction_mb = ImgSyn_ControlNet_Denoise(
                    sampler_mode=sampler_mode,
                    prompt_str=labels_mb,
                    fg_imgs_e=fg_latents,
                    conditional_element=condition,
                    seed=sample_seed,
                    controlnet_model=model,
                    vae_model=vae_model,
                    initial_noise=initial_noise,
                )

        prediction_parts.append(prediction_mb.float())

    return torch.cat(prediction_parts, dim=0)


@torch.no_grad()
def evaluate_imgsyn_model(
    task_name,
    split_name="test",
    sampler_mode=None,
    save_outputs=True,
    max_batches=None,
    batch_size=None,
):
    if task_name not in {"imgsyn_diffusion", "imgsyn_controlnet"}:
        raise ValueError(f"Unsupported ImgSyn evaluation task: {task_name!r}")
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided.")

    sampler_mode = str(sampler_mode or DRAW_SAMPLER_MODE).strip().lower()
    if sampler_mode not in {"ddpm", "ddim"}:
        raise ValueError(f"Unsupported sampler mode: {sampler_mode!r}")

    config = (
        IMGSYN_DIFFUSION_CONFIG
        if task_name == "imgsyn_diffusion"
        else IMGSYN_CONTROLNET_CONFIG
    )
    display_name = (
        "ImgSyn_Diffusion"
        if task_name == "imgsyn_diffusion"
        else "ImgSyn_ControlNet"
    )
    checkpoint_path = Path(config.MODEL_SAVEPATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{display_name} checkpoint does not exist: {checkpoint_path}"
        )

    set_random_seed(RANDOM_SEED, deterministic=False)
    data_loader, microbatch_size = _build_data_loader(
        split_name,
        batch_size=batch_size,
    )
    effective_batch_size = int(data_loader.batch_size)
    vae_model, _, _ = load_standard_vae(
        VAE_MODEL_SAVEPATH,
        device=DEVICE,
        load_ema=True,
    )
    model = (
        _build_diffusion_model(config)
        if task_name == "imgsyn_diffusion"
        else _build_controlnet_model(config)
    )

    output_root = (
        Path(PROJECT_ROOT)
        / "ImgSyn"
        / f"{display_name}_Evaluation"
        / config.EXP_NAME
        / split_name
    )
    output_root.mkdir(parents=True, exist_ok=True)

    total_samples = len(data_loader.dataset)
    if max_batches is not None:
        total_samples = min(
            total_samples,
            int(max_batches) * effective_batch_size,
        )
    print(
        f"[{display_name}] Evaluating {total_samples} {split_name} samples "
        f"with batch_size={effective_batch_size}."
    )

    metric_sums = {}
    sample_count = 0
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        target_imgs = batch[3].to(
            DEVICE,
            non_blocking=DATA_TRANSFER_NON_BLOCKING,
        )
        prediction = _evaluate_batch(
            task_name,
            config,
            model,
            vae_model,
            batch,
            microbatch_size,
            sampler_mode,
            sample_count,
        )
        metrics = _metric_values(
            prediction,
            target_imgs,
            include_paired_spatial_metrics=True,
        )
        current_batch_size = int(target_imgs.shape[0])
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value * current_batch_size

        if save_outputs:
            _save_outputs(
                prediction,
                batch[0],
                batch[4],
                batch[5],
                output_root,
                batch_idx,
            )
        sample_count += current_batch_size
        print(f"[{display_name}] Completed {sample_count}/{total_samples} samples.")

    if sample_count == 0:
        raise RuntimeError(
            f"{display_name} evaluation processed no {split_name} samples."
        )

    metrics = {key: value / sample_count for key, value in metric_sums.items()}
    result = {
        "task": task_name,
        "evaluation_protocol": "oracle_condition_internal_diagnostic",
        "main_benchmark_eligible": False,
        "protocol_note": (
            "This diagnostic injects the real foreground latent; the ControlNet "
            "variant also injects real-target background high-frequency information. "
            "Use IMPGM_Full_Pipeline_Inference.py and then "
            "IMPGM_Full_Pipeline_Evaluation.py for the mask-only comparison table."
        ),
        "dataset": DATASET_NAME,
        "split": split_name,
        "sampler": sampler_mode,
        "seed": int(DRAW_RANDOM_SEED),
        "batch_size": effective_batch_size,
        "microbatch_size": int(microbatch_size),
        "num_samples": int(sample_count),
        "max_batches": None if max_batches is None else int(max_batches),
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }
    result_path = output_root / "evaluation_metrics.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metric_text = ", ".join(
        f"{key}={value:.6f}" for key, value in sorted(metrics.items())
    )
    summary = (
        f"[{display_name}] Dataset={DATASET_NAME}, Split={split_name}, "
        f"Sampler={sampler_mode}, Samples={sample_count}: {metric_text}"
    )
    print(summary)
    (output_root / "EvaluationInfo.txt").write_text(
        summary + "\n",
        encoding="utf-8",
    )
    return metrics
