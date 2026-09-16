import copy
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from osgeo import osr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid

from IMPGM_TifReader import Tif_Read_and_Write
from IMPGM_Config import (
    AMP_DTYPE,
    AMP_ENABLED,
    DEVICE,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_CHANNELS,
    LATENT_HIDDENCHANNEL,
    LATENT_SCALING_FACTOR,
    LATENT_SCALING_FACTOR_FIELD,
    RGB_VIS_STRETCH,
    VAE_DOWN_CHANNELS,
    VAE_MID_CHANNELS,
    VAE_MODEL_SAVEPATH,
    VAE_NORM_CHANNELS,
    VAE_NUM_DOWN_LAYERS,
    VAE_NUM_MID_LAYERS,
    VAE_NUM_UP_LAYERS,
    VAE_Z_CHANNEL,
    VIS_BAND_ORDER,
    PLACEHOLDER_TOP_LEFT_X,
    PLACEHOLDER_TOP_LEFT_Y,
    PLACEHOLDER_PIXEL_WIDTH,
    PLACEHOLDER_PIXEL_HEIGHT,
    PLACEHOLDER_EPSG,
)


# Precision and AMP

def _xpu_state_api_available(name: str) -> bool:
    return hasattr(torch, "xpu") and hasattr(torch.xpu, name)


def build_train_autocast():
    device_type = torch.device(DEVICE).type
    return torch.amp.autocast(device_type, dtype=AMP_DTYPE, enabled=AMP_ENABLED)


def build_scheduler(optimizer, min_lr, patience):
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        min_lr=min_lr,
        factor=0.5,
        threshold=0.0,
        patience=patience,
    )


def build_diagonal_tile_schedule(rows, cols, multiplier):
    result = []
    for diagonal_index in range(rows + cols - 1):
        diagonal = []
        current_row = min(diagonal_index, rows - 1)
        while current_row >= 0:
            current_col = diagonal_index - current_row
            if current_col >= cols:
                break
            diagonal.append(
                [current_row * multiplier, current_col * multiplier]
            )
            current_row -= 1
        result.append(diagonal)
    return result


def resolve_microbatch_size(microbatch_size, batch_size: int) -> int:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}")

    if microbatch_size is None:
        return batch_size

    microbatch_size = int(microbatch_size)
    return max(1, min(microbatch_size, batch_size))


def iter_microbatch_slices(batch_size: int, microbatch_size):
    """Yield contiguous ``(start, end)`` pairs that partition a batch.

    This matches the microbatch loop style used in ``CSUA_LDM`` so that
    training scripts can share the same higher-level organization:

    ``for mb_start, mb_end in iter_microbatch_slices(...):``
    """
    chunk_size = resolve_microbatch_size(microbatch_size, batch_size)
    for start in range(0, int(batch_size), chunk_size):
        end = min(start + chunk_size, int(batch_size))
        yield start, end


def slice_microbatch(value, start: int, end: int):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(value[start:end])
    return value[start:end]


def extract_diffusion_coefficient(values, timesteps, target_shape):
    """Extract per-sample diffusion coefficients and reshape for broadcasting."""
    output = torch.gather(values, index=timesteps, dim=0)
    output = output.to(device=timesteps.device, dtype=torch.float32)
    return output.view([timesteps.shape[0]] + [1] * (len(target_shape) - 1))


def extract_high_frequency(imgs, gaussian_scale):
    """Extract a Gaussian high-pass residual in FP32 on the input device."""
    if not torch.is_tensor(imgs) or imgs.ndim not in (3, 4):
        shape = getattr(imgs, "shape", None)
        raise ValueError(f"Expected a 3D or 4D image tensor, got shape {shape}.")

    gaussian_scale = float(gaussian_scale)
    if not 0.0 < gaussian_scale <= 1.0:
        raise ValueError(
            f"`gaussian_scale` must be in (0, 1], got {gaussian_scale}."
        )

    height, width = imgs.shape[-2:]
    if height < 2 or width < 2:
        raise ValueError(f"Image dimensions must be at least 2x2, got {height}x{width}.")

    device_type = imgs.device.type
    with torch.amp.autocast(device_type, enabled=False):
        imgs_fp32 = imgs.to(dtype=torch.float32)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=imgs.device, dtype=torch.float32),
            torch.arange(width, device=imgs.device, dtype=torch.float32),
            indexing="ij",
        )
        center_y, center_x = height // 2, width // 2
        radius = torch.sqrt((yy - center_y).square() + (xx - center_x).square())
        sigma = float(min(height // 2, width // 2)) * gaussian_scale / 2.0

        low_freq_mask = torch.exp(-0.5 * (radius / sigma).square())
        low_freq_mask = torch.fft.ifftshift(low_freq_mask)
        high_freq_mask = 1.0 - low_freq_mask[:, : width // 2 + 1]

        img_fft = torch.fft.rfft2(imgs_fp32, norm="ortho")
        high_fft = img_fft * high_freq_mask
        return torch.fft.irfft2(high_fft, s=(height, width), norm="ortho")


# Visualization and denormalization

def denormalize_image_tensor(tensor, mean=None, std=None):
    """Reverse z-score normalization on a torch tensor.

    Args:
        tensor: torch.Tensor of shape (..., C, H, W) or (C, H, W).
        mean:   list/array of length C. Defaults to IMAGE_MEAN.
        std:    list/array of length C. Defaults to IMAGE_STD.

    Returns:
        Denormalized tensor with same shape and device.
    """
    if mean is None:
        mean = IMAGE_MEAN
    if std is None:
        std = IMAGE_STD

    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Build broadcast shape aligned with channel dimension.
    # For (B, C, H, W) -> (1, C, 1, 1); for (C, H, W) -> (C, 1, 1).
    c_dim = tensor.ndim - 3  # channel dim index (0 for 3D, 1 for 4D)
    view_shape = [1] * tensor.ndim
    view_shape[c_dim] = -1
    mean_t = mean_t.view(view_shape)
    std_t = std_t.view(view_shape)

    return tensor * std_t + mean_t


def apply_rgb_vis_stretch(rgb, stretch=None):
    if stretch is None:
        stretch = RGB_VIS_STRETCH
    stretch = str(stretch).strip().lower()

    if stretch in ("none", "linear", "identity", "off"):
        return rgb
    if stretch == "sqrt":
        return torch.sqrt(torch.clamp(rgb, min=0.0))
    raise ValueError(f"Unsupported RGB visualization stretch: {stretch!r}")


def prepare_rgb_vis_tensor(
    imgs,
    vis_band_order=None,
    clamp_range=(0.0, 1.0),
    stretch=None,
    masks=None,
    mask_threshold=0.5,
    denormalize=True,
):
    """Prepare a model-space image tensor for RGB or RGBA visualization.

    Steps:
        1. Denormalize from z-score back to real pixel space.
        2. Select 3 display bands according to *vis_band_order*.
        3. Clamp to the specified range.
        4. Apply an optional display stretch such as sqrt.

    Args:
        imgs: torch.Tensor of shape (N, C, H, W) or (C, H, W).
        vis_band_order: list of 3 int channel indices. Defaults to VIS_BAND_ORDER.
        clamp_range: tuple (min, max) for clamping. Defaults to (0.0, 1.0).
        stretch: visualization stretch mode. Defaults to RGB_VIS_STRETCH from config.
        masks: optional foreground masks. When provided, mask-zero regions become
            transparent and the returned tensor has four RGBA channels.
        mask_threshold: mask values above this threshold are treated as foreground.
        denormalize: whether to reverse IMPGM z-score normalization before display.

    Returns:
        A three-channel RGB tensor when masks is None, otherwise a four-channel
        RGBA tensor. The batch shape and device follow the input.
    """
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    denormed = denormalize_image_tensor(imgs) if denormalize else imgs

    if denormed.ndim == 4:
        rgb = denormed[:, vis_band_order, :, :]
    elif denormed.ndim == 3:
        rgb = denormed[vis_band_order, :, :]
    else:
        raise ValueError(
            f"prepare_rgb_vis_tensor expects 3D or 4D tensor, got {denormed.ndim}D"
        )

    if clamp_range is not None:
        rgb = torch.clamp(rgb, clamp_range[0], clamp_range[1])

    rgb = apply_rgb_vis_stretch(rgb, stretch=stretch)

    if masks is not None:
        alpha = _prepare_foreground_masks(masks, rgb, threshold=mask_threshold)
        rgb = rgb * alpha
        channel_dim = 1 if rgb.ndim == 4 else 0
        rgb = torch.cat([rgb, alpha], dim=channel_dim)

    return rgb


def _prepare_foreground_masks(masks, reference, threshold=0.5):
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)
    masks = masks.to(device=reference.device)

    if reference.ndim == 4:
        if masks.ndim == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)
        elif masks.ndim == 3:
            if masks.shape[0] == reference.shape[0]:
                masks = masks.unsqueeze(1)
            else:
                masks = masks.unsqueeze(0)
        elif masks.ndim != 4:
            raise ValueError(f"Expected foreground masks with 2-4 dimensions, got {masks.ndim}.")
        if masks.shape[0] == 1 and reference.shape[0] > 1:
            masks = masks.expand(reference.shape[0], -1, -1, -1)
        if masks.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Foreground mask batch size {masks.shape[0]} does not match image batch size {reference.shape[0]}."
            )
    elif reference.ndim == 3:
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        elif masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks.squeeze(0)
        if masks.ndim != 3:
            raise ValueError(f"Expected foreground masks with 2-3 dimensions, got {masks.ndim}.")
    else:
        raise ValueError(f"Expected a 3D or 4D image tensor, got {reference.ndim}D.")

    if masks.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"Foreground mask size {tuple(masks.shape[-2:])} does not match image size {tuple(reference.shape[-2:])}."
        )
    if masks.shape[-3] != 1:
        raise ValueError(f"Foreground masks must have one channel, got {masks.shape[-3]}.")
    return (masks > float(threshold)).to(dtype=reference.dtype)


def prepare_mask_vis_tensor(msk, threshold=0.0):
    """Prepare a mask tensor for visualization.

    Args:
        msk: torch.Tensor of shape (H, W), (1, H, W), or (N, 1, H, W).
        threshold: values > threshold are considered foreground.

    Returns:
        torch.Tensor of shape (H, W) with values in {0.0, 1.0}.
    """
    if msk.ndim == 4:
        msk = msk[0]
    msk = msk.squeeze()
    if msk.ndim != 2:
        raise ValueError(
            f"prepare_mask_vis_tensor expects a 2D mask after squeezing, got shape {msk.shape}"
        )
    return (msk > threshold).float()


def prepare_high_freq_vis_tensor(imgs, vis_band_order=None, eps=1e-6):
    """Prepare a high-frequency residual tensor for visualization.

    Zero-centered symmetric scaling to [0, 1] for edge/texture display.
    This is NOT a physical pixel recovery; it is purely for visual inspection.

    Args:
        imgs: torch.Tensor of shape (N, C, H, W) or (C, H, W).
        vis_band_order: list of 3 int channel indices. Defaults to VIS_BAND_ORDER.
        eps: small constant to avoid division by zero.

    Returns:
        torch.Tensor of shape (..., 3, H, W) with values in [0, 1].
    """
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    if imgs.ndim == 4:
        rgb = imgs[:, vis_band_order, :, :]
    elif imgs.ndim == 3:
        rgb = imgs[vis_band_order, :, :]
    else:
        raise ValueError(
            f"prepare_high_freq_vis_tensor expects 3D or 4D tensor, got {imgs.ndim}D"
        )

    max_abs = torch.max(torch.abs(rgb))
    rgb = torch.clamp(rgb / (2 * max_abs + eps) + 0.5, 0.0, 1.0)
    return rgb


min_loss = float("inf")
best_epoch_idx = -1
patience_counter = 0
patience_counter_after_min_lr = 0


def _format_loss_for_log(value, precision: int = 6) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_format_loss_for_log(v, precision)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
        return left + ", ".join(_format_loss_for_log(v, precision) for v in value) + right
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return repr(value)


def get_training_monitor_state() -> Dict[str, float]:
    return {
        "min_loss": float(min_loss),
        "best_epoch_idx": int(best_epoch_idx),
        "patience_counter": int(patience_counter),
        "patience_counter_after_min_lr": int(patience_counter_after_min_lr),
    }


def save_model(
    ckpt_path,
    epoch,
    model=None,
    optimizer=None,
    scheduler=None,
    loss_dict=None,
    extra=None,
    verbose=True,
    scaler=None,
):
    """Atomically save model and training state using a completed epoch number."""
    ckpt_path = str(ckpt_path)
    save_dir = os.path.dirname(ckpt_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    extra = dict(extra or {})
    extra.setdefault("rng_state", stash_rng_state())

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict() if model is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "loss_dict": loss_dict if loss_dict is not None else None,
        "extra": extra,
    }

    temp_handle, temp_path = tempfile.mkstemp(
        prefix=f".{Path(ckpt_path).name}.",
        suffix=".tmp",
        dir=save_dir,
    )
    os.close(temp_handle)
    try:
        try:
            torch.save(checkpoint, temp_path)
        except (OSError, RuntimeError) as exc:
            temp_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            try:
                free_space = shutil.disk_usage(save_dir).free
                free_text = f"{free_space / (1024 ** 3):.2f} GiB"
            except OSError:
                free_text = "unknown"
            raise RuntimeError(
                f"Checkpoint write failed for {ckpt_path}. The temporary file reached "
                f"{temp_size / (1024 ** 3):.2f} GiB and the destination filesystem "
                f"reports {free_text} free. Atomic saving writes a complete temporary "
                "checkpoint before replacing the previous file; check disk space, "
                "container quota, and filesystem file-size limits."
            ) from exc

        os.replace(temp_path, ckpt_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if verbose:
        print(f"[Checkpoint] Saved at epoch {int(epoch)}")
    return ckpt_path


def _normalize_skip_prefix(skip_prefix):
    if skip_prefix is None:
        return None
    if isinstance(skip_prefix, str):
        return (skip_prefix,)
    if isinstance(skip_prefix, Iterable):
        return tuple(str(prefix) for prefix in skip_prefix)
    raise TypeError(f"Unsupported skip_prefix type: {type(skip_prefix)}")


def _load_state_dict_with_allowed_shape_mismatches(
    model,
    state_dict,
    *,
    strict,
    allowed_prefixes,
):
    """Load a state dict while skipping only explicitly allowed shape mismatches."""
    target_state = model.state_dict()
    shape_mismatches = {}
    for key, source_value in state_dict.items():
        target_value = target_state.get(key)
        if target_value is None:
            continue
        source_shape = tuple(source_value.shape)
        target_shape = tuple(target_value.shape)
        if source_shape != target_shape:
            shape_mismatches[key] = (source_shape, target_shape)

    disallowed = {
        key: shapes
        for key, shapes in shape_mismatches.items()
        if not key.startswith(allowed_prefixes)
    }
    if disallowed:
        details = "\n  - ".join(
            f"{key}: checkpoint={source_shape}, model={target_shape}"
            for key, (source_shape, target_shape) in sorted(disallowed.items())
        )
        raise RuntimeError(
            "Checkpoint contains shape mismatches outside the explicitly allowed "
            f"prefixes {allowed_prefixes}:\n  - {details}"
        )

    if not shape_mismatches:
        model.load_state_dict(state_dict, strict=strict)
        return

    active_skip_prefixes = tuple(
        prefix
        for prefix in allowed_prefixes
        if any(key.startswith(prefix) for key in shape_mismatches)
    )
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(active_skip_prefixes)
    }
    if strict:
        target_keys = set(target_state)
        filtered_keys = set(filtered_state)
        allowed_missing = {
            key for key in target_keys if key.startswith(active_skip_prefixes)
        }
        unexpected_keys = sorted(filtered_keys - target_keys)
        missing_keys = sorted((target_keys - filtered_keys) - allowed_missing)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Checkpoint remains structurally incompatible after skipping allowed "
                "shape mismatches. "
                f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}."
            )

    model.load_state_dict(filtered_state, strict=False)
    mismatch_details = ", ".join(
        f"{key} {source_shape}->{target_shape}"
        for key, (source_shape, target_shape) in sorted(shape_mismatches.items())
    )
    print(
        "[Checkpoint] Skipped channel-dependent module prefixes "
        f"{active_skip_prefixes} because of shape mismatches: {mismatch_details}"
    )


def _move_optimizer_state_to_device(optimizer, device):
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _safe_torch_load(save_path, map_location=None):
    try:
        return torch.load(save_path, weights_only=False, map_location=map_location)
    except Exception as exc:
        if "xpu" not in str(map_location).lower():
            raise
        print(
            f"[WARN] torch.load(map_location={map_location!r}) failed for {save_path}: {exc}. "
            "Retrying with map_location='cpu'."
        )
        return torch.load(save_path, weights_only=False, map_location="cpu")


def _restore_diffusers_ema_model_state(checkpoint, model, save_path):
    """Materialize model-form EMA weights from a compact Diffusers EMA state."""
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        return None

    extra = checkpoint.get("extra", {}) or {}
    if extra.get("model_state_is_ema", False):
        return model_state

    ema_training_state = extra.get("ema_training_state_dict")
    if not isinstance(ema_training_state, dict):
        return None
    shadow_params = ema_training_state.get("shadow_params")
    if shadow_params is None:
        return None

    named_parameters = list(model.named_parameters())
    if len(named_parameters) != len(shadow_params):
        raise RuntimeError(
            "Diffusers EMA parameter count does not match the target model while "
            f"loading {save_path}: {len(shadow_params)} != {len(named_parameters)}."
        )

    ema_state = model_state.copy()
    for (name, _), shadow_param in zip(named_parameters, shadow_params):
        ema_state[name] = shadow_param
    return ema_state


def _coerce_rng_state_tensor(state, target_device="cpu"):
    tensor = state.detach() if torch.is_tensor(state) else torch.as_tensor(state)
    return tensor.to(device=target_device, dtype=torch.uint8).contiguous()


def _coerce_rng_state_sequence(states, target_device="cpu"):
    if states is None:
        return None
    return [_coerce_rng_state_tensor(state, target_device=target_device) for state in states]


def load_model(
    save_path,
    model=None,
    optimizer=None,
    scheduler=None,
    resume_training=True,
    preload_only=False,
    skip_prefix=None,
    load_ema=False,
    scaler=None,
    strict=True,
    ema_model=None,
    ema_decay=None,
    map_location=None,
    allowed_shape_mismatch_prefixes=None,
    restore_rng=True,
):
    """Load a training checkpoint using the CSUA_LDM checkpoint contract.

    The return value is always ``(model, optimizer, scheduler,
    completed_epoch_num, loss_dict, extra, ema_decay)``.
    """
    if not resume_training and preload_only:
        raise ValueError("`preload_only=True` requires `resume_training=True`.")

    save_path = Path(save_path)
    if map_location is None:
        if model is not None:
            try:
                map_location = next(model.parameters()).device
            except StopIteration:
                try:
                    map_location = next(model.buffers()).device
                except StopIteration:
                    map_location = "cpu"
        else:
            map_location = "cpu"

    checkpoint = _safe_torch_load(save_path, map_location=map_location)
    epoch_num = int(checkpoint.get("epoch", -1))
    loss_dict = checkpoint.get("loss_dict", {})
    extra = checkpoint.get("extra", {}) or {}
    if (
        resume_training
        and not preload_only
        and restore_rng
        and extra.get("rng_state") is not None
    ):
        restore_rng_state(extra.get("rng_state"))

    ema_state_dict = extra.get("ema_model_state_dict")
    if load_ema and ema_state_dict is None and model is not None:
        ema_state_dict = _restore_diffusers_ema_model_state(
            checkpoint,
            model,
            save_path,
        )

    if model is not None:
        state_dict = checkpoint.get("model_state_dict")
        if load_ema:
            if ema_state_dict is None:
                raise KeyError(
                    "Neither `ema_model_state_dict` nor a recoverable compact "
                    f"Diffusers EMA state was found in checkpoint: {save_path}"
                )
            state_dict = ema_state_dict
        if state_dict is None:
            raise KeyError(f"`model_state_dict` was not found in checkpoint: {save_path}")

        prefixes = _normalize_skip_prefix(skip_prefix)
        allowed_mismatch_prefixes = _normalize_skip_prefix(
            allowed_shape_mismatch_prefixes
        )
        if prefixes is not None and allowed_mismatch_prefixes is not None:
            raise ValueError(
                "skip_prefix and allowed_shape_mismatch_prefixes cannot be used "
                "together."
            )
        if prefixes is None:
            if allowed_mismatch_prefixes is None:
                model.load_state_dict(state_dict, strict=strict)
            else:
                _load_state_dict_with_allowed_shape_mismatches(
                    model,
                    state_dict,
                    strict=strict,
                    allowed_prefixes=allowed_mismatch_prefixes,
                )
        else:
            filtered_state = {k: v for k, v in state_dict.items() if not k.startswith(prefixes)}
            model.load_state_dict(filtered_state, strict=False)

    if optimizer is not None:
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is None:
            raise KeyError(f"`optimizer_state_dict` was not found in checkpoint: {save_path}")
        optimizer.load_state_dict(opt_state)
        _move_optimizer_state_to_device(optimizer, DEVICE)

    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            raise KeyError(f"`scheduler_state_dict` was not found in checkpoint: {save_path}")
        scheduler.load_state_dict(scheduler_state)

    if scaler is not None:
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is None:
            if resume_training and not preload_only:
                print("[Checkpoint] scaler_state_dict missing; using fresh GradScaler state.")
        else:
            scaler.load_state_dict(scaler_state)

    if ema_model is not None:
        if ema_state_dict is not None:
            ema_model.load_state_dict(ema_state_dict, strict=bool(strict))
            if ema_decay is not None:
                ema_decay = float(extra.get("ema_decay", ema_decay))
            print(f"Loaded EMA state from checkpoint. ema_decay={ema_decay}")
        elif model is not None:
            ema_model.load_state_dict(model.state_dict(), strict=bool(strict))
            print("EMA state not found in checkpoint; initialized EMA from current model.")
        else:
            raise ValueError("`model` is required when an EMA state is unavailable.")

    if resume_training and not preload_only:
        formatted_loss = _format_loss_for_log(loss_dict, precision=6)
        print(f"Resume after completed epoch: [{epoch_num}] | loss: {formatted_loss}")
    elif resume_training and preload_only:
        print(f"Preloading weights from: {save_path.stem}")

    return model, optimizer, scheduler, epoch_num, loss_dict, extra, ema_decay


def load_model_for_eval(save_path, model, map_location=None):
    model, _, _, _, _, extra, _ = load_model(
        save_path,
        model=model,
        optimizer=None,
        scheduler=None,
        resume_training=False,
        preload_only=False,
        load_ema=True,
        map_location=map_location,
    )
    return model, extra


def load_controlnet_model_for_eval(
    save_path,
    model,
    branch_attr="ControlNet_model",
    map_location=None,
    skip_prefix=None,
    strict=True,
):
    """Load branch-only ControlNet EMA weights without replacing the backbone."""
    save_path = Path(save_path)
    if map_location is None:
        try:
            map_location = next(model.parameters()).device
        except StopIteration:
            map_location = "cpu"

    checkpoint = _safe_torch_load(save_path, map_location=map_location)
    extra = checkpoint.get("extra", {}) or {}
    ema_state_dict = extra.get("controlnet_ema_state_dict")
    if ema_state_dict is None:
        raise KeyError(
            f"`controlnet_ema_state_dict` was not found in checkpoint: {save_path}"
        )

    controlnet_branch = getattr(model, branch_attr, None)
    if controlnet_branch is None:
        raise AttributeError(
            f"Model {type(model).__name__!r} has no ControlNet branch {branch_attr!r}."
        )
    prefixes = _normalize_skip_prefix(skip_prefix)
    if prefixes is None:
        controlnet_branch.load_state_dict(ema_state_dict, strict=bool(strict))
    else:
        filtered_state = {
            key: value
            for key, value in ema_state_dict.items()
            if not key.startswith(prefixes)
        }
        controlnet_branch.load_state_dict(filtered_state, strict=False)
    model.eval()
    return model, extra


# VAE and latent scaling

def normalize_latent_scaling_factor(latent_scaling_factor=LATENT_SCALING_FACTOR):
    if latent_scaling_factor is None:
        raise ValueError(
            f"latent.{LATENT_SCALING_FACTOR_FIELD} is not calibrated. Run "
            "VAE/VAE_Code/IMPGM_Compute_latent_scaling_factor.py first."
        )
    if isinstance(latent_scaling_factor, torch.Tensor):
        if latent_scaling_factor.numel() != 1:
            raise ValueError(
                f"latent scaling factor must be scalar, got {tuple(latent_scaling_factor.shape)}."
            )
        value = float(latent_scaling_factor.detach().view(()).item())
    else:
        value = float(latent_scaling_factor)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid latent scaling factor: {value}")
    return value


def scale_latent(latent, latent_scaling_factor=LATENT_SCALING_FACTOR):
    scale = torch.as_tensor(
        normalize_latent_scaling_factor(latent_scaling_factor),
        device=latent.device,
        dtype=latent.dtype,
    )
    return latent * scale


def descale_latent(latent, latent_scaling_factor=LATENT_SCALING_FACTOR):
    scale = torch.as_tensor(
        normalize_latent_scaling_factor(latent_scaling_factor),
        device=latent.device,
        dtype=latent.dtype,
    )
    return latent / scale


def build_standard_vae(img_channel=INPUT_CHANNELS, device=DEVICE):
    # Keep VAE architecture imports lazy for callers that only need generic utilities.
    from VAE.VAE_Code.VAE_model import AutoEncoder

    return AutoEncoder(
        img_channel=img_channel,
        down_channels=VAE_DOWN_CHANNELS,
        mid_inout_channels=VAE_MID_CHANNELS,
        num_down_layers=VAE_NUM_DOWN_LAYERS,
        num_mid_layers=VAE_NUM_MID_LAYERS,
        num_up_layers=VAE_NUM_UP_LAYERS,
        z_channel=VAE_Z_CHANNEL,
        norm_channels=VAE_NORM_CHANNELS,
    ).to(device)


@torch.no_grad()
def load_standard_vae(save_path=VAE_MODEL_SAVEPATH, device=DEVICE, load_ema=True):
    vae_model = build_standard_vae(img_channel=INPUT_CHANNELS, device=device)
    if load_ema:
        vae_model, _ = load_model_for_eval(save_path, vae_model, map_location=device)
    else:
        vae_model, _, _, _, _, _, _ = load_model(
            save_path,
            model=vae_model,
            optimizer=None,
            resume_training=False,
            preload_only=False,
            load_ema=False,
            map_location=device,
        )
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False
    return vae_model, normalize_latent_scaling_factor(), str(save_path)


@torch.no_grad()
def encode_to_scaled_latent(vae_model, imgs, latent_scaling_factor=LATENT_SCALING_FACTOR):
    latent_params = vae_model.encode(imgs)
    latent, _, _ = vae_model.reparameterize(latent_params)
    return scale_latent(latent, latent_scaling_factor)


@torch.no_grad()
def decode_from_scaled_latent(vae_model, latent, latent_scaling_factor=LATENT_SCALING_FACTOR):
    return vae_model.decode(descale_latent(latent, latent_scaling_factor))


@torch.no_grad()
def build_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()

    for key, ema_value in ema_state.items():
        model_value = model_state[key]
        if not torch.is_floating_point(ema_value):
            ema_value.copy_(model_value)
        else:
            ema_value.mul_(float(decay)).add_(model_value, alpha=1.0 - float(decay))


def set_random_seed(seed, deterministic=False):
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_type = torch.device(DEVICE).type
    if device_type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if device_type == "xpu" and _xpu_state_api_available("manual_seed"):
        torch.xpu.manual_seed(seed)
    if device_type == "xpu" and _xpu_state_api_available("manual_seed_all"):
        torch.xpu.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(
                bool(deterministic),
                warn_only=bool(deterministic),
            )
        except TypeError:
            torch.use_deterministic_algorithms(bool(deterministic))
    return seed


def stash_rng_state():
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    device_type = torch.device(DEVICE).type
    if device_type == "cuda":
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if device_type == "xpu" and _xpu_state_api_available("get_rng_state_all"):
        rng_state["torch_xpu"] = torch.xpu.get_rng_state_all()
    if hasattr(torch.backends, "cudnn"):
        rng_state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        rng_state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        rng_state["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        rng_state["deterministic_algorithms_warn_only"] = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
    return rng_state


def restore_rng_state(rng_state):
    """Strictly restore previously captured random number generator states."""
    if rng_state is None:
        return
    device_type = torch.device(DEVICE).type
    saved_accelerators = {
        name
        for name, key in (("cuda", "torch_cuda"), ("xpu", "torch_xpu"))
        if key in rng_state
    }
    if saved_accelerators and saved_accelerators != {device_type}:
        raise RuntimeError(
            "Checkpoint RNG state was captured for "
            f"{sorted(saved_accelerators)}, but the current device is {device_type!r}. "
            "Pass restore_rng=False to load_model for an intentional cross-device resume."
        )
    if device_type in {"cuda", "xpu"} and f"torch_{device_type}" not in rng_state:
        raise RuntimeError(
            f"Checkpoint does not contain the {device_type.upper()} RNG state required "
            "for an exact resume. Pass restore_rng=False for an intentional "
            "cross-device resume."
        )
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch_cpu" in rng_state:
        torch.set_rng_state(
            _coerce_rng_state_tensor(rng_state["torch_cpu"], target_device="cpu")
        )
    if device_type == "cuda" and "torch_cuda" in rng_state:
        cuda_states = _coerce_rng_state_sequence(
            rng_state["torch_cuda"],
            target_device="cpu",
        )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG state count does not match the visible CUDA "
                f"device count: {len(cuda_states)} vs {torch.cuda.device_count()}. "
                "Pass restore_rng=False for an intentional topology change."
            )
        torch.cuda.set_rng_state_all(cuda_states)
    if device_type == "xpu" and "torch_xpu" in rng_state:
        if not _xpu_state_api_available("set_rng_state_all"):
            raise RuntimeError("This PyTorch build cannot restore XPU RNG states.")
        torch.xpu.set_rng_state_all(
            _coerce_rng_state_sequence(rng_state["torch_xpu"], target_device="cpu")
        )
    if hasattr(torch.backends, "cudnn"):
        if "cudnn_deterministic" in rng_state:
            torch.backends.cudnn.deterministic = bool(rng_state["cudnn_deterministic"])
        if "cudnn_benchmark" in rng_state:
            torch.backends.cudnn.benchmark = bool(rng_state["cudnn_benchmark"])
    if hasattr(torch, "use_deterministic_algorithms") and "deterministic_algorithms_enabled" in rng_state:
        enabled = bool(rng_state["deterministic_algorithms_enabled"])
        warn_only = bool(rng_state.get("deterministic_algorithms_warn_only", False))
        try:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(enabled)


def seed_dataloader_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader_generator(seed):
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    return generator


def build_torch_generator(seed, device):
    """Build a seeded random generator compatible with the target device."""
    if seed is None:
        return None
    device = torch.device(device)
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"Failed to build a seeded generator for device {device} with "
            f"PyTorch {torch.__version__}."
        ) from exc
    return generator


def randn(shape, device, dtype=torch.float32, generator=None):
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def randn_like(tensor, generator=None):
    return torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator,
    )


def training_monitor(epoch_idx, min_lr, current_lr, current_loss, patience_threshold, resume_train=False, load_dict=None):

    global min_loss, best_epoch_idx, patience_counter, patience_counter_after_min_lr

    if resume_train:
        if load_dict is None:
            print("Resume training is enabled, but load_dict is None.")
            min_loss = float('inf')
            best_epoch_idx = epoch_idx
            patience_counter = 0
            patience_counter_after_min_lr = 0
            return False, False, patience_counter, patience_counter_after_min_lr

        min_loss = float(load_dict.get('min_loss', float('inf')))
        best_epoch_idx = int(load_dict.get('best_epoch_idx', epoch_idx))
        patience_counter = int(load_dict.get('patience_counter', 0))
        patience_counter_after_min_lr = int(load_dict.get('patience_counter_after_min_lr', 0))
        return False, False, patience_counter, patience_counter_after_min_lr

    if epoch_idx == 0:
        min_loss = current_loss
        best_epoch_idx = epoch_idx
        patience_counter = 0
        patience_counter_after_min_lr = 0
        return False, True, patience_counter, patience_counter_after_min_lr

    if current_lr > min_lr:
        if current_loss < min_loss:
            min_loss = current_loss
            best_epoch_idx = epoch_idx
            patience_counter = 0
            patience_counter_after_min_lr = 0
            return False, True, patience_counter, patience_counter_after_min_lr

        patience_counter += 1
        return False, False, patience_counter, patience_counter_after_min_lr

    remaining_epochs = patience_threshold - patience_counter_after_min_lr
    print(f"-- Notice -- learning rate has reached the minimum value {min_lr:.8f}; continue observing for {remaining_epochs} epoch(s).")
    if current_loss < min_loss:
        min_loss = current_loss
        best_epoch_idx = epoch_idx
        patience_counter = 0
        patience_counter_after_min_lr = 0
        return False, True, patience_counter, patience_counter_after_min_lr

    patience_counter += 1
    patience_counter_after_min_lr += 1
    if patience_threshold < patience_counter_after_min_lr:
        print(f"-- Notice -- no further improvement was observed for {patience_threshold} epoch(s) after reaching min LR, stopping training.")
        return True, False, patience_counter, patience_counter_after_min_lr
    return False, False, patience_counter, patience_counter_after_min_lr


def save_rgb_datas(
    imgs,
    nrow,
    savepath=None,
    format=None,
    is_showminmax=False,
    is_makegrid=True,
    prompt_strs=None,
):

    if is_showminmax:
        print(f'RGB_min: {torch.min(imgs)} --- RGB_max: {torch.max(imgs)}')

    # Contract: imgs must already be denormalized and band-selected as RGB or RGBA.
    # We only clamp here for safe uint8 conversion.
    imgs = torch.clamp(imgs, 0.0, 1.0)
    if imgs.shape[-3] == 4:
        format = 'PNG'
        if savepath is not None:
            savepath = str(Path(savepath).with_suffix('.png'))

    if is_makegrid:
        img_grid = make_grid(imgs, nrow=nrow)
        img_grid = img_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        if img_grid.shape[-1] not in (3, 4):
            raise ValueError(f"RGB snapshot tensors must have 3 or 4 channels, got {img_grid.shape[-1]}.")

        im = Image.fromarray(img_grid)
        if savepath is not None:
            im.save(savepath, format=format)

        return img_grid

    imgs = imgs.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    if imgs.shape[-1] not in (3, 4):
        raise ValueError(f"RGB snapshot tensors must have 3 or 4 channels, got {imgs.shape[-1]}.")
    if savepath is not None:
        if len(imgs) == 1:
            im = Image.fromarray(imgs[0])
            im.save(savepath, format=format)
        else:
            for idx, img in enumerate(imgs):
                im = Image.fromarray(img)
                save_path_obj = Path(savepath)
                sample_suffix = f'_p{idx + 1}'
                if prompt_strs is not None:
                    prompt_str = prompt_strs[idx]
                    sample_suffix += f'_{prompt_str}'
                savepath_separate = str(
                    save_path_obj.with_name(f"{save_path_obj.stem}{sample_suffix}{save_path_obj.suffix}")
                )
                im.save(savepath_separate, format=format)

    return imgs


def _validate_projection_wkts(projections, batch_size):
    """Validate non-empty projection WKT strings without requiring EPSG codes."""
    if len(projections) != batch_size:
        raise ValueError(
            "The number of projections must match the batch size, "
            f"got {len(projections)} and {batch_size}."
        )

    normalized_wkts = []
    for idx, projection in enumerate(projections):
        if not isinstance(projection, str) or not projection.strip():
            raise ValueError(f"Projection WKT at batch index {idx} is empty.")
        spatial_reference = osr.SpatialReference()
        if spatial_reference.ImportFromWkt(projection) != 0:
            raise ValueError(
                f"Projection WKT at batch index {idx} is invalid."
            )
        normalized_wkts.append(spatial_reference.ExportToWkt())
    return normalized_wkts


def save_msk_datas(
    msks,
    savepath=None,
    is_showminmax=False,
    is_makegrid=True,
    nrow=3,
    projections=None,
    geotransforms=None,
):
    """Save mask tensors as preview images or georeferenced TIF files."""

    if not is_makegrid:
        nrow = None

    # Treat empty/whitespace-only projection strings as missing so that
    # non-georeferenced inputs fall back to placeholder georeference instead
    # of raising inside _validate_projection_wkts.
    if projections is not None:
        projections = [
            p.strip() if isinstance(p, str) and p.strip() else None
            for p in projections
        ]
        if any(p is None for p in projections):
            projections = None

    if is_showminmax:
        print(f'min: {torch.min(msks)} --- max: {torch.max(msks)}')

    if is_makegrid:
        assert nrow is not None, "nrow must be provided when is_makegrid=True."
        msk_grid = make_grid(msks, nrow=nrow)
        msk_grid = msk_grid.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        msk_grid = Image.fromarray(msk_grid)
        if savepath is not None:
            msk_grid.save(savepath, format=None)

        return msk_grid

    if geotransforms is None:
        geotransforms = []
        for _ in range(msks.shape[0]):
            top_left_lon = float(random.randint(0, 900000))
            top_left_lat = float(random.randint(0, 9000000))
            pixel_width = 10.0
            pixel_height = -10.0
            geotransforms.append([top_left_lon, pixel_width, 0.0, top_left_lat, 0.0, pixel_height])
    elif len(geotransforms) != msks.shape[0]:
        raise ValueError(
            "The number of geotransforms must match the mask batch size, "
            f"got {len(geotransforms)} and {msks.shape[0]}."
        )

    projection_wkts = None
    if projections is None:
        epsg_codes = []
        for _ in range(msks.shape[0]):
            num_1 = [32600, 32700][random.randint(0, 1)]
            num_2 = random.randint(1, 60)
            epsg_code = num_1 + num_2
            epsg_codes.append(epsg_code)
    else:
        epsg_codes = [None] * msks.shape[0]
        projection_wkts = _validate_projection_wkts(
            projections,
            msks.shape[0],
        )

    msks = msks.squeeze(1)
    msks = msks.to("cpu").numpy()

    if len(msks) == 1:
        msk, epsg_code, geotransform = msks[0], epsg_codes[0], geotransforms[0]
        top_left_lon, top_left_lat, pixel_width, pixel_height = geotransform[0], geotransform[3], geotransform[1], geotransform[5]
        projection_wkt = None if projection_wkts is None else projection_wkts[0]
        Tif_Read_and_Write().Numpy_to_Tif(
            msk,
            savepath,
            top_left_lon,
            top_left_lat,
            pixel_width,
            pixel_height,
            epsg_code=epsg_code,
            prj_info=projection_wkt,
            nodata_value=np.nan,
        )
    else:
        for idx, (msk, epsg_code, geotransform) in enumerate(zip(msks, epsg_codes, geotransforms)):
            savepath_separate = savepath.replace('.tif', f'_p{idx + 1}.tif')
            top_left_lon, top_left_lat, pixel_width, pixel_height = geotransform[0], geotransform[3], geotransform[1], geotransform[5]
            projection_wkt = (
                None if projection_wkts is None else projection_wkts[idx]
            )
            Tif_Read_and_Write().Numpy_to_Tif(
                msk,
                savepath_separate,
                top_left_lon,
                top_left_lat,
                pixel_width,
                pixel_height,
                epsg_code=epsg_code,
                prj_info=projection_wkt,
                nodata_value=np.nan,
            )

    if projection_wkts is None:
        projections = []
        for epsg_code in epsg_codes:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg_code)
            projection = srs.ExportToWkt()
            projections.append(projection)
    else:
        projections = projection_wkts

    return msks, projections, geotransforms


def save_tif_datas(
    datas,
    projections=None,
    geotransforms=None,
    savepath=None,
    is_showminmax=False,
    masks=None,
    nodata_value=np.nan,
    mask_threshold=0.5,
    denormalize=True,
    prompt_strs=None,
):
    """Save image tensors as georeferenced TIF files.

    Set ``denormalize=False`` for tensors already stored in real pixel space.
    """
    # Treat empty/whitespace-only projection strings as missing so that
    # non-georeferenced inputs fall back to placeholder georeference instead
    # of raising inside _validate_projection_wkts.
    if projections is not None:
        projections = [
            p.strip() if isinstance(p, str) and p.strip() else None
            for p in projections
        ]
        if any(p is None for p in projections):
            projections = None

    if is_showminmax:
        print(f'TIF_min: {torch.min(datas)} --- TIF_max: {torch.max(datas)}')

    if denormalize:
        datas = denormalize_image_tensor(datas)
    if masks is not None:
        foreground_masks = _prepare_foreground_masks(masks, datas, threshold=mask_threshold).bool()
        datas = datas.masked_fill(~foreground_masks, float(nodata_value))
    datas = datas.to("cpu").numpy()

    # If either geotransforms or projections is missing, use full placeholder
    # georef to avoid misleading "half-real, half-placeholder" mixtures.
    use_placeholder = (geotransforms is None) or (projections is None)

    projection_wkts = None
    if use_placeholder:
        geotransforms = [
            [
                PLACEHOLDER_TOP_LEFT_X,
                PLACEHOLDER_PIXEL_WIDTH,
                0.0,
                PLACEHOLDER_TOP_LEFT_Y,
                0.0,
                PLACEHOLDER_PIXEL_HEIGHT,
            ]
        ] * datas.shape[0]
        epsg_codes = [PLACEHOLDER_EPSG] * datas.shape[0]
    else:
        if len(geotransforms) != datas.shape[0]:
            raise ValueError(
                "The number of geotransforms must match the image batch size, "
                f"got {len(geotransforms)} and {datas.shape[0]}."
            )
        epsg_codes = [None] * datas.shape[0]
        projection_wkts = _validate_projection_wkts(
            projections,
            datas.shape[0],
        )

    if len(datas) == 1:
        data, epsg_code, geotransform = datas[0], epsg_codes[0], geotransforms[0]
        top_left_lon, top_left_lat, pixel_width, pixel_height = geotransform[0], geotransform[3], geotransform[1], geotransform[5]
        projection_wkt = None if projection_wkts is None else projection_wkts[0]
        Tif_Read_and_Write().Numpy_to_Tif(
            data,
            savepath,
            top_left_lon,
            top_left_lat,
            pixel_width,
            pixel_height,
            epsg_code=epsg_code,
            prj_info=projection_wkt,
            nodata_value=nodata_value,
        )
    else:
        for idx, (data, epsg_code, geotransform) in enumerate(zip(datas, epsg_codes, geotransforms)):
            if prompt_strs is not None:
                prompt_str = prompt_strs[idx]
                savepath_separate = savepath.replace('.tif', f'_p{idx + 1}_{prompt_str}.tif')
            else:
                savepath_separate = savepath.replace('.tif', f'_p{idx + 1}.tif')
            top_left_lon, top_left_lat, pixel_width, pixel_height = geotransform[0], geotransform[3], geotransform[1], geotransform[5]
            projection_wkt = (
                None if projection_wkts is None else projection_wkts[idx]
            )
            Tif_Read_and_Write().Numpy_to_Tif(
                data,
                savepath_separate,
                top_left_lon,
                top_left_lat,
                pixel_width,
                pixel_height,
                epsg_code=epsg_code,
                prj_info=projection_wkt,
                nodata_value=nodata_value,
            )

    if projection_wkts is None:
        projections = []
        for epsg_code in epsg_codes:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg_code)
            projection = srs.ExportToWkt()
            projections.append(projection)
    else:
        projections = projection_wkts

    return datas, projections, geotransforms


def log_info(log_path, text):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open(mode='a', encoding='utf-8') as file_handle:
        file_handle.write(text + '\n')


def tensor_dilate(input_tensor, kernel_size=7, iter=1):

    x = input_tensor.detach().cpu().numpy()

    out_list = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for sample in x:
        m0 = (sample[0] >= 0.5).astype(np.uint8)
        m1 = cv2.dilate(m0, kernel=kernel, iterations=iter)
        out_list.append(m1)

    out = np.stack(out_list, axis=0)
    out = torch.from_numpy(out).to(torch.float32).to(input_tensor.device)
    out = out.unsqueeze(1)
    return out
