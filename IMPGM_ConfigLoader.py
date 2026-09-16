"""Unified YAML configuration loader for IMPGM.

Provides backward-compatible loading: training scripts still import from
IMPGM_Config.py, but the underlying values can come from YAML files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


# Internal helpers

def _flatten_dict(nested: dict, prefix: str = "", sep: str = ".") -> dict:
    items: dict[str, object] = {}
    for key, value in nested.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value
    return items


def _to_namespace(flat: dict, sep: str = ".") -> SimpleNamespace:
    root: dict[str, object] = {}
    for key, value in flat.items():
        parts = key.split(sep)
        node = root
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]  # type: ignore[assignment]
        node[parts[-1]] = value

    def _build(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _build(v) for k, v in obj.items()})
        return obj

    return _build(root)


def _namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {key: _namespace_to_dict(value) for key, value in vars(obj).items()}
    return obj


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_nested(obj: Any, path: str, default: Any = None, sep: str = "."):
    node = obj
    for part in path.split(sep):
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, SimpleNamespace):
            node = getattr(node, part, None)
        else:
            return default
        if node is None:
            return default
    return node


# Public API: load & merge

def load_yaml_config(yaml_path: str | Path) -> SimpleNamespace:
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}

    flat = _flatten_dict(raw)
    return _to_namespace(flat)


def merge_configs(*configs: SimpleNamespace) -> SimpleNamespace:
    merged_dict: dict[str, object] = {}
    for cfg in configs:
        merged_dict = _deep_merge_dicts(merged_dict, _namespace_to_dict(cfg))
    return _to_namespace(_flatten_dict(merged_dict))


# Validation

class ConfigValidationError(Exception):
    """Raised when a config fails structural validation."""
    pass


def _as_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _as_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_as_dict(v) for v in obj]
    return obj


def validate_config(cfg: SimpleNamespace, strict: bool = False) -> list[str]:
    """Validate a merged config and return a list of warnings/errors.

    When *strict* is True, warnings are also treated as fatal errors.
    """
    errors: list[str] = []
    warnings: list[str] = []
    d = _as_dict(cfg)

    # Geometry and normalization
    input_channels = _get_nested(d, "geometry.input_channels")
    img_mean = _get_nested(d, "normalization.image_mean")
    img_std = _get_nested(d, "normalization.image_std")

    if img_mean is None:
        errors.append("normalization.image_mean is required but missing")
    if img_std is None:
        errors.append("normalization.image_std is required but missing")

    if input_channels is not None and img_mean is not None:
        if len(img_mean) != input_channels:
            errors.append(
                f"normalization.image_mean length ({len(img_mean)}) != "
                f"geometry.input_channels ({input_channels})"
            )
    if input_channels is not None and img_std is not None:
        if len(img_std) != input_channels:
            errors.append(
                f"normalization.image_std length ({len(img_std)}) != "
                f"geometry.input_channels ({input_channels})"
            )
        elif any(not isinstance(v, (int, float)) or v <= 0 for v in img_std):
            errors.append(
                f"normalization.image_std must contain strictly positive numbers, got {img_std}"
            )

    # Visualization bands
    vis_band_order = _get_nested(d, "visualization.vis_band_order")
    if vis_band_order is not None:
        if not isinstance(vis_band_order, list):
            errors.append("visualization.vis_band_order must be a list")
        elif len(vis_band_order) != 3:
            errors.append(
                f"visualization.vis_band_order must have exactly 3 elements, got {vis_band_order}"
            )
        elif not all(isinstance(v, int) for v in vis_band_order):
            errors.append(
                f"visualization.vis_band_order must contain integers, got {vis_band_order}"
            )
        elif input_channels is not None and not all(0 <= v < input_channels for v in vis_band_order):
            errors.append(
                f"visualization.vis_band_order indices must be in [0, {input_channels - 1}], got {vis_band_order}"
            )
        elif len(set(vis_band_order)) != 3:
            errors.append(
                f"visualization.vis_band_order must not contain duplicates, got {vis_band_order}"
            )

    # Dataset roots
    # Type-check image_roots/mask_roots for the train, test and draw splits.
    # Non-existent paths only produce warnings, and only for the train split.
    # The mandatory valid split is not covered here; it is enforced separately
    # by IMPGM_Config._require_legacy_valid_split when DATASET_DICT is built.
    for split in ("train", "test", "draw"):
        img_roots = _get_nested(d, f"dataset.splits.{split}.image_roots")
        if img_roots is not None:
            if not isinstance(img_roots, list):
                errors.append(f"dataset.splits.{split}.image_roots must be a list")
            else:
                for p in img_roots:
                    if not isinstance(p, str):
                        errors.append(
                            f"dataset.splits.{split}.image_roots contains non-string: {p!r}"
                        )
                    elif split == "train" and not Path(p).exists():
                        warnings.append(
                            f"dataset.splits.{split}.image_roots path does not exist: {p}"
                        )
        mask_roots = _get_nested(d, f"dataset.splits.{split}.mask_roots")
        if mask_roots is not None:
            if not isinstance(mask_roots, list):
                errors.append(f"dataset.splits.{split}.mask_roots must be a list")
            else:
                for p in mask_roots:
                    if not isinstance(p, str):
                        errors.append(
                            f"dataset.splits.{split}.mask_roots contains non-string: {p!r}"
                        )
                    elif split == "train" and not Path(p).exists():
                        warnings.append(
                            f"dataset.splits.{split}.mask_roots path does not exist: {p}"
                        )

    # Numeric values
    latent_sf = _get_nested(d, "latent.scaling_factor")
    if latent_sf is not None and (not isinstance(latent_sf, (int, float)) or latent_sf <= 0):
        errors.append(f"latent.scaling_factor must be a positive number, got {latent_sf}")

    latent_sf_transfer = _get_nested(d, "latent.scaling_factor_transfer")
    if latent_sf_transfer is not None and (
        not isinstance(latent_sf_transfer, (int, float))
        or latent_sf_transfer <= 0
    ):
        errors.append(
            "latent.scaling_factor_transfer must be null or a positive number, "
            f"got {latent_sf_transfer}"
        )

    steps = _get_nested(d, "diffusion.sampling.steps")
    if steps is not None and (not isinstance(steps, int) or steps <= 0):
        errors.append(f"diffusion.sampling.steps must be a positive integer, got {steps}")

    img_size = _get_nested(d, "geometry.image_size")
    if img_size is not None and (not isinstance(img_size, int) or img_size <= 0):
        errors.append(f"geometry.image_size must be a positive integer, got {img_size}")

    latent_res = _get_nested(d, "geometry.latent_resolution")
    if latent_res is not None and (not isinstance(latent_res, int) or latent_res <= 0):
        errors.append(f"geometry.latent_resolution must be a positive integer, got {latent_res}")

    for key in (
        "diffusion.unet.in_channels",
        "diffusion.unet.out_channels",
        "controlnet.conditional_channels",
    ):
        val = _get_nested(d, key)
        if val is not None and (not isinstance(val, int) or val <= 0):
            errors.append(f"{key} must be a positive integer, got {val}")

    # Prompt mapping
    prompt_map = _get_nested(d, "dataset.prompt_map")
    if prompt_map is not None:
        if not isinstance(prompt_map, dict):
            errors.append("dataset.prompt_map must be a dict")
        else:
            vals = list(prompt_map.values())
            if vals != list(range(len(vals))):
                warnings.append(
                    f"dataset.prompt_map values {vals} are not contiguous 0..{len(vals)-1}"
                )

    # Strict mode treats warnings as errors.
    if strict:
        errors.extend(warnings)
    else:
        for w in warnings:
            print(f"[Config Warning] {w}")

    if errors:
        raise ConfigValidationError("\n".join(errors))

    return warnings
