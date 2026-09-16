"""Evaluate traceable IMPGM large-image generation results.

The evaluator covers four complementary aspects:

* raster integrity and georeferencing;
* generated/reference seam continuity at internal tile transitions;
* condition consistency with 256 x 256 sliding-window inference;
* local 256 x 256 distribution and spectral quality.

Large images are never resized for condition evaluation.  The frozen evaluator
is applied at its training crop size and overlapping logits are blended before
global IoU and Dice are computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


SCHEMA_VERSION = 3
DEFAULT_GENERATION_MANIFEST = "generation_manifest.jsonl"
DEFAULT_GENERATION_CONFIG = "generation_config.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path):
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}."
                ) from exc
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _gdal_modules():
    try:
        from osgeo import gdal, osr
    except ImportError as exc:
        raise RuntimeError("GDAL is required for large-image evaluation.") from exc
    gdal.UseExceptions()
    return gdal, osr


def _read_raster(path, *, read_array=True):
    gdal, _ = _gdal_modules()
    path = Path(path)
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Unable to open raster: {path}")
    metadata = {
        "path": str(path.resolve()),
        "width": int(dataset.RasterXSize),
        "height": int(dataset.RasterYSize),
        "bands": int(dataset.RasterCount),
        "projection_wkt": dataset.GetProjection() or "",
        "geotransform": [float(value) for value in dataset.GetGeoTransform()],
    }
    array = dataset.ReadAsArray() if read_array else None
    dataset = None
    if read_array:
        if array is None:
            raise RuntimeError(f"GDAL failed to read raster: {path}")
        array = np.asarray(array)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(f"Expected raster [C,H,W], got {array.shape} in {path}.")
    return array, metadata


def _projection_is_same(left_wkt, right_wkt):
    left_wkt = str(left_wkt or "").strip()
    right_wkt = str(right_wkt or "").strip()
    if not left_wkt and not right_wkt:
        return True
    if not left_wkt or not right_wkt:
        return False
    if left_wkt == right_wkt:
        return True
    _, osr = _gdal_modules()
    left = osr.SpatialReference()
    right = osr.SpatialReference()
    if left.ImportFromWkt(left_wkt) != 0 or right.ImportFromWkt(right_wkt) != 0:
        return False
    return bool(left.IsSame(right))


def _extent_corners(geotransform, width, height):
    gt = [float(value) for value in geotransform]
    corners = []
    for col, row in ((0, 0), (width, 0), (0, height), (width, height)):
        corners.append(
            [
                gt[0] + col * gt[1] + row * gt[2],
                gt[3] + col * gt[4] + row * gt[5],
            ]
        )
    return corners


def _finite_ratio(array):
    if array.size == 0:
        return 0.0
    return float(np.isfinite(array).sum() / array.size)


def _resolve_record_path(record, field_name, generation_root):
    value = record.get(field_name)
    if not value:
        raise KeyError(f"Generation record does not provide {field_name!r}.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = generation_root / path
    if path.is_file():
        return path.resolve()

    # Manifests can be copied from the training server together with the
    # generation root. Preserve the recorded path below its `scenes` segment.
    if path.is_absolute() and "scenes" in path.parts:
        scenes_index = path.parts.index("scenes")
        relocated = generation_root.joinpath(*path.parts[scenes_index:])
        if relocated.is_file():
            return relocated.resolve()

    raise FileNotFoundError(f"Missing {field_name}: {value}")


def _load_visible_band_order(inputs, scene_ids):
    orders = {}
    for scene_id in scene_ids:
        record = inputs["record_by_id"][scene_id]
        scene_config_path = _resolve_record_path(
            record,
            "scene_config",
            inputs["generation_root"],
        )
        scene_config = _load_json(scene_config_path)
        try:
            order = scene_config["generation"]["normalization"]["visible_band_order"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "Scene config does not provide "
                f"generation.normalization.visible_band_order: {scene_config_path}"
            ) from exc
        if (
            not isinstance(order, list)
            or len(order) != 3
            or any(not isinstance(value, int) or value < 0 for value in order)
            or len(set(order)) != 3
        ):
            raise ValueError(
                f"Invalid visible_band_order in {scene_config_path}: {order!r}"
            )
        orders[scene_id] = tuple(order)

    unique_orders = set(orders.values())
    if len(unique_orders) != 1:
        raise ValueError(f"Scenes use inconsistent visible band orders: {orders}")
    return list(unique_orders.pop())


def _load_inputs(generation_root, scene_manifest_path):
    generation_root = Path(generation_root).expanduser().resolve()
    if not generation_root.is_dir():
        raise FileNotFoundError(f"Generation root does not exist: {generation_root}")
    scene_manifest_path = Path(scene_manifest_path).expanduser().resolve()
    scene_payload = _load_json(scene_manifest_path)
    if scene_payload.get("schema") != "impgm-large-image-scenes":
        raise ValueError(f"Unsupported scene manifest schema: {scene_manifest_path}")
    scenes = scene_payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"No scenes were found in {scene_manifest_path}.")
    scene_by_id = {}
    for scene in scenes:
        scene_id = str(scene.get("scene_id", ""))
        if not scene_id or scene_id in scene_by_id:
            raise ValueError(f"Invalid or duplicate scene_id: {scene_id!r}")
        scene_by_id[scene_id] = scene

    generation_manifest_path = generation_root / DEFAULT_GENERATION_MANIFEST
    records = _load_jsonl(generation_manifest_path)
    record_by_id = {}
    for record in records:
        if record.get("schema") != "impgm-large-image-generation-record":
            raise ValueError("Unsupported large-image generation record schema.")
        scene_id = str(record.get("scene_id", ""))
        if scene_id not in scene_by_id:
            raise ValueError(
                f"Generation record {scene_id!r} is absent from the scene manifest."
            )
        if scene_id in record_by_id:
            raise ValueError(f"Duplicate generated scene: {scene_id}")
        record_by_id[scene_id] = record
    ordered_scene_ids = [
        str(scene["scene_id"])
        for scene in scenes
        if str(scene["scene_id"]) in record_by_id
    ]
    if not ordered_scene_ids:
        raise ValueError("No generated records match the supplied scene manifest.")

    generation_config_path = generation_root / DEFAULT_GENERATION_CONFIG
    if not generation_config_path.is_file():
        raise FileNotFoundError(f"Missing generation config: {generation_config_path}")
    generation_config = _load_json(generation_config_path)
    return {
        "generation_root": generation_root,
        "scene_manifest_path": scene_manifest_path,
        "scene_manifest": scene_payload,
        "generation_manifest_path": generation_manifest_path,
        "generation_config_path": generation_config_path,
        "generation_config": generation_config,
        "scene_by_id": scene_by_id,
        "record_by_id": record_by_id,
        "scene_ids": ordered_scene_ids,
        "missing_scene_ids": sorted(set(scene_by_id) - set(record_by_id)),
    }


def _check_raster(
    *,
    name,
    path,
    expected_height,
    expected_width,
    expected_bands,
    expected_projection,
    expected_geotransform,
):
    try:
        array, metadata = _read_raster(path, read_array=True)
    except Exception as exc:
        return None, {
            "name": name,
            "path": str(path),
            "status": "failed",
            "failures": [f"unreadable: {exc}"],
        }
    failures = []
    if metadata["height"] != int(expected_height) or metadata["width"] != int(
        expected_width
    ):
        failures.append(
            "size_mismatch: "
            f"expected={expected_height}x{expected_width}, "
            f"actual={metadata['height']}x{metadata['width']}"
        )
    if metadata["bands"] != int(expected_bands):
        failures.append(
            f"band_mismatch: expected={expected_bands}, actual={metadata['bands']}"
        )
    if not _projection_is_same(metadata["projection_wkt"], expected_projection):
        failures.append("projection_mismatch")
    if not np.allclose(
        metadata["geotransform"],
        expected_geotransform,
        rtol=0.0,
        atol=1.0e-8,
    ):
        failures.append("geotransform_mismatch")
    expected_extent = _extent_corners(
        expected_geotransform, expected_width, expected_height
    )
    actual_extent = _extent_corners(
        metadata["geotransform"], metadata["width"], metadata["height"]
    )
    if not np.allclose(actual_extent, expected_extent, rtol=0.0, atol=1.0e-6):
        failures.append("extent_mismatch")
    finite_ratio = _finite_ratio(array)
    if finite_ratio < 1.0:
        failures.append(f"nonfinite_values: ratio={1.0 - finite_ratio:.8f}")
    return array, {
        "name": name,
        "path": metadata["path"],
        "status": "ok" if not failures else "failed",
        "height": metadata["height"],
        "width": metadata["width"],
        "bands": metadata["bands"],
        "finite_ratio": finite_ratio,
        "geotransform": metadata["geotransform"],
        "extent_corners": actual_extent,
        "projection_matches": _projection_is_same(
            metadata["projection_wkt"], expected_projection
        ),
        "failures": failures,
    }


def evaluate_integrity(inputs):
    per_scene = {}
    valid_scene_ids = []
    for scene_id in inputs["scene_ids"]:
        scene = inputs["scene_by_id"][scene_id]
        record = inputs["record_by_id"][scene_id]
        crop = scene["crop"]
        height = int(crop["height"])
        width = int(crop["width"])
        bands = int(scene["source_shape"]["bands"])
        projection = scene.get("projection_wkt", "")
        geotransform = scene["geotransform"]
        raster_specs = (
            ("generated", "generated_tif", bands),
            ("reference", "reference_tif", bands),
            ("condition_mask", "condition_mask_tif", 1),
            ("generated_foreground", "generated_foreground_tif", bands),
            ("generated_mask", "generated_mask_tif", 1),
        )
        checks = {}
        failures = []
        for name, field_name, expected_bands in raster_specs:
            try:
                path = _resolve_record_path(
                    record, field_name, inputs["generation_root"]
                )
                _, check = _check_raster(
                    name=name,
                    path=path,
                    expected_height=height,
                    expected_width=width,
                    expected_bands=expected_bands,
                    expected_projection=projection,
                    expected_geotransform=geotransform,
                )
            except Exception as exc:
                check = {
                    "name": name,
                    "status": "failed",
                    "failures": [str(exc)],
                }
            checks[name] = check
            failures.extend(f"{name}: {item}" for item in check["failures"])

        for field_name in ("tile_metadata", "scene_config", "generated_rgb"):
            try:
                _resolve_record_path(record, field_name, inputs["generation_root"])
            except Exception as exc:
                failures.append(f"{field_name}: {exc}")
        source_image = Path(scene["source_image"])
        source_mask = Path(scene["source_mask"])
        if not source_image.is_file():
            failures.append(f"source_image_missing: {source_image}")
        if not source_mask.is_file():
            failures.append(f"source_mask_missing: {source_mask}")

        status = "ok" if not failures else "failed"
        if status == "ok":
            valid_scene_ids.append(scene_id)
        per_scene[scene_id] = {
            "status": status,
            "expected": {
                "height": height,
                "width": width,
                "bands": bands,
                "geotransform": [float(value) for value in geotransform],
                "extent_corners": _extent_corners(geotransform, width, height),
            },
            "rasters": checks,
            "failures": failures,
        }
    missing_scene_ids = inputs["missing_scene_ids"]
    failed_generated_scenes = len(inputs["scene_ids"]) - len(valid_scene_ids)
    return {
        "status": (
            "ok"
            if not missing_scene_ids and failed_generated_scenes == 0
            else "failed"
        ),
        "expected_scenes": len(inputs["scene_by_id"]),
        "evaluated_scenes": len(inputs["scene_ids"]),
        "passed_scenes": len(valid_scene_ids),
        "failed_scenes": failed_generated_scenes + len(missing_scene_ids),
        "missing_scene_ids": missing_scene_ids,
        "valid_scene_ids": valid_scene_ids,
        "per_scene": per_scene,
    }


def _common_transition_ranges(
    height,
    width,
    tile_size,
    overlap_rate,
    overlap_buffer,
):
    tile_size = int(tile_size)
    overlap_rate = float(overlap_rate)
    overlap_buffer = int(overlap_buffer)
    if tile_size <= 0:
        raise ValueError("--seam-tile-size must be positive.")
    if not 0.0 < overlap_rate < 1.0:
        raise ValueError("--seam-overlap-rate must satisfy 0 < rate < 1.")
    overlap_float = tile_size * overlap_rate
    overlap_width = int(round(overlap_float))
    if not np.isclose(overlap_width, overlap_float):
        raise ValueError(
            "--seam-overlap-rate must produce an integer overlap width."
        )
    if overlap_buffer <= 0 or overlap_buffer > overlap_width:
        raise ValueError(
            "--seam-overlap-buffer must satisfy 0 < buffer <= overlap width."
        )
    overlap_stride = tile_size - overlap_width

    def starts(length, stride):
        if length < tile_size or (length - tile_size) % stride != 0:
            raise ValueError(
                f"Image dimension {length} is incompatible with "
                f"tile_size={tile_size}, stride={stride}."
            )
        return list(range(0, length - tile_size + 1, stride))

    def direct_ranges(length):
        return [[start, start + 1] for start in starts(length, tile_size)[1:]]

    def overlap_ranges(length):
        return [
            [start + overlap_width - overlap_buffer, start + overlap_width]
            for start in starts(length, overlap_stride)[1:]
        ]

    direct_horizontal = direct_ranges(int(height))
    direct_vertical = direct_ranges(int(width))
    overlap_horizontal = overlap_ranges(int(height))
    overlap_vertical = overlap_ranges(int(width))
    return {
        "horizontal": direct_horizontal + overlap_horizontal,
        "vertical": direct_vertical + overlap_vertical,
        "definition": "union_of_direct_boundaries_and_overlap_transition_strips",
        "components": {
            "direct_tile_boundaries": {
                "horizontal": direct_horizontal,
                "vertical": direct_vertical,
            },
            "overlap_transition_strips": {
                "horizontal": overlap_horizontal,
                "vertical": overlap_vertical,
            },
        },
        "parameters": {
            "tile_size": tile_size,
            "overlap_rate": overlap_rate,
            "overlap_width": overlap_width,
            "overlap_stride": overlap_stride,
            "overlap_buffer": overlap_buffer,
        },
    }


def _band_groups(channel_count, visible_band_order):
    visible = [int(index) for index in visible_band_order]
    if len(visible) != 3 or any(index < 0 or index >= channel_count for index in visible):
        raise ValueError(
            f"Invalid visible band order {visible} for {channel_count} channels."
        )
    remaining = [index for index in range(channel_count) if index not in set(visible)]
    return {
        "rgb": visible,
        "nir": remaining if len(remaining) == 1 else None,
    }


def _axis_difference(image, band_indices, orientation):
    selected = np.asarray(image[band_indices], dtype=np.float64)
    if orientation == "horizontal":
        return np.abs(selected[:, 1:, :] - selected[:, :-1, :])
    if orientation == "vertical":
        return np.abs(selected[:, :, 1:] - selected[:, :, :-1])
    raise ValueError(f"Unknown seam orientation: {orientation}")


def _difference_statistics(difference, ranges, orientation, control_offset):
    axis_length = difference.shape[1] + 1 if orientation == "horizontal" else difference.shape[2] + 1
    seam_positions = {
        position
        for left, right in ranges
        for position in range(max(1, left), min(axis_length, right))
    }
    local_positions = set()
    for left, right in ranges:
        width = max(1, right - left)
        candidate_ranges = (
            (left - control_offset - width, left - control_offset),
            (right + control_offset, right + control_offset + width),
        )
        for local_left, local_right in candidate_ranges:
            for position in range(max(1, local_left), min(axis_length, local_right)):
                if position not in seam_positions:
                    local_positions.add(position)
    if not seam_positions:
        return {
            "status": "unavailable",
            "reason": "no_internal_transition",
            "d_seam": None,
            "d_local": None,
            "r_seam": None,
            "seam_value_count": 0,
            "local_value_count": 0,
            "_seam_sum": 0.0,
            "_local_sum": 0.0,
        }
    if not local_positions:
        return {
            "status": "unavailable",
            "reason": "no_valid_local_controls",
            "d_seam": None,
            "d_local": None,
            "r_seam": None,
            "seam_value_count": 0,
            "local_value_count": 0,
            "_seam_sum": 0.0,
            "_local_sum": 0.0,
        }

    seam_indices = np.asarray(sorted(position - 1 for position in seam_positions))
    local_indices = np.asarray(sorted(position - 1 for position in local_positions))
    if orientation == "horizontal":
        seam_values = difference[:, seam_indices, :]
        local_values = difference[:, local_indices, :]
    else:
        seam_values = difference[:, :, seam_indices]
        local_values = difference[:, :, local_indices]
    seam_values = seam_values[np.isfinite(seam_values)]
    local_values = local_values[np.isfinite(local_values)]
    seam_sum = float(seam_values.sum())
    local_sum = float(local_values.sum())
    seam_count = int(seam_values.size)
    local_count = int(local_values.size)
    d_seam = seam_sum / max(seam_count, 1)
    d_local = local_sum / max(local_count, 1)
    return {
        "status": "ok",
        "d_seam": float(d_seam),
        "d_local": float(d_local),
        "r_seam": float(d_seam / (d_local + 1.0e-12)),
        "seam_value_count": seam_count,
        "local_value_count": local_count,
        "_seam_sum": seam_sum,
        "_local_sum": local_sum,
    }


def _public_seam_stats(stats):
    return {key: value for key, value in stats.items() if not key.startswith("_")}


def _combine_statistics(items):
    valid = [item for item in items if item.get("status") == "ok"]
    if not valid:
        return {
            "status": "unavailable",
            "d_seam": None,
            "d_local": None,
            "r_seam": None,
            "seam_value_count": 0,
            "local_value_count": 0,
            "_seam_sum": 0.0,
            "_local_sum": 0.0,
        }
    seam_sum = sum(item["_seam_sum"] for item in valid)
    local_sum = sum(item["_local_sum"] for item in valid)
    seam_count = sum(item["seam_value_count"] for item in valid)
    local_count = sum(item["local_value_count"] for item in valid)
    d_seam = seam_sum / max(seam_count, 1)
    d_local = local_sum / max(local_count, 1)
    return {
        "status": "ok",
        "d_seam": float(d_seam),
        "d_local": float(d_local),
        "r_seam": float(d_seam / (d_local + 1.0e-12)),
        "seam_value_count": int(seam_count),
        "local_value_count": int(local_count),
        "_seam_sum": float(seam_sum),
        "_local_sum": float(local_sum),
    }


def _evaluate_image_seams(
    image,
    transitions,
    visible_band_order,
    control_offset,
    accumulators,
):
    groups = _band_groups(image.shape[0], visible_band_order)
    image_metrics = {}
    for band_name, indices in groups.items():
        if indices is None:
            image_metrics[band_name] = {
                "status": "unavailable",
                "reason": "no_unique_nir_band",
            }
            continue
        orientation_stats = {}
        for orientation in ("horizontal", "vertical"):
            difference = _axis_difference(image, indices, orientation)
            stats = _difference_statistics(
                difference,
                transitions[orientation],
                orientation,
                int(control_offset),
            )
            orientation_stats[orientation] = stats
            accumulators[band_name][orientation].append(stats)
        orientation_stats["all"] = _combine_statistics(
            [orientation_stats["horizontal"], orientation_stats["vertical"]]
        )
        image_metrics[band_name] = {
            key: _public_seam_stats(value)
            for key, value in orientation_stats.items()
        }
    return image_metrics


def _aggregate_image_seams(accumulators):
    aggregate = {}
    for band_name in ("rgb", "nir"):
        if not accumulators[band_name]["horizontal"]:
            aggregate[band_name] = {"status": "unavailable"}
            continue
        horizontal = _combine_statistics(accumulators[band_name]["horizontal"])
        vertical = _combine_statistics(accumulators[band_name]["vertical"])
        combined = _combine_statistics([horizontal, vertical])
        aggregate[band_name] = {
            "horizontal": _public_seam_stats(horizontal),
            "vertical": _public_seam_stats(vertical),
            "all": _public_seam_stats(combined),
        }
    return aggregate


def evaluate_seams(
    inputs,
    scene_ids,
    visible_band_order,
    control_offset,
    tile_size,
    overlap_rate,
    overlap_buffer,
):
    per_scene = {}
    transition_parameters = None
    accumulators = {
        subject: {
            band: {orientation: [] for orientation in ("horizontal", "vertical")}
            for band in ("rgb", "nir")
        }
        for subject in ("generated", "reference")
    }
    for scene_index, scene_id in enumerate(scene_ids, start=1):
        print(f"[Seams {scene_index}/{len(scene_ids)}] {scene_id}")
        record = inputs["record_by_id"][scene_id]
        generated_path = _resolve_record_path(
            record, "generated_tif", inputs["generation_root"]
        )
        generated, _ = _read_raster(generated_path, read_array=True)
        reference_path = _resolve_record_path(
            record, "reference_tif", inputs["generation_root"]
        )
        reference, _ = _read_raster(reference_path, read_array=True)
        transitions = _common_transition_ranges(
            generated.shape[1],
            generated.shape[2],
            tile_size,
            overlap_rate,
            overlap_buffer,
        )
        transition_parameters = transitions["parameters"]
        if reference.shape != generated.shape:
            raise ValueError(
                f"Generated/reference shape mismatch for {scene_id}: "
                f"{generated.shape} vs {reference.shape}."
            )
        generated_metrics = _evaluate_image_seams(
            generated,
            transitions,
            visible_band_order,
            control_offset,
            accumulators["generated"],
        )
        reference_metrics = _evaluate_image_seams(
            reference,
            transitions,
            visible_band_order,
            control_offset,
            accumulators["reference"],
        )
        per_scene[scene_id] = {
            "transition_definition": transitions["definition"],
            "horizontal_transition_ranges": transitions["horizontal"],
            "vertical_transition_ranges": transitions["vertical"],
            "transition_components": transitions["components"],
            "metrics": generated_metrics,
            "reference_metrics": reference_metrics,
        }

    return {
        "protocol": {
            "metric": "mean_absolute_adjacent_pixel_difference",
            "r_seam": "d_seam / (d_local + 1e-12)",
            "preferred_value": 1.0,
            "evaluated_images": ["generated", "reference"],
            "position_selection": (
                "fixed_union_shared_by_all_images_and_generation_paths"
            ),
            "transition_parameters": transition_parameters,
            "local_control_offset_pixels": int(control_offset),
            "visible_band_order": [int(value) for value in visible_band_order],
        },
        # Keep `aggregate` as the generated-image result for compatibility.
        "aggregate": _aggregate_image_seams(accumulators["generated"]),
        "reference_aggregate": _aggregate_image_seams(accumulators["reference"]),
        "per_scene": per_scene,
    }


def _axis_window_positions(length, window_size, stride):
    if window_size > length:
        raise ValueError(
            f"Window size {window_size} exceeds image dimension {length}."
        )
    positions = list(range(0, length - window_size + 1, stride))
    final = length - window_size
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def _condition_runtime(checkpoint_path):
    import torch

    from IMPGM_Config import (
        DATASET_NAME,
        DEVICE,
        IMAGE_MEAN,
        IMAGE_STD,
        INPUT_CHANNELS,
        PROMPT_DICT,
    )
    from IMPGM_Utils import load_model_for_eval
    from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_Config import (
        BASE_CHANNELS,
        CONDITION_CHANNELS,
        MODEL_PATH,
        number_of_classes,
    )
    from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_model import EvaluationUNet

    resolved_checkpoint = Path(checkpoint_path or MODEL_PATH).expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(
            f"Evaluation_UNet checkpoint does not exist: {resolved_checkpoint}"
        )
    model = EvaluationUNet(
        image_channels=INPUT_CHANNELS,
        num_classes=number_of_classes(),
        base_channels=BASE_CHANNELS,
        condition_channels=CONDITION_CHANNELS,
    ).to(DEVICE)
    model, metadata = load_model_for_eval(
        resolved_checkpoint, model, map_location=DEVICE
    )
    if metadata.get("dataset_name") != DATASET_NAME:
        raise ValueError(
            "Evaluation_UNet dataset mismatch: "
            f"checkpoint={metadata.get('dataset_name')!r}, active={DATASET_NAME!r}."
        )
    if int(metadata.get("num_classes") or -1) != len(PROMPT_DICT):
        raise ValueError(
            "Evaluation_UNet class-count mismatch: "
            f"checkpoint={metadata.get('num_classes')}, active={len(PROMPT_DICT)}."
        )
    if metadata.get("input_domain") != "normalized_complete_image":
        raise ValueError(
            "Evaluation_UNet must be trained in the normalized complete-image domain."
        )
    model.eval()
    return {
        "torch": torch,
        "device": torch.device(DEVICE),
        "model": model,
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_metadata": metadata,
        "dataset_name": DATASET_NAME,
        "input_channels": int(INPUT_CHANNELS),
        "prompt_dict": dict(PROMPT_DICT),
        "mean": list(IMAGE_MEAN),
        "std": list(IMAGE_STD),
    }


def _sliding_logits(runtime, image, label_id, window_size, stride, batch_size):
    torch = runtime["torch"]
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != runtime["input_channels"]:
        raise ValueError(
            f"Expected [{runtime['input_channels']},H,W], got {image.shape}."
        )
    height, width = image.shape[1:]
    rows = _axis_window_positions(height, window_size, stride)
    cols = _axis_window_positions(width, window_size, stride)
    coordinates = [(row, col) for row in rows for col in cols]
    mean = torch.tensor(runtime["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(runtime["std"], dtype=torch.float32).view(-1, 1, 1)
    image_tensor = torch.from_numpy(
        np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    ).float().clamp(0.0, 1.0)
    image_tensor = (image_tensor - mean) / std

    one_dimensional = torch.hann_window(
        window_size, periodic=False, dtype=torch.float32
    ).clamp_min(0.05)
    blend_weight = (one_dimensional[:, None] * one_dimensional[None, :]).to(
        runtime["device"]
    )
    logits_sum = torch.zeros(
        (height, width), dtype=torch.float32, device=runtime["device"]
    )
    weight_sum = torch.zeros_like(logits_sum)
    model = runtime["model"]
    with torch.no_grad():
        for start in range(0, len(coordinates), batch_size):
            batch_coordinates = coordinates[start : start + batch_size]
            patches = torch.stack(
                [
                    image_tensor[
                        :, row : row + window_size, col : col + window_size
                    ]
                    for row, col in batch_coordinates
                ]
            ).to(runtime["device"])
            labels = torch.full(
                (len(batch_coordinates),),
                int(label_id),
                dtype=torch.long,
                device=runtime["device"],
            )
            logits = model(patches, labels).float()[:, 0]
            for index, (row, col) in enumerate(batch_coordinates):
                logits_sum[
                    row : row + window_size, col : col + window_size
                ] += logits[index] * blend_weight
                weight_sum[
                    row : row + window_size, col : col + window_size
                ] += blend_weight
    if torch.any(weight_sum <= 0):
        raise RuntimeError("Sliding-window blending left uncovered output pixels.")
    return (logits_sum / weight_sum).detach().cpu()


def _confusion(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    prediction_sum = int(prediction.sum())
    target_sum = int(target.sum())
    return {
        "intersection": intersection,
        "union": union,
        "prediction_sum": prediction_sum,
        "target_sum": target_sum,
    }


def _scores(confusion):
    intersection = int(confusion["intersection"])
    union = int(confusion["union"])
    denominator = int(confusion["prediction_sum"]) + int(confusion["target_sum"])
    return {
        "iou": float(intersection / max(union, 1)),
        "dice": float(2.0 * intersection / max(denominator, 1)),
    }


def _add_confusion(total, value):
    for key in total:
        total[key] += int(value[key])


def evaluate_condition(
    inputs,
    scene_ids,
    *,
    checkpoint_path,
    object_label,
    window_size,
    overlap,
    batch_size,
    threshold,
):
    if window_size <= 0 or window_size % 8 != 0:
        raise ValueError("--condition-window-size must be positive and divisible by 8.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("--condition-overlap must satisfy 0 <= overlap < 1.")
    if batch_size <= 0:
        raise ValueError("--condition-batch-size must be positive.")
    stride = int(round(window_size * (1.0 - overlap)))
    if stride <= 0:
        raise ValueError("Condition overlap produces a non-positive stride.")
    runtime = _condition_runtime(checkpoint_path)
    if object_label not in runtime["prompt_dict"]:
        raise KeyError(
            f"Unknown object label {object_label!r}; available={sorted(runtime['prompt_dict'])}."
        )
    label_id = int(runtime["prompt_dict"][object_label])
    aggregate_generated = {
        "intersection": 0,
        "union": 0,
        "prediction_sum": 0,
        "target_sum": 0,
    }
    aggregate_reference = dict(aggregate_generated)
    per_scene = {}
    for scene_index, scene_id in enumerate(scene_ids, start=1):
        print(f"[Condition {scene_index}/{len(scene_ids)}] {scene_id}")
        record = inputs["record_by_id"][scene_id]
        generated, _ = _read_raster(
            _resolve_record_path(record, "generated_tif", inputs["generation_root"]),
            read_array=True,
        )
        reference, _ = _read_raster(
            _resolve_record_path(record, "reference_tif", inputs["generation_root"]),
            read_array=True,
        )
        mask, _ = _read_raster(
            _resolve_record_path(
                record, "condition_mask_tif", inputs["generation_root"]
            ),
            read_array=True,
        )
        target = np.isfinite(mask[0]) & (mask[0] > 0.5)
        generated_logits = _sliding_logits(
            runtime, generated, label_id, window_size, stride, batch_size
        )
        reference_logits = _sliding_logits(
            runtime, reference, label_id, window_size, stride, batch_size
        )
        generated_prediction = runtime["torch"].sigmoid(generated_logits).numpy() >= threshold
        reference_prediction = runtime["torch"].sigmoid(reference_logits).numpy() >= threshold
        generated_confusion = _confusion(generated_prediction, target)
        reference_confusion = _confusion(reference_prediction, target)
        _add_confusion(aggregate_generated, generated_confusion)
        _add_confusion(aggregate_reference, reference_confusion)
        per_scene[scene_id] = {
            "generated_consistency": _scores(generated_confusion),
            "real_image_evaluator_ceiling": _scores(reference_confusion),
            "generated_confusion": generated_confusion,
            "reference_confusion": reference_confusion,
            "window_count": len(
                _axis_window_positions(generated.shape[1], window_size, stride)
            )
            * len(_axis_window_positions(generated.shape[2], window_size, stride)),
        }
    return {
        "protocol": (
            "frozen_class_conditioned_unet_with_overlapping_training_scale_windows_"
            "and_weighted_logit_blending"
        ),
        "object_label": object_label,
        "label_id": label_id,
        "checkpoint": runtime["checkpoint"],
        "checkpoint_metadata": {
            "dataset_name": runtime["checkpoint_metadata"].get("dataset_name"),
            "num_classes": runtime["checkpoint_metadata"].get("num_classes"),
            "input_domain": runtime["checkpoint_metadata"].get("input_domain"),
            "conditioning": runtime["checkpoint_metadata"].get("conditioning"),
        },
        "window_size": int(window_size),
        "window_overlap": float(overlap),
        "window_stride": int(stride),
        "threshold": float(threshold),
        "large_image_resize": False,
        "aggregate": {
            "generated_consistency": _scores(aggregate_generated),
            "real_image_evaluator_ceiling": _scores(aggregate_reference),
            "generated_confusion": aggregate_generated,
            "reference_confusion": aggregate_reference,
        },
        "per_scene": per_scene,
    }


def _stable_seed(seed, token):
    digest = hashlib.sha256(f"{int(seed)}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _classify_patch(row, col, size, transitions):
    horizontal = [
        item
        for item in transitions["horizontal"]
        if row < (item[0] + item[1]) / 2.0 < row + size
    ]
    vertical = [
        item
        for item in transitions["vertical"]
        if col < (item[0] + item[1]) / 2.0 < col + size
    ]
    if horizontal and vertical:
        kind = "cross_both_transitions"
    elif horizontal:
        kind = "cross_horizontal_transition"
    elif vertical:
        kind = "cross_vertical_transition"
    else:
        kind = "regular_interior"
    return kind, horizontal, vertical


def _select_patch_coordinates(
    scene_id, height, width, patch_size, count, transitions, seed
):
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"Patch size {patch_size} exceeds scene {scene_id} size {height}x{width}."
        )
    if count <= 0:
        raise ValueError("--patches-per-scene must be positive.")
    rng = np.random.default_rng(_stable_seed(seed, scene_id))
    candidates = []
    seen = set()

    def add(row, col, role):
        row = int(max(0, min(row, height - patch_size)))
        col = int(max(0, min(col, width - patch_size)))
        key = (row, col)
        if key not in seen:
            seen.add(key)
            candidates.append((row, col, role))

    # Canonical patch-grid boundaries are independent of the stitching mode,
    # so direct and overlap-aware runs receive exactly the same coordinates.
    seam_candidates = []
    for center in range(patch_size, height, patch_size):
        seam_candidates.append(
            (
                center - patch_size // 2,
                int(rng.integers(0, width - patch_size + 1)),
                "canonical_boundary_centered",
            )
        )
    for center in range(patch_size, width, patch_size):
        seam_candidates.append(
            (
                int(rng.integers(0, height - patch_size + 1)),
                center - patch_size // 2,
                "canonical_boundary_centered",
            )
        )
    rng.shuffle(seam_candidates)
    seam_target = min((count + 1) // 2, len(seam_candidates))
    for row, col, role in seam_candidates[:seam_target]:
        add(row, col, role)

    regular_rows = _axis_window_positions(height, patch_size, patch_size)
    regular_cols = _axis_window_positions(width, patch_size, patch_size)
    regular_candidates = [
        (row, col, "regular_grid") for row in regular_rows for col in regular_cols
    ]
    rng.shuffle(regular_candidates)
    for row, col, role in regular_candidates:
        if len(candidates) >= count:
            break
        add(row, col, role)
    attempts = 0
    while len(candidates) < count and attempts < count * 100:
        add(
            int(rng.integers(0, height - patch_size + 1)),
            int(rng.integers(0, width - patch_size + 1)),
            "deterministic_random",
        )
        attempts += 1
    if len(candidates) < count:
        raise RuntimeError(
            f"Could only select {len(candidates)} unique patches for {scene_id}."
        )

    result = []
    for patch_index, (row, col, role) in enumerate(candidates[:count]):
        kind, horizontal, vertical = _classify_patch(
            row, col, patch_size, transitions
        )
        result.append(
            {
                "patch_id": f"{scene_id}_patch_{patch_index + 1:03d}",
                "scene_id": scene_id,
                "row_start": row,
                "row_end": row + patch_size,
                "col_start": col,
                "col_end": col + patch_size,
                "selection_role": role,
                "transition_relation": kind,
                "horizontal_transitions": horizontal,
                "vertical_transitions": vertical,
            }
        )
    return result


def _quality_runtime(visible_band_order):
    import torch

    from IMPGM_Config import DEVICE
    from Evaluation.Evaluation_Code.IMPGM_Quality_Metrics import (
        build_inception_feature_extractor,
        compute_FID_from_features,
        compute_KID_from_features,
        compute_histogram_W1,
        compute_mean_spectrum_SAM_degrees,
        compute_multiband_SWD,
        extract_inception_features,
    )

    return {
        "torch": torch,
        "device": torch.device(DEVICE),
        "visible_band_order": [int(value) for value in visible_band_order],
        "build_inception": build_inception_feature_extractor,
        "extract_inception": extract_inception_features,
        "compute_fid": compute_FID_from_features,
        "compute_kid": compute_KID_from_features,
        "compute_swd": compute_multiband_SWD,
        "compute_sam": compute_mean_spectrum_SAM_degrees,
        "compute_w1": compute_histogram_W1,
    }


def _extract_features(runtime, images, band_mode, batch_size, extractor):
    features = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(runtime["device"])
        with runtime["torch"].no_grad():
            values = runtime["extract_inception"](
                batch,
                band_mode=band_mode,
                vis_band_order=runtime["visible_band_order"],
                feature_extractor=extractor,
            )
        features.append(values.detach().cpu())
    return runtime["torch"].cat(features, dim=0).float()


def _spectral_metrics(runtime, generated, reference, masks, bins):
    torch = runtime["torch"]
    channels = generated.shape[1]
    generated_sum = torch.zeros(channels, dtype=torch.float64)
    reference_sum = torch.zeros(channels, dtype=torch.float64)
    generated_hist = torch.zeros(channels, bins)
    reference_hist = torch.zeros(channels, bins)
    support = 0
    for index in range(generated.shape[0]):
        mask = masks[index, 0] > 0.5
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)
        generated_pixels = generated[index, :, mask]
        reference_pixels = reference[index, :, mask]
        support += int(mask.sum().item())
        generated_sum += generated_pixels.double().sum(dim=1)
        reference_sum += reference_pixels.double().sum(dim=1)
        for channel in range(channels):
            generated_hist[channel] += torch.histc(
                generated_pixels[channel], bins=bins, min=0.0, max=1.0
            )
            reference_hist[channel] += torch.histc(
                reference_pixels[channel], bins=bins, min=0.0, max=1.0
            )
    generated_mean = (generated_sum / max(support, 1)).float()
    reference_mean = (reference_sum / max(support, 1)).float()
    per_band_w1 = [
        runtime["compute_w1"](
            generated_hist[channel], reference_hist[channel], bins
        )
        for channel in range(channels)
    ]
    sam = runtime["compute_sam"](generated_mean, reference_mean)
    return {
        "pixel_support": int(support),
        "mean_spectrum_generated": generated_mean.tolist(),
        "mean_spectrum_reference": reference_mean.tolist(),
        "mean_spectrum_sam_deg": None if sam is None else float(sam),
        "mean_spectrum_sam_status": "ok" if sam is not None else "undefined_zero_norm",
        "per_band_w1": [float(value) for value in per_band_w1],
        "mean_w1": float(np.mean(per_band_w1)),
    }


def _stable_subset_indices(patch_records, count, seed):
    ranked = sorted(
        range(len(patch_records)),
        key=lambda index: hashlib.sha256(
            f"{int(seed)}:{patch_records[index]['patch_id']}".encode("utf-8")
        ).digest(),
    )
    return ranked[: min(int(count), len(ranked))]


def evaluate_patch_quality(
    inputs,
    scene_ids,
    *,
    visible_band_order,
    patch_size,
    patches_per_scene,
    metric_batch_size,
    max_swd_samples,
    histogram_bins,
    seed,
    seam_tile_size,
    seam_overlap_rate,
    seam_overlap_buffer,
):
    if patch_size <= 0 or metric_batch_size <= 0 or max_swd_samples <= 0:
        raise ValueError("Patch and metric batch parameters must be positive.")
    runtime = _quality_runtime(visible_band_order)
    generated_patches = []
    reference_patches = []
    mask_patches = []
    patch_records = []
    per_scene = {}
    for scene_index, scene_id in enumerate(scene_ids, start=1):
        print(f"[Patches {scene_index}/{len(scene_ids)}] {scene_id}")
        record = inputs["record_by_id"][scene_id]
        generated, _ = _read_raster(
            _resolve_record_path(record, "generated_tif", inputs["generation_root"]),
            read_array=True,
        )
        reference, _ = _read_raster(
            _resolve_record_path(record, "reference_tif", inputs["generation_root"]),
            read_array=True,
        )
        mask, _ = _read_raster(
            _resolve_record_path(
                record, "condition_mask_tif", inputs["generation_root"]
            ),
            read_array=True,
        )
        if generated.shape != reference.shape:
            raise ValueError(
                f"Generated/reference shape mismatch for {scene_id}: "
                f"{generated.shape} vs {reference.shape}."
            )
        _band_groups(generated.shape[0], runtime["visible_band_order"])
        transitions = _common_transition_ranges(
            generated.shape[1],
            generated.shape[2],
            seam_tile_size,
            seam_overlap_rate,
            seam_overlap_buffer,
        )
        coordinates = _select_patch_coordinates(
            scene_id,
            generated.shape[1],
            generated.shape[2],
            patch_size,
            patches_per_scene,
            transitions,
            seed,
        )
        for item in coordinates:
            row_start, row_end = item["row_start"], item["row_end"]
            col_start, col_end = item["col_start"], item["col_end"]
            generated_patches.append(
                np.asarray(
                    generated[:, row_start:row_end, col_start:col_end],
                    dtype=np.float32,
                )
            )
            reference_patches.append(
                np.asarray(
                    reference[:, row_start:row_end, col_start:col_end],
                    dtype=np.float32,
                )
            )
            mask_patches.append(
                np.asarray(mask[:1, row_start:row_end, col_start:col_end], dtype=np.float32)
            )
            patch_records.append(item)
        per_scene[scene_id] = {
            "patch_count": len(coordinates),
            "patches": coordinates,
        }

    torch = runtime["torch"]
    generated_tensor = torch.from_numpy(
        np.nan_to_num(np.stack(generated_patches), nan=0.0, posinf=1.0, neginf=0.0)
    ).float().clamp(0.0, 1.0)
    reference_tensor = torch.from_numpy(
        np.nan_to_num(np.stack(reference_patches), nan=0.0, posinf=1.0, neginf=0.0)
    ).float().clamp(0.0, 1.0)
    masks_tensor = torch.from_numpy(
        np.nan_to_num(np.stack(mask_patches), nan=0.0, posinf=1.0, neginf=0.0)
    ).float()

    print(f"[Patch quality] Extracting Inception features from {len(patch_records)} patches.")
    extractor = runtime["build_inception"](runtime["device"])
    generated_rgb_features = _extract_features(
        runtime, generated_tensor, "rgb", metric_batch_size, extractor
    )
    reference_rgb_features = _extract_features(
        runtime, reference_tensor, "rgb", metric_batch_size, extractor
    )
    generated_nir_features = None
    reference_nir_features = None
    if generated_tensor.shape[1] == 4:
        generated_nir_features = _extract_features(
            runtime, generated_tensor, "nir", metric_batch_size, extractor
        )
        reference_nir_features = _extract_features(
            runtime, reference_tensor, "nir", metric_batch_size, extractor
        )
    fid_rgb = runtime["compute_fid"](
        generated_rgb_features, reference_rgb_features, device=runtime["device"]
    )
    kid_rgb = runtime["compute_kid"](
        generated_rgb_features, reference_rgb_features
    )
    fid_nir = None
    kid_nir = None
    if generated_nir_features is not None:
        fid_nir = runtime["compute_fid"](
            generated_nir_features,
            reference_nir_features,
            device=runtime["device"],
        )
        kid_nir = runtime["compute_kid"](
            generated_nir_features, reference_nir_features
        )
    swd_indices = _stable_subset_indices(
        patch_records, max_swd_samples, seed
    )
    swd_value = runtime["compute_swd"](
        generated_tensor[swd_indices].to(runtime["device"]),
        reference_tensor[swd_indices].to(runtime["device"]),
        seed=seed,
    )
    spectral = _spectral_metrics(
        runtime,
        generated_tensor,
        reference_tensor,
        masks_tensor,
        histogram_bins,
    )
    relation_counts = {}
    for item in patch_records:
        relation = item["transition_relation"]
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    return {
        "protocol": {
            "patch_size": int(patch_size),
            "patches_per_scene": int(patches_per_scene),
            "selection": "shared_deterministic_regular_and_canonical_boundary_patches",
            "coordinate_invariance": (
                "coordinates depend only on scene_id, image size, patch size, and seed"
            ),
            "seed": int(seed),
            "raw_tif_values_clamped_to_unit_interval": True,
            "visible_band_order": runtime["visible_band_order"],
            "spectral_region": "condition_mask_or_full_patch_when_empty",
        },
        "patch_count": len(patch_records),
        "transition_relation_counts": relation_counts,
        "distribution": {
            "fid_rgb": None if fid_rgb is None else float(fid_rgb),
            "fid_rgb_status": "ok" if fid_rgb is not None else "unavailable",
            "kid_rgb": None if kid_rgb is None else float(kid_rgb),
            "kid_rgb_status": "ok" if kid_rgb is not None else "unavailable",
            "fid_nir": None if fid_nir is None else float(fid_nir),
            "fid_nir_status": "ok" if fid_nir is not None else "unavailable",
            "kid_nir": None if kid_nir is None else float(kid_nir),
            "kid_nir_status": "ok" if kid_nir is not None else "unavailable",
            "swd_all_bands": (
                None if swd_value is None else float(swd_value.detach().cpu().item())
            ),
            "swd_status": "ok" if swd_value is not None else "unavailable",
            "swd_patch_count": len(swd_indices),
        },
        "spectral": spectral,
        "per_scene": per_scene,
    }


def evaluate(args):
    inputs = _load_inputs(args.generation_root, args.scene_manifest)
    print("[1/4] Checking raster integrity and georeferencing.")
    integrity = evaluate_integrity(inputs)
    scene_ids = integrity["valid_scene_ids"]
    if not scene_ids:
        raise RuntimeError("No scene passed integrity checks; metric evaluation cannot continue.")

    generation_config = inputs["generation_config"]
    visible_band_order = _load_visible_band_order(inputs, scene_ids)
    object_label = args.object_label or generation_config.get("object_label", "Water")
    result = {
        "schema": "impgm-large-image-evaluation",
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "generation_root": str(inputs["generation_root"]),
        "scene_manifest": str(inputs["scene_manifest_path"]),
        "generation_manifest": str(inputs["generation_manifest_path"]),
        "generation_config": str(inputs["generation_config_path"]),
        "dataset_yaml": os.environ.get("IMPGM_DATASET_YAML"),
        "stitch_mode": generation_config.get("stitch_mode"),
        "sampler": generation_config.get("sampler"),
        "requested_evaluations": {
            "seams": bool(args.evaluate_seams),
            "condition": bool(args.evaluate_condition),
            "patch_quality": bool(args.evaluate_patch_quality),
        },
        "scene_count": len(inputs["scene_ids"]),
        "metric_scene_count": len(scene_ids),
        "integrity": integrity,
    }
    if args.evaluate_seams:
        print("[2/4] Evaluating seam continuity.")
        result["seam_continuity"] = evaluate_seams(
            inputs,
            scene_ids,
            visible_band_order,
            args.seam_control_offset,
            args.seam_tile_size,
            args.seam_overlap_rate,
            args.seam_overlap_buffer,
        )
    if args.evaluate_condition:
        print("[3/4] Evaluating condition consistency with sliding windows.")
        result["condition_consistency"] = evaluate_condition(
            inputs,
            scene_ids,
            checkpoint_path=args.evaluation_checkpoint,
            object_label=object_label,
            window_size=args.condition_window_size,
            overlap=args.condition_overlap,
            batch_size=args.condition_batch_size,
            threshold=args.condition_threshold,
        )
    if args.evaluate_patch_quality:
        print("[4/4] Evaluating local distribution and spectral quality.")
        result["patch_quality"] = evaluate_patch_quality(
            inputs,
            scene_ids,
            visible_band_order=visible_band_order,
            patch_size=args.patch_size,
            patches_per_scene=args.patches_per_scene,
            metric_batch_size=args.metric_batch_size,
            max_swd_samples=args.max_swd_samples,
            histogram_bins=args.histogram_bins,
            seed=args.seed,
            seam_tile_size=args.seam_tile_size,
            seam_overlap_rate=args.seam_overlap_rate,
            seam_overlap_buffer=args.seam_overlap_buffer,
        )
    result["status"] = "complete" if integrity["status"] == "ok" else "partial"
    _write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "scene_count": result["scene_count"],
        "metric_scene_count": result["metric_scene_count"],
        "output": str(Path(args.output).expanduser().resolve()),
    }, ensure_ascii=False, indent=2))
    return result


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate IMPGM direct or overlap-aware large-image generation."
    )
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--evaluate-seams", action="store_true")
    parser.add_argument("--evaluate-condition", action="store_true")
    parser.add_argument("--evaluate-patch-quality", action="store_true")
    parser.add_argument("--seam-control-offset", type=int, default=32)
    parser.add_argument("--seam-tile-size", type=int, default=256)
    parser.add_argument("--seam-overlap-rate", type=float, default=0.25)
    parser.add_argument("--seam-overlap-buffer", type=int, default=24)
    parser.add_argument("--evaluation-checkpoint", default=None)
    parser.add_argument("--object-label", default=None)
    parser.add_argument("--condition-window-size", type=int, default=256)
    parser.add_argument("--condition-overlap", type=float, default=0.5)
    parser.add_argument("--condition-batch-size", type=int, default=8)
    parser.add_argument("--condition-threshold", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-scene", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument("--max-swd-samples", type=int, default=128)
    parser.add_argument("--histogram-bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.seam_control_offset < 0:
        raise ValueError("--seam-control-offset must be non-negative.")
    evaluate(args)


if __name__ == "__main__":
    main()
