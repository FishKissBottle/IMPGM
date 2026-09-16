"""Prepare fixed large-image scenes and run traceable IMPGM inference."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


SCHEMA_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _gdal():
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise RuntimeError(
            "GDAL is required for large-image scene preparation and I/O."
        ) from exc
    gdal.UseExceptions()
    return gdal


def _open_raster(path):
    gdal = _gdal()
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open raster: {path}")
    return dataset


def _raster_metadata(path):
    dataset = _open_raster(path)
    metadata = {
        "width": int(dataset.RasterXSize),
        "height": int(dataset.RasterYSize),
        "bands": int(dataset.RasterCount),
        "projection_wkt": dataset.GetProjection() or "",
        "geotransform": [float(value) for value in dataset.GetGeoTransform()],
    }
    dataset = None
    return metadata


def _read_window(path, row, col, height, width):
    dataset = _open_raster(path)
    if row < 0 or col < 0 or row + height > dataset.RasterYSize or col + width > dataset.RasterXSize:
        dataset = None
        raise ValueError(
            f"Window {(row, col, height, width)} exceeds raster bounds for {path}."
        )
    array = dataset.ReadAsArray(int(col), int(row), int(width), int(height))
    dataset = None
    if array is None:
        raise RuntimeError(f"GDAL failed to read window from {path}.")
    return np.asarray(array)


def _write_tif(path, array, projection_wkt, geotransform, nodata=None):
    gdal = _gdal()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"TIF data must be [C, H, W] or [H, W], got {array.shape}.")
    bands, height, width = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path),
        int(width),
        int(height),
        int(bands),
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    if dataset is None:
        raise RuntimeError(f"Unable to create output raster: {path}")
    dataset.SetGeoTransform(tuple(float(value) for value in geotransform))
    if projection_wkt:
        dataset.SetProjection(str(projection_wkt))
    for band_index in range(bands):
        band = dataset.GetRasterBand(band_index + 1)
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
        band.WriteArray(array[band_index])
    dataset.FlushCache()
    dataset = None


def _crop_geotransform(geotransform, row, col):
    gt = [float(value) for value in geotransform]
    return [
        gt[0] + col * gt[1] + row * gt[2],
        gt[1],
        gt[2],
        gt[3] + col * gt[4] + row * gt[5],
        gt[4],
        gt[5],
    ]


def _pair_key(root, path, suffix):
    relative = path.relative_to(root)
    stem = relative.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Expected {path} to end with {suffix!r} before its extension.")
    stem = stem[: -len(suffix)]
    return (relative.parent / stem).as_posix()


def _discover_pairs(source_root, image_pattern, mask_pattern):
    source_root = Path(source_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    image_paths = sorted(
        path.resolve() for path in source_root.glob(image_pattern) if path.is_file()
    )
    mask_paths = sorted(
        path.resolve() for path in source_root.glob(mask_pattern) if path.is_file()
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images matched {image_pattern!r} under {source_root}."
        )
    if not mask_paths:
        raise FileNotFoundError(
            f"No masks matched {mask_pattern!r} under {source_root}."
        )
    images = {_pair_key(source_root, path, "_data"): path for path in image_paths}
    masks = {_pair_key(source_root, path, "_mask"): path for path in mask_paths}
    missing_images = sorted(set(masks) - set(images))
    missing_masks = sorted(set(images) - set(masks))
    if missing_images or missing_masks:
        raise ValueError(
            "Large-image data/mask pairing is incomplete: "
            f"missing_images={missing_images[:5]}, missing_masks={missing_masks[:5]}."
        )
    return source_root, [
        (key, images[key], masks[key]) for key in sorted(images)
    ]


def _axis_positions(length, crop_size, stride):
    if crop_size > length:
        return []
    positions = list(range(0, length - crop_size + 1, stride))
    final = length - crop_size
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def _mask_2d(array, path):
    array = np.asarray(array)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Mask must contain one band, got {array.shape} in {path}.")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional, got {array.shape} in {path}.")
    return array


def prepare_scenes(
    *,
    source_root,
    image_pattern,
    mask_pattern,
    crop_size,
    num_scenes,
    min_target_ratio,
    max_target_ratio,
    seed,
    output,
):
    if crop_size <= 0 or num_scenes <= 0:
        raise ValueError("crop_size and num_scenes must be positive.")
    if not 0.0 <= min_target_ratio <= max_target_ratio <= 1.0:
        raise ValueError("Target-ratio bounds must satisfy 0 <= min <= max <= 1.")

    source_root, pairs = _discover_pairs(source_root, image_pattern, mask_pattern)
    rng = random.Random(int(seed))
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)
    grid_stride = max(1, crop_size // 2)
    candidates_by_source = []

    for source_key, image_path, mask_path in shuffled_pairs:
        image_meta = _raster_metadata(image_path)
        mask_meta = _raster_metadata(mask_path)
        if mask_meta["bands"] != 1:
            raise ValueError(f"Mask must contain one band for {source_key!r}.")
        if (image_meta["height"], image_meta["width"]) != (
            mask_meta["height"], mask_meta["width"]
        ):
            raise ValueError(f"Image and mask dimensions differ for {source_key!r}.")
        rows = _axis_positions(image_meta["height"], crop_size, grid_stride)
        cols = _axis_positions(image_meta["width"], crop_size, grid_stride)
        if not rows or not cols:
            continue
        mask = _mask_2d(
            _read_window(mask_path, 0, 0, mask_meta["height"], mask_meta["width"]),
            mask_path,
        )
        foreground = np.isfinite(mask) & (mask > 0)
        candidates = []
        for row in rows:
            for col in cols:
                ratio = float(
                    foreground[row : row + crop_size, col : col + crop_size].mean()
                )
                if min_target_ratio <= ratio <= max_target_ratio:
                    candidates.append((row, col, ratio))
        rng.shuffle(candidates)
        if candidates:
            candidates_by_source.append(
                (source_key, image_path, mask_path, image_meta, candidates)
            )

    selected = []
    for source in candidates_by_source:
        selected.append((*source[:4], source[4][0]))
        if len(selected) == num_scenes:
            break
    if len(selected) < num_scenes:
        remaining = []
        for source in candidates_by_source:
            for candidate in source[4][1:]:
                remaining.append((*source[:4], candidate))
        rng.shuffle(remaining)
        selected.extend(remaining[: num_scenes - len(selected)])
    if len(selected) < num_scenes:
        raise RuntimeError(
            f"Only {len(selected)} valid crops satisfy the requested ratio range; "
            "adjust the thresholds explicitly and rerun prepare-scenes."
        )

    scenes = []
    for index, (source_key, image_path, mask_path, image_meta, candidate) in enumerate(
        selected, start=1
    ):
        row, col, ratio = candidate
        scene_id = f"scene_{index:03d}"
        scenes.append(
            {
                "scene_id": scene_id,
                "source_key": source_key,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "source_shape": {
                    "height": image_meta["height"],
                    "width": image_meta["width"],
                    "bands": image_meta["bands"],
                },
                "crop": {
                    "row_start": int(row),
                    "row_end": int(row + crop_size),
                    "col_start": int(col),
                    "col_end": int(col + crop_size),
                    "height": int(crop_size),
                    "width": int(crop_size),
                },
                "target_ratio": ratio,
                "projection_wkt": image_meta["projection_wkt"],
                "geotransform": _crop_geotransform(
                    image_meta["geotransform"], row, col
                ),
            }
        )

    payload = {
        "schema": "impgm-large-image-scenes",
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_root": str(source_root),
        "selection": {
            "image_pattern": image_pattern,
            "mask_pattern": mask_pattern,
            "crop_size": int(crop_size),
            "num_scenes": int(num_scenes),
            "min_target_ratio": float(min_target_ratio),
            "max_target_ratio": float(max_target_ratio),
            "grid_stride": int(grid_stride),
            "seed": int(seed),
            "prefer_distinct_sources": True,
        },
        "scenes": scenes,
    }
    _write_json(output, payload)
    print(f"Prepared {len(scenes)} scenes: {Path(output).resolve()}")


def _validate_scene_manifest(payload, path):
    if payload.get("schema") != "impgm-large-image-scenes":
        raise ValueError(f"Unsupported scene manifest schema in {path}.")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"No scenes were found in {path}.")
    required = {
        "scene_id",
        "source_image",
        "source_mask",
        "crop",
        "target_ratio",
        "projection_wkt",
        "geotransform",
    }
    for scene in scenes:
        missing = required - set(scene)
        if missing:
            raise ValueError(f"Scene is missing fields {sorted(missing)}: {scene}")
    return scenes


def _tile_positions(length, tile_size, stride):
    if length < tile_size or (length - tile_size) % stride != 0:
        raise ValueError(
            f"Size {length} is incompatible with tile_size={tile_size}, stride={stride}."
        )
    return list(range(0, length - tile_size + 1, stride))


def _tile_records(height, width, tile_size, stride, scene_seed, direct):
    rows = _tile_positions(height, tile_size, stride)
    cols = _tile_positions(width, tile_size, stride)
    coordinates = [(row, col) for row in rows for col in cols]
    if direct:
        ordered = coordinates
    else:
        ordered = sorted(coordinates, key=lambda item: (item[0] + item[1], -item[0]))
    order = {coordinate: index for index, coordinate in enumerate(ordered)}
    records = []
    for row, col in coordinates:
        stream_index = order[(row, col)]
        records.append(
            {
                "tile_index": len(records),
                "row_start": row,
                "row_end": row + tile_size,
                "col_start": col,
                "col_end": col + tile_size,
                "rng_stream_index": stream_index,
                "seed": int(scene_seed + stream_index) if direct else None,
                "fggen_seed": int(scene_seed + stream_index) if direct else None,
                "imgsyn_seed": (
                    int(scene_seed + 10000 + stream_index) if direct else None
                ),
            }
        )
    return records


def _normalize_image(image, expected_channels, custom_normalize):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[None, ...]
    if image.ndim != 3 or image.shape[0] != expected_channels:
        raise ValueError(
            f"Expected a [C, H, W] image with {expected_channels} channels, got {image.shape}."
        )
    normalized = custom_normalize(np.moveaxis(image, 0, -1))
    return np.moveaxis(normalized, -1, 0).copy()


def _percentile_rgb(array, band_order, percentiles=(1.0, 99.0)):
    rgb = np.asarray(array, dtype=np.float32)[list(band_order)]
    output = np.zeros_like(rgb, dtype=np.float32)
    for index, band in enumerate(rgb):
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, percentiles)
        if high <= low:
            output[index] = np.clip(band, 0.0, 1.0)
        else:
            output[index] = np.clip((band - low) / (high - low), 0.0, 1.0)
    return output


def _save_rgb(path, array, band_order):
    from PIL import Image

    rgb = _percentile_rgb(array, band_order)
    uint8 = np.rint(rgb * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(np.moveaxis(uint8, 0, -1), mode="RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _checkpoint_metadata(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint does not exist: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def _load_runtime(args):
    import torch

    from IMPGM_Config import (
        DEVICE,
        FGGEN_CONTROLNET_MODEL_SAVEPATH,
        FGGEN_DIFFUSION_CONFIG,
        FGGEN_DIFFUSION_MODEL_SAVEPATH,
        FGSEG_UNET_MODEL_SAVEPATH,
        IMAGE_MEAN,
        IMAGE_STD,
        IMGSYN_CONTROLNET_MODEL_SAVEPATH,
        IMGSYN_DIFFUSION_CONFIG,
        IMGSYN_DIFFUSION_MODEL_SAVEPATH,
        IMG_SIZE,
        INPUT_CHANNELS,
        LATENT_HIDDENCHANNEL,
        PROMPT_DICT,
        SIDELENGTH_SCALE_FACTOR,
        VAE_MODEL_SAVEPATH,
        VIS_BAND_ORDER,
        custom_normalize,
    )
    from Large_Image_Generation.IMPGM_LargeImg import Generate_LargeImg
    from IMPGM_Scheduler import build_beta_schedule_from_config

    for label in (args.object_label, args.object_reference_label):
        if label not in PROMPT_DICT:
            raise ValueError(f"Unknown class label for the selected dataset: {label!r}.")

    fg_beta = build_beta_schedule_from_config(FGGEN_DIFFUSION_CONFIG)
    img_beta = build_beta_schedule_from_config(IMGSYN_DIFFUSION_CONFIG)
    if fg_beta.shape != img_beta.shape or not torch.allclose(fg_beta, img_beta):
        raise ValueError(
            "Generate_LargeImg requires matching FgGen and ImgSyn noise schedules."
        )

    checkpoint_paths = {
        "vae": VAE_MODEL_SAVEPATH,
        "fggen_diffusion": FGGEN_DIFFUSION_MODEL_SAVEPATH,
        "imgsyn_diffusion": IMGSYN_DIFFUSION_MODEL_SAVEPATH,
        "fgseg": FGSEG_UNET_MODEL_SAVEPATH,
    }
    if args.fggen_controlnet:
        checkpoint_paths["fggen_controlnet"] = FGGEN_CONTROLNET_MODEL_SAVEPATH
    if args.imgsyn_controlnet:
        checkpoint_paths["imgsyn_controlnet"] = IMGSYN_CONTROLNET_MODEL_SAVEPATH
    checkpoints = {
        name: _checkpoint_metadata(path) for name, path in checkpoint_paths.items()
    }

    generator = Generate_LargeImg(
        sampler_mode=args.sampler,
        VAE_model_savepath=VAE_MODEL_SAVEPATH,
        FgGen_Diffusion_model_savepath=FGGEN_DIFFUSION_MODEL_SAVEPATH,
        ImgSyn_Diffusion_model_savepath=IMGSYN_DIFFUSION_MODEL_SAVEPATH,
        FgGen_ControlNet_model_savepath=FGGEN_CONTROLNET_MODEL_SAVEPATH,
        ImgSyn_ControlNet_model_savepath=IMGSYN_CONTROLNET_MODEL_SAVEPATH,
        FgSeg_model_savepath=FGSEG_UNET_MODEL_SAVEPATH,
        beta_t=fg_beta,
        overlap_rate=args.overlap_rate,
        overlap_buffer_thx=args.overlap_buffer,
        use_fggen_controlnet=args.fggen_controlnet,
        use_imgsyn_controlnet=args.imgsyn_controlnet,
        is_FgGenInpaint_Resample=args.fggen_inpaint_resample,
        is_ImgSynInpaint_Resample=args.imgsyn_inpaint_resample,
        batch_size=args.tile_batch_size,
        is_Fixed_Prompt=True,
        ddim_steps=args.ddim_steps if args.ddim_steps is not None else 500,
        ddim_eta=args.ddim_eta if args.ddim_eta is not None else 0.0,
    )
    return {
        "torch": torch,
        "generator": generator,
        "device": DEVICE,
        "image_size": IMG_SIZE,
        "input_channels": INPUT_CHANNELS,
        "latent_channels": LATENT_HIDDENCHANNEL,
        "scale_factor": SIDELENGTH_SCALE_FACTOR,
        "band_order": VIS_BAND_ORDER,
        "custom_normalize": custom_normalize,
        "image_mean": list(IMAGE_MEAN),
        "image_std": list(IMAGE_STD),
        "checkpoints": checkpoints,
        "training_timesteps": int(len(fg_beta)),
    }


def _generate_one(generator, mask, image, args, scene_seed, *, size=None):
    del scene_seed
    height, width = size if size is not None else mask.shape[-2:]
    return generator.main(
        int(height),
        int(width),
        prompt_str=args.object_label,
        obj_prompt_str=args.object_label,
        noobj_prompt_str=args.object_reference_label,
        FgGen_original_msk=mask if args.fggen_controlnet or args.imgsyn_controlnet else None,
        FgGen_conditional_element=mask if args.fggen_controlnet else None,
        ImgSyn_conditional_element=image if args.imgsyn_controlnet else None,
    )


def _generate_direct(runtime, mask, image, args, scene_seed, *, size=None):
    torch = runtime["torch"]
    tile_size = runtime["image_size"]
    height, width = size if size is not None else mask.shape[-2:]
    tiles = _tile_records(height, width, tile_size, tile_size, scene_seed, direct=True)
    foreground = torch.zeros(
        (1, runtime["input_channels"], height, width),
        dtype=torch.float32 if image is None else image.dtype,
        device=runtime["device"] if image is None else image.device,
    )
    generated = torch.zeros_like(foreground)
    generated_mask = torch.zeros(
        (1, 1, height, width),
        dtype=torch.float32 if mask is None else mask.dtype,
        device=runtime["device"] if mask is None else mask.device,
    )

    # Without background reference control, independent tiles can be sampled
    # in real batches while preserving a separate RNG stream for every tile.
    if not args.imgsyn_controlnet:
        from IMPGM_Utils import build_torch_generator

        generator = runtime["generator"]
        sampling_kwargs = generator._sampling_kwargs()
        latent_size = tile_size // runtime["scale_factor"]
        for batch_start in range(0, len(tiles), args.tile_batch_size):
            batch_tiles = tiles[batch_start : batch_start + args.tile_batch_size]
            mask_batch = torch.cat(
                [
                    mask[
                        :,
                        :,
                        tile["row_start"] : tile["row_end"],
                        tile["col_start"] : tile["col_end"],
                    ]
                    for tile in batch_tiles
                ],
                dim=0,
            ) if args.fggen_controlnet else None
            fg_generators = [
                build_torch_generator(tile["fggen_seed"], runtime["device"])
                for tile in batch_tiles
            ]
            img_generators = [
                build_torch_generator(tile["imgsyn_seed"], runtime["device"])
                for tile in batch_tiles
            ]
            fg_noise = torch.cat(
                [
                    torch.randn(
                        (1, runtime["latent_channels"], latent_size, latent_size),
                        device=runtime["device"],
                        generator=generator_item,
                    )
                    for generator_item in fg_generators
                ],
                dim=0,
            )
            img_noise = torch.cat(
                [
                    torch.randn(
                        (1, runtime["latent_channels"], latent_size, latent_size),
                        device=runtime["device"],
                        generator=generator_item,
                    )
                    for generator_item in img_generators
                ],
                dim=0,
            )
            if args.fggen_controlnet:
                prompts = [
                    args.object_label
                    if torch.sum(mask_batch[index]).item() >= 1.0
                    else args.object_reference_label
                    for index in range(len(batch_tiles))
                ]
            else:
                prompts = [args.object_label] * len(batch_tiles)
            if args.fggen_controlnet:
                fg_latent = generator.FgGen_CtrlNet_Sampler.forward(
                    fg_noise,
                    prompts,
                    is_record_process=False,
                    conditional_element=mask_batch,
                    generator=fg_generators,
                    **sampling_kwargs,
                )
            else:
                fg_latent = generator.FgGen_Sampler.forward(
                    fg_noise,
                    prompts,
                    is_record_process=False,
                    generator=fg_generators,
                    **sampling_kwargs,
                )
            img_latent = generator.ImgSyn_Sampler.forward(
                img_noise,
                prompts,
                is_record_process=False,
                fg_imgs_e=fg_latent,
                generator=img_generators,
                **sampling_kwargs,
            )
            foreground_batch = generator.VAE_model.decode(fg_latent)
            generated_batch = generator.VAE_model.decode(img_latent)
            generated_mask_batch = (
                torch.sigmoid(generator.FgSeg_model(foreground_batch)) >= 0.5
            ).float()
            for local_index, tile in enumerate(batch_tiles):
                row_start, row_end = tile["row_start"], tile["row_end"]
                col_start, col_end = tile["col_start"], tile["col_end"]
                foreground[:, :, row_start:row_end, col_start:col_end] = (
                    foreground_batch[local_index : local_index + 1]
                )
                generated[:, :, row_start:row_end, col_start:col_end] = (
                    generated_batch[local_index : local_index + 1]
                )
                generated_mask[:, :, row_start:row_end, col_start:col_end] = (
                    generated_mask_batch[local_index : local_index + 1]
                )
        return foreground, generated, generated_mask, tiles

    # Background-reference preprocessing follows the legacy per-tile path.
    from IMPGM_Utils import set_random_seed

    for tile in tiles:
        row_start, row_end = tile["row_start"], tile["row_end"]
        col_start, col_end = tile["col_start"], tile["col_end"]
        set_random_seed(tile["seed"], deterministic=False)
        fg_tile, image_tile, mask_tile = _generate_one(
            runtime["generator"],
            mask[:, :, row_start:row_end, col_start:col_end],
            image[:, :, row_start:row_end, col_start:col_end],
            args,
            tile["seed"],
        )
        foreground[:, :, row_start:row_end, col_start:col_end] = fg_tile
        generated[:, :, row_start:row_end, col_start:col_end] = image_tile
        generated_mask[:, :, row_start:row_end, col_start:col_end] = mask_tile
    return foreground, generated, generated_mask, tiles


def _generate_overlap(runtime, mask, image, args, scene_seed, *, size=None):
    from IMPGM_Utils import set_random_seed

    tile_size = runtime["image_size"]
    overlap = int(round(tile_size * args.overlap_rate))
    stride = tile_size - overlap
    height, width = size if size is not None else mask.shape[-2:]
    tiles = _tile_records(height, width, tile_size, stride, scene_seed, direct=False)
    set_random_seed(scene_seed, deterministic=False)
    foreground, generated, generated_mask = _generate_one(
        runtime["generator"], mask, image, args, scene_seed, size=size
    )
    return foreground, generated, generated_mask, tiles


def _validate_generation_geometry(scenes, args, runtime):
    tile_size = runtime["image_size"]
    if args.tile_size != tile_size:
        raise ValueError(
            f"--tile-size must match the trained model size {tile_size}, got {args.tile_size}."
        )
    if not 0.0 < args.overlap_rate < 1.0:
        raise ValueError("--overlap-rate must satisfy 0 < rate < 1.")
    overlap = int(round(tile_size * args.overlap_rate))
    if not np.isclose(overlap, tile_size * args.overlap_rate):
        raise ValueError("The requested overlap rate does not yield an integer pixel width.")
    if overlap % runtime["scale_factor"] != 0:
        raise ValueError("Overlap width must be divisible by the VAE scale factor.")
    if args.overlap_buffer <= 0 or args.overlap_buffer > overlap:
        raise ValueError("--overlap-buffer must be positive and no larger than overlap width.")
    if args.overlap_buffer % runtime["scale_factor"] != 0:
        raise ValueError("--overlap-buffer must be divisible by the VAE scale factor.")
    stride = tile_size if args.stitch_mode == "direct" else tile_size - overlap
    for scene in scenes:
        crop = scene["crop"]
        _tile_positions(int(crop["height"]), tile_size, stride)
        _tile_positions(int(crop["width"]), tile_size, stride)


def generate(args):
    manifest_path = Path(args.scene_manifest).expanduser().resolve()
    payload = _load_json(manifest_path)
    scenes = _validate_scene_manifest(payload, manifest_path)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise ValueError("--max-scenes must be positive.")
        scenes = scenes[: args.max_scenes]
    if args.mode != "single-condition":
        raise ValueError("Only single-condition large-image generation is supported.")
    if args.sampler == "ddim":
        if args.ddim_steps is None or args.ddim_eta is None:
            raise ValueError("DDIM requires --ddim-steps and --ddim-eta.")
    elif args.ddim_steps is not None or args.ddim_eta is not None:
        raise ValueError("DDIM parameters may only be used with --sampler ddim.")
    if args.tile_batch_size <= 0:
        raise ValueError("--tile-batch-size must be positive.")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_output = output_root / "generation_manifest.jsonl"
    if manifest_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Generation manifest already exists: {manifest_output}. Use --overwrite to replace it."
        )
    runtime = _load_runtime(args)
    _validate_generation_geometry(scenes, args, runtime)

    config = {
        "schema": "impgm-large-image-generation-config",
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "scene_manifest": str(manifest_path),
        "dataset_yaml": os.environ.get("IMPGM_DATASET_YAML"),
        "mode": args.mode,
        "stitch_mode": args.stitch_mode,
        "object_label": args.object_label,
        "object_reference_label": args.object_reference_label,
        "sampler": args.sampler,
        "training_timesteps": runtime["training_timesteps"],
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "tile_size": args.tile_size,
        "tile_batch_size": args.tile_batch_size,
        "overlap_rate": args.overlap_rate if args.stitch_mode == "overlap-aware" else 0.0,
        "overlap_width": (
            int(round(args.tile_size * args.overlap_rate))
            if args.stitch_mode == "overlap-aware"
            else 0
        ),
        "overlap_buffer": args.overlap_buffer if args.stitch_mode == "overlap-aware" else 0,
        "fggen_controlnet": args.fggen_controlnet,
        "imgsyn_controlnet": args.imgsyn_controlnet,
        "fggen_inpaint_resample": args.fggen_inpaint_resample,
        "imgsyn_inpaint_resample": args.imgsyn_inpaint_resample,
        "seed": args.seed,
        "normalization": {
            "mean": runtime["image_mean"],
            "std": runtime["image_std"],
            "visible_band_order": runtime["band_order"],
        },
        "rgb_preview_stretch_percentiles": [1.0, 99.0],
        "checkpoints": runtime["checkpoints"],
    }
    _write_json(output_root / "generation_config.json", config)
    manifest_output.write_text("", encoding="utf-8")

    from IMPGM_Utils import denormalize_image_tensor

    for scene_index, scene in enumerate(scenes):
        crop = scene["crop"]
        row = int(crop["row_start"])
        col = int(crop["col_start"])
        height = int(crop["height"])
        width = int(crop["width"])
        raw_image = _read_window(scene["source_image"], row, col, height, width)
        raw_mask = _mask_2d(
            _read_window(scene["source_mask"], row, col, height, width),
            scene["source_mask"],
        )
        normalized = _normalize_image(
            raw_image, runtime["input_channels"], runtime["custom_normalize"]
        )
        torch = runtime["torch"]
        image = torch.from_numpy(normalized).unsqueeze(0).to(
            runtime["device"], dtype=torch.float32
        )
        mask = torch.from_numpy(
            (np.isfinite(raw_mask) & (raw_mask > args.mask_threshold)).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(
            runtime["device"], dtype=torch.float32
        )
        scene_seed = int(args.seed + scene_index * 100000)

        if args.stitch_mode == "direct":
            foreground, generated, generated_mask, tiles = _generate_direct(
                runtime, mask, image, args, scene_seed
            )
        else:
            foreground, generated, generated_mask, tiles = _generate_overlap(
                runtime, mask, image, args, scene_seed
            )

        generated_raw = denormalize_image_tensor(generated).detach().cpu().numpy()[0]
        foreground_raw = denormalize_image_tensor(foreground).detach().cpu().numpy()[0]
        generated_mask_raw = generated_mask.detach().cpu().numpy()[0, 0]
        condition_mask_raw = mask.detach().cpu().numpy()[0, 0]
        projection = scene["projection_wkt"]
        geotransform = scene["geotransform"]
        scene_root = output_root / "scenes" / scene["scene_id"]
        paths = {
            "reference_tif": scene_root / "reference.tif",
            "condition_mask_tif": scene_root / "condition_mask.tif",
            "generated_foreground_tif": scene_root / "generated_foreground.tif",
            "generated_mask_tif": scene_root / "generated_mask.tif",
            "generated_tif": scene_root / "generated.tif",
            "generated_rgb": scene_root / "generated_rgb.png",
            "tile_metadata": scene_root / "tile_metadata.json",
            "scene_config": scene_root / "scene_config.json",
        }
        _write_tif(paths["reference_tif"], raw_image, projection, geotransform)
        _write_tif(paths["condition_mask_tif"], condition_mask_raw, projection, geotransform)
        _write_tif(paths["generated_foreground_tif"], foreground_raw, projection, geotransform)
        _write_tif(paths["generated_mask_tif"], generated_mask_raw, projection, geotransform)
        _write_tif(paths["generated_tif"], generated_raw, projection, geotransform)
        _save_rgb(paths["generated_rgb"], generated_raw, runtime["band_order"])

        tile_metadata = {
            "scene_id": scene["scene_id"],
            "stitch_mode": args.stitch_mode,
            "scene_seed": scene_seed,
            "rng_policy": (
                "independent_seed_per_tile"
                if args.stitch_mode == "direct"
                else "single_scene_rng_stream_in_diagonal_generation_order"
            ),
            "tile_size": args.tile_size,
            "stride": (
                args.tile_size
                if args.stitch_mode == "direct"
                else args.tile_size - int(round(args.tile_size * args.overlap_rate))
            ),
            "overlap_width": (
                0
                if args.stitch_mode == "direct"
                else int(round(args.tile_size * args.overlap_rate))
            ),
            "overlap_buffer": 0 if args.stitch_mode == "direct" else args.overlap_buffer,
            "tiles": tiles,
        }
        _write_json(paths["tile_metadata"], tile_metadata)
        _write_json(
            paths["scene_config"],
            {
                "schema": "impgm-large-image-scene-config",
                "schema_version": SCHEMA_VERSION,
                "scene": scene,
                "generation": config,
                "tiles": tile_metadata,
            },
        )
        record = {
            "schema": "impgm-large-image-generation-record",
            "schema_version": SCHEMA_VERSION,
            "scene_id": scene["scene_id"],
            "source_image": scene["source_image"],
            "source_mask": scene["source_mask"],
            "crop": crop,
            "target_ratio": scene["target_ratio"],
            "projection_wkt": projection,
            "geotransform": geotransform,
            "scene_seed": scene_seed,
            "stitch_mode": args.stitch_mode,
            "sampler": args.sampler,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            **{name: str(path.resolve()) for name, path in paths.items()},
        }
        with manifest_output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        print(f"[{scene_index + 1}/{len(scenes)}] Generated {scene['scene_id']}")

    print(f"Generation manifest: {manifest_output}")


def generate_autonomous(args):
    if args.height <= 0 or args.width <= 0 or args.num_scenes <= 0:
        raise ValueError("--height, --width, and --num-scenes must be positive.")
    if args.tile_batch_size <= 0:
        raise ValueError("--tile-batch-size must be positive.")
    if args.sampler == "ddim":
        if args.ddim_steps is None or args.ddim_eta is None:
            raise ValueError("DDIM requires --ddim-steps and --ddim-eta.")
        if args.ddim_steps <= 0 or not np.isfinite(args.ddim_eta) or args.ddim_eta < 0:
            raise ValueError("DDIM steps must be positive and eta must be finite and non-negative.")
    elif args.ddim_steps is not None or args.ddim_eta is not None:
        raise ValueError("DDIM parameters may only be used with --sampler ddim.")

    output_root = Path(args.output_root).expanduser().resolve()
    manifest_output = output_root / "generation_manifest.jsonl"
    if manifest_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Generation manifest already exists: {manifest_output}. Use --overwrite to replace it."
        )
    runtime = _load_runtime(args)
    _validate_generation_geometry(
        [{"crop": {"height": args.height, "width": args.width}}], args, runtime
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "impgm-large-image-autonomous-generation-config",
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "generation_path": "autonomous",
        "dataset_yaml": os.environ.get("IMPGM_DATASET_YAML"),
        **vars(args),
        "training_timesteps": runtime["training_timesteps"],
        "checkpoints": runtime["checkpoints"],
        "normalization": {
            "mean": runtime["image_mean"], "std": runtime["image_std"],
            "visible_band_order": runtime["band_order"],
        },
    }
    _write_json(output_root / "generation_config.json", config)
    manifest_output.write_text("", encoding="utf-8")
    from IMPGM_Utils import denormalize_image_tensor

    # Autonomous scenes have pixel coordinates, not a real-world location.
    projection = ""
    geotransform = [0, 1, 0, 0, 0, -1]
    generate_scene = _generate_direct if args.stitch_mode == "direct" else _generate_overlap
    for scene_index in range(args.num_scenes):
        scene_id = f"autonomous_{scene_index + 1:03d}"
        scene_seed = int(args.seed + scene_index * 100000)
        with runtime["torch"].no_grad():
            foreground, generated, generated_mask, tiles = generate_scene(
                runtime, None, None, args, scene_seed, size=(args.height, args.width)
            )
        expected_shape = (1, runtime["input_channels"], args.height, args.width)
        if tuple(generated.shape) != expected_shape or tuple(foreground.shape) != expected_shape:
            raise ValueError(f"Large-image output size differs from the requested shape {expected_shape}.")
        if tuple(generated_mask.shape) != (1, 1, args.height, args.width):
            raise ValueError("The generated mask must have one channel and match the requested image size.")
        generated_raw = denormalize_image_tensor(generated).detach().cpu().numpy()[0]
        foreground_raw = denormalize_image_tensor(foreground).detach().cpu().numpy()[0]
        mask_raw = generated_mask.detach().cpu().numpy()[0, 0]
        scene_root = output_root / "scenes" / scene_id
        paths = {
            "generated_foreground_tif": scene_root / "generated_foreground.tif",
            "generated_mask_tif": scene_root / "generated_mask.tif",
            "generated_tif": scene_root / "generated.tif",
            "generated_rgb": scene_root / "generated_rgb.png",
            "tile_metadata": scene_root / "tile_metadata.json",
        }
        _write_tif(paths["generated_foreground_tif"], foreground_raw, projection, geotransform)
        _write_tif(paths["generated_mask_tif"], mask_raw, projection, geotransform)
        _write_tif(paths["generated_tif"], generated_raw, projection, geotransform)
        _save_rgb(paths["generated_rgb"], generated_raw, runtime["band_order"])
        _write_json(paths["tile_metadata"], {
            "scene_id": scene_id, "scene_seed": scene_seed,
            "stitch_mode": args.stitch_mode, "tile_size": args.tile_size, "tiles": tiles,
            "rng_policy": "independent_seed_per_tile" if args.stitch_mode == "direct"
            else "single_scene_rng_stream_in_diagonal_generation_order",
        })
        record = {
            "schema": "impgm-large-image-autonomous-generation-record",
            "schema_version": SCHEMA_VERSION,
            "scene_id": scene_id, "scene_seed": scene_seed,
            "generation_path": "autonomous", "external_mask_used": False,
            "mask_source": "fgseg_generated", "object_label": args.object_label,
            "source_image": None, "source_mask": None,
            "reference_tif": None, "condition_mask_tif": None,
            "height": args.height, "width": args.width,
            "projection_wkt": projection, "geotransform": geotransform,
            "stitch_mode": args.stitch_mode, "sampler": args.sampler,
            "ddim_steps": args.ddim_steps, "ddim_eta": args.ddim_eta,
            **{name: str(path.resolve()) for name, path in paths.items()},
        }
        with manifest_output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{scene_index + 1}/{args.num_scenes}] Generated {scene_id}")
    print(f"Autonomous generation manifest: {manifest_output}")
    return manifest_output


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare fixed scenes and generate large IMPGM images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-scenes", help="Select deterministic large-image crop regions."
    )
    prepare.add_argument("--source-root", required=True)
    prepare.add_argument("--image-pattern", default="LargeImg_*/*_data.tif")
    prepare.add_argument("--mask-pattern", default="LargeImg_*/*_mask.tif")
    prepare.add_argument("--crop-size", type=int, default=1024)
    prepare.add_argument("--num-scenes", type=int, default=10)
    prepare.add_argument("--min-target-ratio", type=float, default=0.05)
    prepare.add_argument("--max-target-ratio", type=float, default=0.60)
    prepare.add_argument("--seed", type=int, default=999)
    prepare.add_argument("--output", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate direct-stitch or overlap-aware large images."
    )
    generate_parser.add_argument("--scene-manifest", required=True)
    generate_parser.add_argument(
        "--mode", choices=("single-condition",), default="single-condition"
    )
    generate_parser.add_argument(
        "--stitch-mode", choices=("direct", "overlap-aware"), required=True
    )
    generate_parser.add_argument(
        "--fggen-controlnet", action=argparse.BooleanOptionalAction, default=True
    )
    generate_parser.add_argument(
        "--imgsyn-controlnet", action=argparse.BooleanOptionalAction, default=False
    )
    generate_parser.add_argument("--mask-threshold", type=float, default=0.5)
    generate_parser.add_argument("--max-scenes", type=int, default=None)

    autonomous = subparsers.add_parser(
        "generate-autonomous", help="Generate large image-mask pairs without source scenes or input masks."
    )
    autonomous.add_argument("--height", type=int, default=1024)
    autonomous.add_argument("--width", type=int, default=1024)
    autonomous.add_argument("--num-scenes", type=int, default=10)
    autonomous.add_argument("--stitch-mode", choices=("direct", "overlap-aware"), default="overlap-aware")
    autonomous.set_defaults(fggen_controlnet=False, imgsyn_controlnet=False)

    for generation_parser in (generate_parser, autonomous):
        generation_parser.add_argument("--object-label", default="Water")
        generation_parser.add_argument("--object-reference-label", default="NoObj")
        generation_parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
        generation_parser.add_argument("--ddim-steps", type=int, default=None)
        generation_parser.add_argument("--ddim-eta", type=float, default=None)
        generation_parser.add_argument("--tile-size", type=int, default=256)
        generation_parser.add_argument("--tile-batch-size", type=int, default=8)
        generation_parser.add_argument("--overlap-rate", type=float, default=0.25)
        generation_parser.add_argument("--overlap-buffer", type=int, default=24)
        generation_parser.add_argument(
            "--fggen-inpaint-resample", action=argparse.BooleanOptionalAction, default=False
        )
        generation_parser.add_argument(
            "--imgsyn-inpaint-resample", action=argparse.BooleanOptionalAction, default=False
        )
        generation_parser.add_argument("--seed", type=int, default=999)
        generation_parser.add_argument("--output-root", required=True)
        generation_parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "prepare-scenes":
        prepare_scenes(
            source_root=args.source_root,
            image_pattern=args.image_pattern,
            mask_pattern=args.mask_pattern,
            crop_size=args.crop_size,
            num_scenes=args.num_scenes,
            min_target_ratio=args.min_target_ratio,
            max_target_ratio=args.max_target_ratio,
            seed=args.seed,
            output=args.output,
        )
        return
    if args.command == "generate-autonomous":
        generate_autonomous(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
