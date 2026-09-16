"""Benchmark model complexity and inference efficiency for IMPGM comparisons.

The benchmark deliberately reuses each method's formal model loader and
generation function. Disk input, image output, manifest writing, and quality
metric computation are outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCHEMA_VERSION = 1
METHOD_KEYS = ("spade", "rsguideddiffusion", "seg2sat", "impgm")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(path_like, *, base=PROJECT_ROOT):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _atomic_write_json(path, payload, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _atomic_write_csv(path, rows, fieldnames, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path_metadata(path_like, *, include_hash=True):
    path = _resolve_path(path_like)
    record = {
        "path": str(path),
        "exists": path.exists(),
        "type": "missing",
    }
    if not path.exists():
        return record
    stat = path.stat()
    record["modified_time"] = datetime.fromtimestamp(
        stat.st_mtime, timezone.utc
    ).isoformat()
    if path.is_file():
        record.update({
            "type": "file",
            "size_bytes": int(stat.st_size),
        })
        if include_hash:
            print(f"[Efficiency] SHA-256: {path}", flush=True)
            record["sha256"] = _sha256_file(path)
    elif path.is_dir():
        record["type"] = "directory"
    return record


def _count_parameters(module, *, trainable_only=False):
    return sum(
        int(parameter.numel())
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _count_unique_parameters(modules, *, trainable_only=False):
    seen = set()
    count = 0
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if trainable_only and not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            count += int(parameter.numel())
    return count


def _parameter_dtype_report(modules):
    report = {}
    seen = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            key = str(parameter.dtype).replace("torch.", "")
            report[key] = report.get(key, 0) + int(parameter.numel())
    return report


def _parameter_report(active_modules, training_stages):
    component_counts = {
        name: _count_parameters(module)
        for name, module in active_modules.items()
        if module is not None
    }
    active_count = _count_unique_parameters(active_modules.values())
    stage_counts = {}
    for stage_name, modules in training_stages.items():
        stage_counts[stage_name] = _count_unique_parameters(modules)
    max_stage_count = max(stage_counts.values(), default=0)
    return {
        "inference_parameters": active_count,
        "inference_parameters_m": active_count / 1_000_000.0,
        "inference_components": component_counts,
        "inference_component_parameters_m": {
            key: value / 1_000_000.0 for key, value in component_counts.items()
        },
        "parameter_dtypes": _parameter_dtype_report(active_modules.values()),
        "training_stage_parameters": stage_counts,
        "training_stage_parameters_m": {
            key: value / 1_000_000.0 for key, value in stage_counts.items()
        },
        "max_stage_trainable_parameters": max_stage_count,
        "max_stage_trainable_parameters_m": max_stage_count / 1_000_000.0,
    }


def _percentile(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty sequence.")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(percentile) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _latency_statistics(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("No latency measurements were recorded.")
    return {
        "sample_count": len(values),
        "mean_seconds": statistics.fmean(values),
        "std_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_seconds": statistics.median(values),
        "p95_seconds": _percentile(values, 95.0),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "throughput_samples_per_second": (
            len(values) / sum(values) if sum(values) > 0 else None
        ),
    }


def _output_shapes(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _output_shapes(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_output_shapes(item) for item in value]
    return type(value).__name__


def _validate_output(value):
    import torch

    tensors = []

    def collect(item):
        if torch.is_tensor(item):
            tensors.append(item)
        elif isinstance(item, dict):
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)

    collect(value)
    if not tensors:
        raise TypeError("The generation adapter returned no tensor output.")
    for tensor in tensors:
        if tensor.numel() == 0:
            raise ValueError("The generation adapter returned an empty tensor.")
        if not torch.isfinite(tensor).all():
            raise ValueError("The generation adapter returned NaN or Inf values.")


class ForwardCallCounter:
    def __init__(self, modules):
        self.counts = {name: 0 for name in modules}
        self.handles = []
        for name, module in modules.items():
            if module is None:
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def record(_module, _inputs, _output):
            self.counts[name] += 1

        return record

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class BaseAdapter:
    method_key = None
    display_name = None

    def __init__(self, args):
        self.args = args
        self.device = None
        self.dataset_name = None
        self.dataset = None
        self.active_modules = {}
        self.training_stages = {}
        self.checkpoint_paths = {}
        self.config_paths = {}

    def load(self):
        raise NotImplementedError

    def sample_catalog(self):
        raise NotImplementedError

    def prepare(self, dataset_index):
        raise NotImplementedError

    def generate(self, prepared, seed):
        raise NotImplementedError

    def forward_counter_modules(self):
        raise NotImplementedError

    def primary_nfe_components(self):
        raise NotImplementedError

    def sampling_config(self):
        return {}

    def parameter_report(self):
        return _parameter_report(self.active_modules, self.training_stages)


class SPADEAdapter(BaseAdapter):
    method_key = "spade"

    def load(self):
        from Comparison_Models import bind_comparison_model_yaml

        model_yaml = bind_comparison_model_yaml(PROJECT_ROOT, "spade.yaml")
        import Comparison_Models.SPADE.IMPGM_SPADE as model_module
        from Comparison_Models.IMPGM_Strict_Comparison_Dataset import (
            IMPGMStrictSPADEDataset,
        )

        self.module = model_module
        self.display_name = str(model_module.COMPARISON_MODEL_DISPLAY_NAME)
        self.dataset_name = str(model_module.DATASET_NAME)
        self.device = model_module.DEVICE
        self.dataset = IMPGMStrictSPADEDataset(self.args.split, is_train=False)
        self.generator = model_module._load_eval_generator()
        discriminator = model_module.MultiscaleDiscriminator(
            model_module._build_discriminator_options()
        )
        self.active_modules = {"generator": self.generator}
        self.training_stages = {
            "adversarial_training": [self.generator, discriminator],
        }
        self._training_only_modules = [discriminator]
        checkpoint = Path(model_module.SPADE_CHECKPOINT_PATH)
        self.checkpoint_paths = {"generator": str(checkpoint)}
        self.config_paths = {"model_yaml": str(model_yaml)}

    def sample_catalog(self):
        return [
            (index, Path(source).stem)
            for index, (_, source) in enumerate(self.dataset.resolved_entries)
        ]

    def prepare(self, dataset_index):
        sample = self.dataset[int(dataset_index)]
        return sample["label_map"].unsqueeze(0).to(self.device)

    def generate(self, prepared, seed):
        del seed
        with self.module.build_train_autocast():
            return self.generator(self.module._build_semantics(prepared))

    def forward_counter_modules(self):
        return {"generator": self.generator}

    def primary_nfe_components(self):
        return ("generator",)

    def sampling_config(self):
        return {
            "sampler": "single_forward",
            "sampling_steps": 1,
            "deterministic": True,
        }


class RSGuidedDiffusionAdapter(BaseAdapter):
    method_key = "rsguideddiffusion"

    def load(self):
        from Comparison_Models import bind_comparison_model_yaml

        model_yaml = bind_comparison_model_yaml(PROJECT_ROOT, "rsguideddiffusion.yaml")
        import Comparison_Models.RSGuidedDiffusion.IMPGM_RSGuidedDiffusion as model_module
        from Comparison_Models.IMPGM_Strict_Comparison_Dataset import (
            IMPGMStrictPixelDiffusionDataset,
        )

        self.module = model_module
        self.display_name = str(model_module.COMPARISON_MODEL_DISPLAY_NAME)
        self.dataset_name = str(model_module.DATASET_NAME)
        self.device = model_module.DEVICE
        self.dataset = IMPGMStrictPixelDiffusionDataset(
            self.args.split, is_train=False
        )
        self.model = model_module._load_eval_model()
        self.noise_scheduler = model_module.build_noise_scheduler()
        self.num_inference_steps = 1000
        self.active_modules = {"diffusion_unet": self.model}
        self.training_stages = {"diffusion_training": [self.model]}
        checkpoint = Path(model_module.STRICT_CHECKPOINT_PATH)
        self.checkpoint_paths = {"diffusion_unet": str(checkpoint)}
        self.config_paths = {"model_yaml": str(model_yaml)}

    def sample_catalog(self):
        return [
            (index, Path(source).stem)
            for index, (_, source) in enumerate(self.dataset.resolved_entries)
        ]

    def prepare(self, dataset_index):
        sample = self.dataset[int(dataset_index)]
        return sample["seg_all"].unsqueeze(0).to(self.device)

    def generate(self, prepared, seed):
        return self.module.generate_predictions(
            self.model,
            self.noise_scheduler,
            prepared,
            microbatch_size=1,
            num_inference_steps=self.num_inference_steps,
            seeds=[int(seed)],
        )

    def forward_counter_modules(self):
        return {"diffusion_unet": self.model}

    def primary_nfe_components(self):
        return ("diffusion_unet",)

    def sampling_config(self):
        return {
            "sampler": str(self.module.STRICT_MODEL_TYPE).lower(),
            "sampling_steps": self.num_inference_steps,
        }


class Seg2SatAdapter(BaseAdapter):
    method_key = "seg2sat"

    def load(self):
        from Comparison_Models import bind_comparison_model_yaml

        model_yaml = bind_comparison_model_yaml(PROJECT_ROOT, "seg2sat.yaml")
        import Comparison_Models.Seg2Sat.IMPGM_Seg2Sat as model_module
        from Comparison_Models.IMPGM_Strict_Comparison_Dataset import (
            IMPGMStrictSeg2SatDataset,
        )

        self.module = model_module
        self.display_name = str(model_module.COMPARISON_MODEL_DISPLAY_NAME)
        self.dataset_name = str(model_module.DATASET_NAME)
        self.device = model_module.DEVICE
        self.dataset = IMPGMStrictSeg2SatDataset(self.args.split, is_train=False)
        (
            self.pipe,
            self.vae,
            self.scaling_factor,
            checkpoints,
        ) = model_module.load_seg2sat_eval_pipeline()
        self.active_modules = {
            "vae": self.pipe.vae,
            "text_encoder": self.pipe.text_encoder,
            "diffusion_unet": self.pipe.unet,
            "controlnet": self.pipe.controlnet,
        }
        self.training_stages = {
            "vae_training": [self.pipe.vae],
            "text_to_image_training": [self.pipe.unet],
            "controlnet_training": [self.pipe.controlnet],
        }
        self.checkpoint_paths = checkpoints
        self.config_paths = {"model_yaml": str(model_yaml)}

    def sample_catalog(self):
        return [
            (index, Path(source).stem)
            for index, (_, source) in enumerate(self.dataset.resolved_entries)
        ]

    def prepare(self, dataset_index):
        sample = self.dataset[int(dataset_index)]
        condition = sample["conditioning_pixel_values"].unsqueeze(0).to(self.device)
        return {
            "prompt": sample["prompt_text"],
            "condition": condition,
        }

    def generate(self, prepared, seed):
        generator = self.module.build_torch_generator(int(seed), self.device)
        return self.module.generate_seg2sat_predictions(
            self.pipe,
            self.vae,
            self.scaling_factor,
            prompts=[prepared["prompt"]],
            conditioning_images=prepared["condition"],
            generators=[generator],
        )

    def forward_counter_modules(self):
        modules = {
            "diffusion_unet": self.pipe.unet,
            "controlnet": self.pipe.controlnet,
        }
        if getattr(self.pipe.vae, "decoder", None) is not None:
            modules["vae_decoder"] = self.pipe.vae.decoder
        if getattr(self.pipe, "text_encoder", None) is not None:
            modules["text_encoder"] = self.pipe.text_encoder
        return modules

    def primary_nfe_components(self):
        return ("diffusion_unet",)

    def sampling_config(self):
        scheduler_name = self.pipe.scheduler.__class__.__name__
        return {
            "sampler": scheduler_name,
            "sampling_steps": int(
                self.module.STRICT_SEG2SAT_NUM_INFERENCE_STEPS
            ),
            "guidance_scale": float(self.module.STRICT_SEG2SAT_GUIDANCE_SCALE),
            "controlnet_conditioning_scale": float(
                self.module.STRICT_SEG2SAT_CONTROLNET_CONDITIONING_SCALE
            ),
        }


class IMPGMAdapter(BaseAdapter):
    method_key = "impgm"

    def load(self):
        import ImgSyn.ImgSyn_Code.IMPGM_Full_Pipeline_Inference as model_module

        self.module = model_module
        self.display_name = "IMPGM"
        self.dataset_name = str(model_module.DATASET_NAME)
        self.device = model_module.DEVICE
        self.dataset = model_module._build_dataset(self.args.split)
        self.entries = model_module._resolved_evaluation_entries(self.dataset)
        self.sampler_mode = str(
            self.args.sampler or model_module.DRAW_SAMPLER_MODE
        ).lower()
        self.sampling_metadata = model_module.resolve_full_pipeline_sampling_config(
            self.sampler_mode,
            ddim_steps=self.args.ddim_steps,
            ddim_eta=self.args.ddim_eta,
        )
        self.vae, self.fg_model, self.img_model = model_module._build_models()
        self.active_modules = {
            "shared_vae": self.vae,
            "fggen_diffusion": self.fg_model.FgGen_Diffusion_model,
            "fggen_controlnet": self.fg_model.ControlNet_model,
            "imgsyn_diffusion": self.img_model,
        }
        self.training_stages = {
            "vae_training": [self.vae],
            "fggen_diffusion_training": [self.fg_model.FgGen_Diffusion_model],
            "fggen_controlnet_training": [self.fg_model.ControlNet_model],
            "imgsyn_diffusion_training": [self.img_model],
        }
        self.checkpoint_paths = {
            "vae": str(model_module.VAE_MODEL_SAVEPATH),
            "fggen_diffusion": str(model_module.FGGEN_BASE_DIFFUSION_MODEL_SAVEPATH),
            "fggen_controlnet": str(
                model_module.FGGEN_CONTROLNET_CONFIG.MODEL_SAVEPATH
            ),
            "imgsyn_diffusion": str(
                model_module.IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH
            ),
        }
        self.config_paths = {
            "dataset_yaml": str(os.environ["IMPGM_DATASET_YAML"]),
            "fggen_controlnet_yaml": str(
                PROJECT_ROOT / "configs/tasks/fggen_controlnet.yaml"
            ),
            "imgsyn_diffusion_yaml": str(
                PROJECT_ROOT / "configs/tasks/imgsyn_diffusion.yaml"
            ),
        }

    def sample_catalog(self):
        return [
            (catalog_index, Path(source).stem)
            for catalog_index, source in self.entries
        ]

    def prepare(self, dataset_index):
        label, _, mask, _, _, _ = self.dataset[int(dataset_index)]
        if not hasattr(mask, "to"):
            raise TypeError("IMPGM efficiency evaluation requires tensor masks.")
        return {
            "label": str(label),
            "mask": mask.to(self.device),
        }

    def generate(self, prepared, seed):
        return self.module._generate_batch(
            self.vae,
            self.fg_model,
            self.img_model,
            labels=[prepared["label"]],
            masks=[prepared["mask"]],
            sampler_mode=self.sampler_mode,
            seeds=[int(seed)],
            ddim_steps=self.args.ddim_steps,
            ddim_eta=self.args.ddim_eta,
        )

    def forward_counter_modules(self):
        modules = {
            "fggen_denoiser": self.fg_model,
            "imgsyn_denoiser": self.img_model,
        }
        if getattr(self.vae, "encoder", None) is not None:
            modules["vae_encoder"] = self.vae.encoder
        if getattr(self.vae, "decoder", None) is not None:
            modules["vae_decoder"] = self.vae.decoder
        return modules

    def primary_nfe_components(self):
        return ("fggen_denoiser", "imgsyn_denoiser")

    def sampling_config(self):
        method_name, scheduler_tag, scheduler_description = (
            self.module._resolve_full_pipeline_scheduler_identity()
        )
        del method_name
        return {
            **self.sampling_metadata,
            "scheduler": scheduler_tag,
            "scheduler_description": scheduler_description,
            "training_diffusion_steps_per_stage": int(
                self.module.FGGEN_CONTROLNET_CONFIG.STEPS
            ),
            "fggen_sampling_steps": self.sampling_metadata[
                "actual_fggen_steps"
            ],
            "imgsyn_sampling_steps": self.sampling_metadata[
                "actual_imgsyn_steps"
            ],
            "sampling_steps": self.sampling_metadata["actual_total_steps"],
        }


def _build_adapter(method_key, args):
    adapters = {
        "spade": SPADEAdapter,
        "rsguideddiffusion": RSGuidedDiffusionAdapter,
        "seg2sat": Seg2SatAdapter,
        "impgm": IMPGMAdapter,
    }
    return adapters[method_key](args)


def _rank_catalog(catalog, seed):
    unique = {}
    for dataset_index, sample_id in catalog:
        sample_id = str(sample_id)
        if sample_id in unique:
            raise ValueError(f"Duplicate condition ID in benchmark dataset: {sample_id}")
        unique[sample_id] = int(dataset_index)
    return sorted(
        [(index, sample_id) for sample_id, index in unique.items()],
        key=lambda item: hashlib.sha256(
            f"{int(seed)}:{item[1]}".encode("utf-8")
        ).digest(),
    )


def _select_samples(catalog, num_samples, warmup_samples, seed):
    ranked = _rank_catalog(catalog, seed)
    required = int(num_samples) + int(warmup_samples)
    if len(ranked) < required:
        raise ValueError(
            f"Benchmark requires {required} unique conditions "
            f"({warmup_samples} warm-up + {num_samples} measured), but only "
            f"{len(ranked)} are available."
        )
    warmup = ranked[: int(warmup_samples)]
    measured = ranked[int(warmup_samples):required]
    digest = hashlib.sha256()
    for _, sample_id in measured:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return warmup, measured, digest.hexdigest()


def _software_hardware_report(torch, device):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            f"Efficiency evaluation requires a CUDA device, got {device}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the active PyTorch environment.")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    cudnn_version = torch.backends.cudnn.version()
    return {
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": cudnn_version,
        },
        "hardware": {
            "device": str(device),
            "device_index": int(device_index),
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_total_memory_gib": int(properties.total_memory) / (1024 ** 3),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        },
    }


def _measure_forward_calls(adapter, sample, seed, torch):
    counter = ForwardCallCounter(adapter.forward_counter_modules())
    try:
        with torch.inference_mode():
            prepared = adapter.prepare(sample[0])
            output = adapter.generate(prepared, seed)
            torch.cuda.synchronize()
            _validate_output(output)
            output_shapes = _output_shapes(output)
            del output, prepared
    finally:
        counter.close()
    primary_components = adapter.primary_nfe_components()
    nfe = sum(counter.counts.get(name, 0) for name in primary_components)
    if nfe <= 0:
        raise RuntimeError(
            "No primary generation-network forwards were recorded; check the adapter hooks."
        )
    return {
        "component_forward_calls_per_sample": counter.counts,
        "primary_nfe_components": list(primary_components),
        "nfe_per_sample": int(nfe),
        "output_shapes": output_shapes,
    }


def _run_warmup(adapter, samples, seed, torch):
    for position, (dataset_index, sample_id) in enumerate(samples, start=1):
        prepared = adapter.prepare(dataset_index)
        with torch.inference_mode():
            output = adapter.generate(prepared, int(seed) + position - 1)
        torch.cuda.synchronize()
        _validate_output(output)
        del output, prepared
        print(
            f"[Efficiency] warm-up {position}/{len(samples)}: {sample_id}",
            flush=True,
        )


def _run_timed_repeats(adapter, samples, repeats, seed, torch):
    all_latencies = []
    repeat_records = []
    output_shapes = None
    for repeat_index in range(int(repeats)):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        repeat_latencies = []
        for sample_position, (dataset_index, sample_id) in enumerate(samples):
            prepared = adapter.prepare(dataset_index)
            sample_seed = int(seed) + sample_position
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.inference_mode():
                output = adapter.generate(prepared, sample_seed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            if output_shapes is None:
                output_shapes = _output_shapes(output)
            repeat_latencies.append(elapsed)
            all_latencies.append(elapsed)
            del output, prepared
            print(
                f"[Efficiency] {adapter.method_key} repeat "
                f"{repeat_index + 1}/{repeats}, sample "
                f"{sample_position + 1}/{len(samples)}: {sample_id}, "
                f"{elapsed:.4f} s",
                flush=True,
            )
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        repeat_records.append({
            "repeat": repeat_index + 1,
            "latency": _latency_statistics(repeat_latencies),
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_gib": peak_allocated / (1024 ** 3),
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_gib": peak_reserved / (1024 ** 3),
        })
    return {
        "latency": _latency_statistics(all_latencies),
        "repeats": repeat_records,
        "peak_allocated_bytes": max(
            record["peak_allocated_bytes"] for record in repeat_records
        ),
        "peak_allocated_gib": max(
            record["peak_allocated_gib"] for record in repeat_records
        ),
        "peak_reserved_bytes": max(
            record["peak_reserved_bytes"] for record in repeat_records
        ),
        "peak_reserved_gib": max(
            record["peak_reserved_gib"] for record in repeat_records
        ),
        "output_shapes": output_shapes,
    }


def run_method(args):
    dataset_yaml = _resolve_path(args.dataset_yaml)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_yaml}")
    os.environ["IMPGM_DATASET_YAML"] = str(dataset_yaml)

    import torch

    adapter = _build_adapter(args.method, args)
    print(f"[Efficiency] Loading {args.method}...", flush=True)
    adapter.load()
    environment = _software_hardware_report(torch, adapter.device)
    warmup, measured, selection_hash = _select_samples(
        adapter.sample_catalog(),
        args.num_samples,
        args.warmup_samples,
        args.seed,
    )
    parameter_report = adapter.parameter_report()
    checkpoints = {
        name: _path_metadata(path, include_hash=not args.skip_checkpoint_hash)
        for name, path in adapter.checkpoint_paths.items()
    }

    torch.cuda.synchronize()
    if warmup:
        forward_sample = warmup[0]
    else:
        forward_sample = measured[0]
    forward_report = _measure_forward_calls(
        adapter, forward_sample, args.seed, torch
    )
    sampling_report = adapter.sampling_config()
    expected_nfe = sampling_report.get("actual_total_steps")
    if expected_nfe is not None:
        measured_nfe = int(forward_report["nfe_per_sample"])
        if measured_nfe != int(expected_nfe):
            raise RuntimeError(
                "Measured NFE does not match the resolved sampling schedule: "
                f"measured={measured_nfe}, expected={int(expected_nfe)}."
            )
        sampling_report["nfe_matches_sampling_steps"] = True
    remaining_warmup = warmup[1:] if warmup else []
    _run_warmup(adapter, remaining_warmup, args.seed + 1, torch)
    timed_report = _run_timed_repeats(
        adapter, measured, args.repeats, args.seed, torch
    )

    output_path = _single_method_output_path(args)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "method_key": adapter.method_key,
        "method_display_name": adapter.display_name,
        "dataset_name": adapter.dataset_name,
        "dataset_yaml": str(dataset_yaml),
        "split": args.split,
        "config_paths": adapter.config_paths,
        "checkpoints": checkpoints,
        **environment,
        "protocol": {
            "num_samples": int(args.num_samples),
            "warmup_samples": int(args.warmup_samples),
            "repeats": int(args.repeats),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "timing_scope": (
                "model_compute_only; excludes disk input, output writing, manifest "
                "writing, and quality metrics"
            ),
            "cuda_synchronized": True,
            "condition_transfer_outside_timed_region": True,
            "sample_selection_sha256": selection_hash,
            "measured_condition_ids": [sample_id for _, sample_id in measured],
            "warmup_condition_ids": [sample_id for _, sample_id in warmup],
            **sampling_report,
        },
        "parameters": parameter_report,
        "forward_calls": forward_report,
        "inference": timed_report,
    }
    _atomic_write_json(output_path, result, overwrite=args.overwrite)
    print(f"[Efficiency] Result: {output_path}", flush=True)
    return output_path


def _summary_row(result, training_memory_result=None):
    parameters = result["parameters"]
    inference = result["inference"]
    latency = inference["latency"]
    forward_calls = result["forward_calls"]
    protocol = result["protocol"]
    return {
        "method_key": result["method_key"],
        "method_display_name": result["method_display_name"],
        "inference_parameters_m": parameters["inference_parameters_m"],
        "max_stage_trainable_parameters_m": parameters[
            "max_stage_trainable_parameters_m"
        ],
        "peak_training_memory_gib": (
            training_memory_result["max_stage_peak_training_memory_gib"]
            if training_memory_result is not None
            else None
        ),
        "peak_training_memory_stage": (
            training_memory_result["max_stage"]
            if training_memory_result is not None
            else None
        ),
        "peak_inference_memory_gib": inference["peak_allocated_gib"],
        "latency_mean_seconds": latency["mean_seconds"],
        "latency_std_seconds": latency["std_seconds"],
        "latency_median_seconds": latency["median_seconds"],
        "latency_p95_seconds": latency["p95_seconds"],
        "throughput_samples_per_second": latency[
            "throughput_samples_per_second"
        ],
        "nfe_per_sample": forward_calls["nfe_per_sample"],
        "sampler": protocol.get("sampler"),
        "sampling_steps": protocol.get("sampling_steps"),
        "num_samples": protocol["num_samples"],
        "repeats": protocol["repeats"],
        "gpu_name": result["hardware"]["gpu_name"],
    }


def _comparison_signature(result):
    protocol = result["protocol"]
    return {
        "dataset_name": result.get("dataset_name"),
        "dataset_yaml": result.get("dataset_yaml"),
        "split": result.get("split"),
        "sample_selection_sha256": protocol.get("sample_selection_sha256"),
        "num_samples": protocol.get("num_samples"),
        "warmup_samples": protocol.get("warmup_samples"),
        "repeats": protocol.get("repeats"),
        "batch_size": protocol.get("batch_size"),
        "seed": protocol.get("seed"),
        "gpu_name": result.get("hardware", {}).get("gpu_name"),
        "pytorch": result.get("software", {}).get("pytorch"),
        "cuda_runtime": result.get("software", {}).get("cuda_runtime"),
    }


def summarize_results(root, output, *, overwrite=False):
    root = _resolve_path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Efficiency result root does not exist: {root}")
    result_paths = sorted(root.glob("*/efficiency_metrics.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No per-method efficiency_metrics.json files were found under {root}."
        )
    methods = {}
    rows = []
    missing_training_memory_methods = []
    benchmark_signature = None
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported result schema in {result_path}")
        if result.get("status") != "complete":
            raise ValueError(f"Incomplete efficiency result: {result_path}")
        current_signature = _comparison_signature(result)
        if benchmark_signature is None:
            benchmark_signature = current_signature
        elif current_signature != benchmark_signature:
            differing_fields = [
                key
                for key in benchmark_signature
                if benchmark_signature[key] != current_signature[key]
            ]
            raise ValueError(
                "Efficiency results are not directly comparable. Mismatched "
                f"protocol/environment fields in {result_path}: "
                + ", ".join(differing_fields)
            )
        method_key = str(result["method_key"])
        if method_key in methods:
            raise ValueError(f"Duplicate efficiency result for {method_key!r}.")
        training_result_path = (
            root / method_key / "training_efficiency_metrics.json"
        )
        training_result = None
        if training_result_path.is_file():
            training_result = json.loads(
                training_result_path.read_text(encoding="utf-8")
            )
            if int(training_result.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported training-memory schema in {training_result_path}"
                )
            if training_result.get("status") != "complete":
                raise ValueError(
                    f"Incomplete training-memory result: {training_result_path}"
                )
            if training_result.get("method_key") != method_key:
                raise ValueError(
                    "Training-memory method does not match its directory: "
                    f"{training_result_path}"
                )
            training_gpu = training_result.get("protocol_signature", {}).get(
                "gpu_name"
            )
            if training_gpu != result["hardware"]["gpu_name"]:
                raise ValueError(
                    "Training and inference memory were measured on different GPUs "
                    f"for {method_key}: {training_gpu!r} versus "
                    f"{result['hardware']['gpu_name']!r}."
                )
        else:
            missing_training_memory_methods.append(method_key)
        methods[method_key] = {
            "source": str(result_path.resolve()),
            "result": result,
            "training_memory_source": (
                str(training_result_path.resolve())
                if training_result is not None
                else None
            ),
            "training_memory_result": training_result,
        }
        rows.append(_summary_row(result, training_result))
    rows.sort(key=lambda row: METHOD_KEYS.index(row["method_key"]))
    missing_methods = [key for key in METHOD_KEYS if key not in methods]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete"
            if not missing_methods and not missing_training_memory_methods
            else "partial"
        ),
        "created_at": _utc_now(),
        "root": str(root),
        "included_methods": [row["method_key"] for row in rows],
        "missing_methods": missing_methods,
        "missing_training_memory_methods": missing_training_memory_methods,
        "benchmark_signature": benchmark_signature,
        "table_rows": rows,
        "method_results": methods,
    }
    output = _resolve_path(output)
    csv_output = output.with_suffix(".csv")
    _atomic_write_json(output, summary, overwrite=overwrite)
    _atomic_write_csv(
        csv_output,
        rows,
        fieldnames=list(rows[0].keys()),
        overwrite=overwrite,
    )
    print(f"[Efficiency] Summary JSON: {output}", flush=True)
    print(f"[Efficiency] Summary CSV:  {csv_output}", flush=True)
    if missing_methods:
        print(
            "[Efficiency] Partial summary; missing: " + ", ".join(missing_methods),
            flush=True,
        )
    if missing_training_memory_methods:
        print(
            "[Efficiency] Missing training-memory summaries: "
            + ", ".join(missing_training_memory_methods),
            flush=True,
        )
    return output


def _default_output_root(args):
    dataset_name = Path(args.dataset_yaml).stem
    return PROJECT_ROOT / "Evaluation" / "Model_Efficiency" / dataset_name


def _single_method_output_path(args):
    if args.output:
        return _resolve_path(args.output)
    if args.method == "impgm" and args.sampler == "ddim":
        dataset_name = Path(args.dataset_yaml).stem
        return (
            PROJECT_ROOT
            / "Evaluation"
            / "Sampling_Strategy_Efficiency"
            / dataset_name
            / f"ddim_{int(args.ddim_steps)}"
            / "efficiency_metrics.json"
        )
    output_root = (
        _resolve_path(args.output_root)
        if args.output_root
        else _default_output_root(args)
    )
    return output_root / args.method / "efficiency_metrics.json"


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure parameter count, CUDA peak memory, end-to-end model "
            "latency, and NFE for IMPGM comparison methods."
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--method",
        choices=METHOD_KEYS,
        help="Evaluate one method.",
    )
    action.add_argument(
        "--summarize",
        metavar="RESULT_ROOT",
        help="Summarize existing per-method efficiency results without loading models.",
    )
    parser.add_argument(
        "--dataset-yaml",
        default="configs/datasets/main.yaml",
        help="Dataset YAML loaded before any project model modules are imported.",
    )
    parser.add_argument(
        "--split", choices=("train", "valid", "test", "draw"), default="test"
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--sampler",
        choices=("ddpm", "ddim"),
        default=None,
        help="Optional IMPGM sampler override; omitted means the formal configured sampler.",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=None,
        help="Requested DDIM steps per IMPGM generation stage.",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=None,
        help="DDIM stochasticity coefficient (use 0.0 for deterministic sampling).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root for per-method results; defaults to Evaluation/Model_Efficiency/<dataset>.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Single-method or summary JSON path.",
    )
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Skip SHA-256 for checkpoint files when faster startup is preferred.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(parser, args):
    if args.summarize:
        if args.output_root:
            parser.error("--output-root is not used with --summarize.")
        if args.sampler or args.ddim_steps is not None or args.ddim_eta is not None:
            parser.error("Sampler options are not used with --summarize.")
        return
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive.")
    if args.warmup_samples < 0:
        parser.error("--warmup-samples must be non-negative.")
    if args.repeats <= 0:
        parser.error("--repeats must be positive.")
    if args.batch_size != 1:
        parser.error(
            "The formal efficiency protocol requires --batch-size 1 so latency, "
            "memory, and NFE remain directly comparable."
        )
    if args.method != "impgm" and (
        args.sampler or args.ddim_steps is not None or args.ddim_eta is not None
    ):
        parser.error("Sampler overrides are only supported for --method impgm.")
    if args.sampler == "ddim":
        if args.ddim_steps is None or args.ddim_eta is None:
            parser.error("DDIM requires both --ddim-steps and --ddim-eta.")
        if args.ddim_steps <= 0:
            parser.error("--ddim-steps must be positive.")
        if args.ddim_steps > 1000:
            parser.error("--ddim-steps must not exceed 1000.")
        if args.ddim_eta < 0.0:
            parser.error("--ddim-eta must be non-negative.")
    elif args.ddim_steps is not None or args.ddim_eta is not None:
        parser.error("--ddim-steps and --ddim-eta require --sampler ddim.")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if args.summarize:
        root = _resolve_path(args.summarize)
        output = (
            _resolve_path(args.output)
            if args.output
            else root / "model_efficiency_summary.json"
        )
        summarize_results(root, output, overwrite=args.overwrite)
    else:
        run_method(args)


if __name__ == "__main__":
    main()
