"""Utilities and CLI helpers for peak training-memory benchmarks.

The recorder is intentionally embedded into each method's existing training
loop so the measured path includes its real losses, backward pass, optimizer
state, gradient handling, and EMA updates. Data loading happens before the
timed interval, while host-to-device transfer remains part of the training
step.
"""

from __future__ import annotations

import argparse
import csv
import json
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
METHOD_STAGES = {
    "spade": ("adversarial_training",),
    "rsguideddiffusion": ("diffusion_training",),
    "seg2sat": (
        "vae_training",
        "text_to_image_training",
        "controlnet_training",
    ),
    "impgm": (
        "vae_training",
        "fggen_diffusion_training",
        "fggen_controlnet_training",
        "imgsyn_diffusion_training",
    ),
}


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


def _count_unique_trainable_parameters(modules):
    seen = set()
    count = 0
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            count += int(parameter.numel())
    return count


def _tensor_shapes(value):
    try:
        import torch
    except ImportError:
        return None
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, dict):
        return {str(key): _tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_shapes(item) for item in value]
    return None


class TrainingMemoryRecorder:
    """Record a bounded sequence of complete optimizer iterations."""

    def __init__(
        self,
        *,
        method_key,
        stage,
        output,
        warmup_steps=5,
        measure_steps=20,
        batch_size=1,
        seed=999,
        overwrite=False,
        precision=None,
        notes=None,
    ):
        if method_key not in METHOD_STAGES:
            raise ValueError(f"Unsupported method: {method_key!r}")
        if stage not in METHOD_STAGES[method_key]:
            raise ValueError(
                f"Unsupported stage {stage!r} for {method_key}; expected one of "
                + ", ".join(METHOD_STAGES[method_key])
            )
        if int(warmup_steps) < 1:
            raise ValueError("At least one warm-up step is required to initialize optimizer state.")
        if int(measure_steps) < 1:
            raise ValueError("--measure-steps must be positive.")
        if int(batch_size) != 1:
            raise ValueError("The formal cross-model training-memory protocol requires batch size 1.")
        self.method_key = str(method_key)
        self.stage = str(stage)
        self.output = _resolve_path(output)
        self.warmup_steps = int(warmup_steps)
        self.measure_steps = int(measure_steps)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.overwrite = bool(overwrite)
        self.precision = precision
        self.notes = notes
        self.completed_steps = 0
        self.measured_step_seconds = []
        self._step_started_at = None
        self._measurement_started = False
        self._finished = False
        self._baseline_allocated = None
        self._baseline_reserved = None
        self._trainable_parameters = None
        self._batch_shapes = None
        self._cuda_device = None

    @property
    def total_steps(self):
        return self.warmup_steps + self.measure_steps

    @property
    def complete(self):
        return self.completed_steps >= self.total_steps

    def bind_trainable_modules(self, modules):
        import torch

        self._trainable_parameters = _count_unique_trainable_parameters(modules)
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.device.type == "cuda":
                    self._cuda_device = parameter.device
                    return
        if torch.cuda.is_available():
            self._cuda_device = torch.device("cuda", torch.cuda.current_device())

    def before_step(self, batch=None):
        import torch

        if self.complete:
            return False
        if not torch.cuda.is_available():
            raise RuntimeError("Training-memory evaluation requires CUDA.")
        if self._cuda_device is None:
            self._cuda_device = torch.device("cuda", torch.cuda.current_device())
        if self._batch_shapes is None and batch is not None:
            self._batch_shapes = _tensor_shapes(batch)
        if self.completed_steps == self.warmup_steps:
            torch.cuda.synchronize(self._cuda_device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._cuda_device)
            self._baseline_allocated = int(
                torch.cuda.memory_allocated(self._cuda_device)
            )
            self._baseline_reserved = int(
                torch.cuda.memory_reserved(self._cuda_device)
            )
            self._measurement_started = True
        if self._measurement_started:
            torch.cuda.synchronize(self._cuda_device)
            self._step_started_at = time.perf_counter()
        return True

    def after_step(self):
        import torch

        if self._measurement_started:
            torch.cuda.synchronize(self._cuda_device)
            elapsed = time.perf_counter() - self._step_started_at
            self.measured_step_seconds.append(float(elapsed))
            self._step_started_at = None
        self.completed_steps += 1
        if self.complete:
            self.finish()
            return True
        return False

    def finish(self):
        import torch

        if self._finished:
            return self.output
        if not self.complete:
            raise RuntimeError(
                f"Training-memory benchmark ended after {self.completed_steps} steps; "
                f"expected {self.total_steps}."
            )
        torch.cuda.synchronize(self._cuda_device)
        peak_allocated = int(torch.cuda.max_memory_allocated(self._cuda_device))
        peak_reserved = int(torch.cuda.max_memory_reserved(self._cuda_device))
        current_device = self._cuda_device.index
        if current_device is None:
            current_device = torch.cuda.current_device()
        current_device = int(current_device)
        properties = torch.cuda.get_device_properties(self._cuda_device)
        latencies = self.measured_step_seconds
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at": _utc_now(),
            "method_key": self.method_key,
            "stage": self.stage,
            "dataset_name": Path(
                os.environ.get("IMPGM_DATASET_YAML", "unknown")
            ).stem,
            "dataset_yaml": os.environ.get("IMPGM_DATASET_YAML"),
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
            "hardware": {
                "device_index": int(current_device),
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_total_memory_gib": int(properties.total_memory) / (1024 ** 3),
            },
            "protocol": {
                "warmup_steps": self.warmup_steps,
                "measure_steps": self.measure_steps,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "precision": self.precision,
                "timing_scope": (
                    "one complete optimization iteration; excludes data loading, "
                    "validation, visualization, checkpoint writing, and metrics"
                ),
                "memory_scope": (
                    "loaded training modules, activations, gradients, optimizer state, "
                    "and method-specific EMA modules retained by the formal training path"
                ),
                "cuda_synchronized": True,
                "notes": self.notes,
            },
            "trainable_parameters": self._trainable_parameters,
            "trainable_parameters_m": (
                self._trainable_parameters / 1_000_000.0
                if self._trainable_parameters is not None
                else None
            ),
            "batch_tensor_shapes": self._batch_shapes,
            "training": {
                "baseline_allocated_bytes": self._baseline_allocated,
                "baseline_allocated_gib": self._baseline_allocated / (1024 ** 3),
                "baseline_reserved_bytes": self._baseline_reserved,
                "baseline_reserved_gib": self._baseline_reserved / (1024 ** 3),
                "peak_allocated_bytes": peak_allocated,
                "peak_allocated_gib": peak_allocated / (1024 ** 3),
                "peak_reserved_bytes": peak_reserved,
                "peak_reserved_gib": peak_reserved / (1024 ** 3),
                "step_seconds": latencies,
                "mean_step_seconds": statistics.fmean(latencies),
                "std_step_seconds": (
                    statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                ),
            },
        }
        _atomic_write_json(self.output, result, overwrite=self.overwrite)
        self._finished = True
        print(f"[Training Memory] Result: {self.output}", flush=True)
        return self.output


def build_recorder_from_args(args, *, method_key, stage, precision=None, notes=None):
    return TrainingMemoryRecorder(
        method_key=method_key,
        stage=stage,
        output=args.training_memory_benchmark_output,
        warmup_steps=args.training_memory_warmup_steps,
        measure_steps=args.training_memory_measure_steps,
        batch_size=args.train_batch_size,
        seed=args.seed,
        overwrite=args.training_memory_overwrite,
        precision=precision,
        notes=notes,
    )


def add_training_memory_arguments(parser):
    parser.add_argument("--training_memory_benchmark_output", type=str, default="")
    parser.add_argument("--training_memory_warmup_steps", type=int, default=5)
    parser.add_argument("--training_memory_measure_steps", type=int, default=20)
    parser.add_argument("--training_memory_overwrite", action="store_true")


def summarize_training_memory(root, output, *, method_key=None, overwrite=False):
    root = _resolve_path(root)
    paths = sorted(root.glob("training_stages/*.json"))
    if not paths:
        raise FileNotFoundError(f"No training-stage JSON files found under {root}.")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    detected_methods = {result.get("method_key") for result in results}
    if method_key is None:
        if len(detected_methods) != 1:
            raise ValueError("Training-stage files contain more than one method.")
        method_key = next(iter(detected_methods))
    if method_key not in METHOD_STAGES:
        raise ValueError(f"Unsupported method: {method_key!r}")
    by_stage = {}
    reference = None
    for path, result in zip(paths, results):
        if result.get("status") != "complete":
            raise ValueError(f"Incomplete training-memory result: {path}")
        if result.get("method_key") != method_key:
            raise ValueError(f"Unexpected method in {path}: {result.get('method_key')}")
        stage = result.get("stage")
        if stage in by_stage:
            raise ValueError(f"Duplicate result for stage {stage!r}.")
        signature = {
            "gpu_name": result["hardware"]["gpu_name"],
            "pytorch": result["software"]["pytorch"],
            "cuda_runtime": result["software"]["cuda_runtime"],
            "warmup_steps": result["protocol"]["warmup_steps"],
            "measure_steps": result["protocol"]["measure_steps"],
            "batch_size": result["protocol"]["batch_size"],
            "seed": result["protocol"]["seed"],
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            mismatch = [key for key in reference if reference[key] != signature[key]]
            raise ValueError(
                f"Training-memory stages are not comparable ({path}): "
                + ", ".join(mismatch)
            )
        by_stage[stage] = {"source": str(path.resolve()), "result": result}
    required = METHOD_STAGES[method_key]
    missing = [stage for stage in required if stage not in by_stage]
    if missing:
        raise ValueError("Missing required training stages: " + ", ".join(missing))
    rows = []
    for stage in required:
        result = by_stage[stage]["result"]
        training = result["training"]
        rows.append({
            "method_key": method_key,
            "stage": stage,
            "trainable_parameters_m": result.get("trainable_parameters_m"),
            "peak_training_memory_gib": training["peak_allocated_gib"],
            "peak_training_reserved_memory_gib": training["peak_reserved_gib"],
            "mean_step_seconds": training["mean_step_seconds"],
            "std_step_seconds": training["std_step_seconds"],
        })
    max_row = max(rows, key=lambda row: row["peak_training_memory_gib"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "method_key": method_key,
        "protocol_signature": reference,
        "required_stages": list(required),
        "max_stage": max_row["stage"],
        "max_stage_peak_training_memory_gib": max_row[
            "peak_training_memory_gib"
        ],
        "stage_rows": rows,
        "stage_results": by_stage,
    }
    output = _resolve_path(output)
    _atomic_write_json(output, payload, overwrite=overwrite)
    _atomic_write_csv(
        output.with_suffix(".csv"),
        rows,
        fieldnames=list(rows[0].keys()),
        overwrite=overwrite,
    )
    print(f"[Training Memory] Summary: {output}", flush=True)
    return output


def _run_stage(args):
    dataset_yaml = _resolve_path(args.dataset_yaml)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_yaml}")
    os.environ["IMPGM_DATASET_YAML"] = str(dataset_yaml)
    if args.stage not in METHOD_STAGES[args.method]:
        raise ValueError(
            f"Unsupported stage {args.stage!r} for {args.method}; expected one of "
            + ", ".join(METHOD_STAGES[args.method])
        )

    common = {
        "output": _resolve_path(args.output),
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "overwrite": args.overwrite,
    }
    if args.method == "spade":
        from Comparison_Models import bind_comparison_model_yaml

        bind_comparison_model_yaml(PROJECT_ROOT, "spade.yaml")
        from Comparison_Models.SPADE.IMPGM_SPADE import benchmark_training_memory

        return benchmark_training_memory(**common)
    if args.method == "rsguideddiffusion":
        from Comparison_Models import bind_comparison_model_yaml

        bind_comparison_model_yaml(PROJECT_ROOT, "rsguideddiffusion.yaml")
        from Comparison_Models.RSGuidedDiffusion.IMPGM_RSGuidedDiffusion import (
            benchmark_training_memory,
        )

        return benchmark_training_memory(**common)
    if args.method == "seg2sat":
        from Comparison_Models import bind_comparison_model_yaml

        bind_comparison_model_yaml(PROJECT_ROOT, "seg2sat.yaml")
        if args.stage == "vae_training":
            from Comparison_Models.Seg2Sat.IMPGM_Seg2Sat_VAE import (
                benchmark_training_memory,
            )

            return benchmark_training_memory(**common)
        from Comparison_Models.Seg2Sat.IMPGM_Seg2Sat import (
            benchmark_seg2sat_training_memory,
        )

        return benchmark_seg2sat_training_memory(stage=args.stage, **common)
    if args.stage == "vae_training":
        from VAE.VAE_Code.VAE_train import benchmark_training_memory
    elif args.stage == "fggen_diffusion_training":
        from FgGen.FgGen_Code.FgGen_Diffusion_Train import (
            benchmark_training_memory,
        )
    elif args.stage == "fggen_controlnet_training":
        from FgGen.FgGen_Code.FgGen_ControlNet_Train import (
            benchmark_training_memory,
        )
    else:
        from ImgSyn.ImgSyn_Code.ImgSyn_Diffusion_Train import (
            benchmark_training_memory,
        )
    return benchmark_training_memory(**common)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Measure peak CUDA memory for one formal training stage."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--method", choices=tuple(METHOD_STAGES))
    action.add_argument("--summarize", metavar="METHOD_RESULT_ROOT")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--dataset-yaml", default="configs/datasets/main.yaml")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.summarize:
        if args.stage is not None:
            parser.error("--stage is not used with --summarize.")
        summarize_training_memory(
            args.summarize,
            args.output,
            overwrite=args.overwrite,
        )
        return
    if args.stage is None:
        parser.error("--stage is required with --method.")
    if args.batch_size != 1:
        parser.error("The formal protocol requires --batch-size 1.")
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("Warm-up and measured step counts must be positive.")
    _run_stage(args)


if __name__ == "__main__":
    main()
