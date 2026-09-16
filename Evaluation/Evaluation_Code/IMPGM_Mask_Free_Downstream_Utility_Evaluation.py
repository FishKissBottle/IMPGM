import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import (
    DATASET_NAME,
    MIN_LEARNING_RATE,
    PATIENCE_THRESHOLD_NUM,
    PROJECT_ROOT,
    PROMPT_DICT,
)
from IMPGM_Dataset import build_unique_resolved_entries
from Evaluation.Evaluation_Code.IMPGM_Downstream_Utility_Evaluation import (
    REPLACEMENT_PROTOCOL_VERSION,
    RealSegmentationDataset,
    SyntheticSegmentationDataset,
    _fit_one_protocol,
    _load_cached_real_report,
    _stable_hash,
)
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import (
    load_generation_manifest,
    resolve_condition_selection,
)
from Evaluation.Evaluation_Code.IMPGM_Mask_Free_Evaluation import (
    MASK_FREE_METHOD,
    validate_mask_free_generation_protocol,
)
from Quality_Evaluation.Downstream_UNet.Downstream_UNet_Config import (
    ARCHITECTURE_ID,
    CONDITION_CHANNELS,
    UNET_CHANNELS,
    number_of_classes,
)


def _label_from_path(path):
    tokens = Path(path).stem.split("_")
    if len(tokens) < 2:
        raise ValueError(f"Cannot derive a class label from {path}.")
    label = tokens[-2]
    if label not in PROMPT_DICT:
        raise KeyError(f"Unknown class label {label!r} in {path}.")
    return label


def _select_real_train_conditions(sample_fraction, selection_seed):
    full_train = RealSegmentationDataset("train")
    resolved_entries = build_unique_resolved_entries(full_train.base)
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
        split="train",
        sample_fraction=sample_fraction,
        seed=selection_seed,
    )
    candidate_by_index = {candidate["index"]: candidate for candidate in candidates}
    selected = [candidate_by_index[index] for index in selected_indices]
    condition_ids = [item["condition_id"] for item in selected]
    class_counts = Counter(item["label"] for item in selected)
    return condition_ids, class_counts, selection_path, effective_fraction


def validate_mask_free_replacement_manifest(records, expected_class_counts):
    validate_mask_free_generation_protocol(records)
    if any(record.split != "train" for record in records):
        raise ValueError("Mask-free downstream manifests may contain train records only.")
    if any(not record.primary or int(record.seed_rank) != 0 for record in records):
        raise ValueError(
            "Mask-free downstream replacement requires one primary rank-0 sample per pair."
        )
    condition_ids = [record.condition_id for record in records]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("Mask-free downstream manifest contains duplicate pair IDs.")
    if any(record.reference_source_image is not None for record in records):
        raise ValueError(
            "Mask-free train records may not retain a real sample as a generation reference."
        )
    observed_class_counts = Counter(record.label for record in records)
    if observed_class_counts != Counter(expected_class_counts):
        raise ValueError(
            "Mask-free synthetic and real-only class counts differ: "
            f"synthetic={dict(sorted(observed_class_counts.items()))}, "
            f"real={dict(sorted(Counter(expected_class_counts).items()))}."
        )
    return {
        "status": "valid",
        "num_samples": len(records),
        "per_class_counts": dict(sorted(observed_class_counts.items())),
        "matching_rule": "equal_total_and_per_class_counts_without_condition_id_pairing",
    }


def _summarize_mask_free_reports(real_path, synthetic_path):
    real = json.loads(Path(real_path).read_text(encoding="utf-8"))
    synthetic = json.loads(Path(synthetic_path).read_text(encoding="utf-8"))
    if int(real["seed"]) != int(synthetic["seed"]):
        raise ValueError("Real-only and mask-free synthetic reports use different seeds.")
    if int(real["num_train_samples"]) != int(synthetic["num_train_samples"]):
        raise ValueError("Real-only and mask-free synthetic reports use different sample counts.")
    if real.get("per_class_counts") != synthetic.get("per_class_counts"):
        raise ValueError("Real-only and mask-free synthetic reports use different class counts.")
    metric_names = ("miou", "dice", "f1", "precision", "recall")
    real_metrics = {name: float(real["metrics"][name]) for name in metric_names}
    synthetic_metrics = {
        name: float(synthetic["metrics"][name]) for name in metric_names
    }
    return {
        "seed": int(real["seed"]),
        "num_train_samples_per_protocol": int(real["num_train_samples"]),
        "per_class_counts": real["per_class_counts"],
        "real_only": real_metrics,
        "synthetic_only": synthetic_metrics,
        "synthetic_minus_real": {
            name: synthetic_metrics[name] - real_metrics[name]
            for name in metric_names
        },
        "real_only_report": str(Path(real_path).resolve()),
        "synthetic_only_report": str(Path(synthetic_path).resolve()),
    }


def run_mask_free_downstream_utility_evaluation(
    manifest_path,
    output_root,
    *,
    real_sample_fraction=0.15,
    selection_seed=999,
    seed=17,
    epochs=100,
    batch_size=8,
    microbatch_size=8,
    learning_rate=1.0e-4,
    baseline_cache_root=None,
):
    records = load_generation_manifest(manifest_path)
    manifest_datasets = {record.dataset for record in records}
    if manifest_datasets != {DATASET_NAME}:
        raise ValueError(
            "Mask-free downstream manifest dataset does not match the active dataset: "
            f"manifest={sorted(manifest_datasets)}, active={DATASET_NAME!r}."
        )
    if {record.method for record in records} != {MASK_FREE_METHOD}:
        raise ValueError(f"Expected mask-free method {MASK_FREE_METHOD!r}.")

    real_condition_ids, real_class_counts, selection_path, effective_fraction = (
        _select_real_train_conditions(real_sample_fraction, int(selection_seed))
    )
    manifest_validation = validate_mask_free_replacement_manifest(
        records, real_class_counts
    )
    real_train = RealSegmentationDataset("train", condition_ids=real_condition_ids)
    synthetic_train = SyntheticSegmentationDataset(records)
    if len(real_train) != len(synthetic_train):
        raise RuntimeError(
            "Mask-free Real-only and Syn-only datasets must contain equal samples."
        )
    real_valid = RealSegmentationDataset("valid")
    real_test = RealSegmentationDataset("test")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    training_config = {
        "protocol_version": REPLACEMENT_PROTOCOL_VERSION,
        "architecture": ARCHITECTURE_ID,
        "unet_channels": list(UNET_CHANNELS),
        "condition_channels": int(CONDITION_CHANNELS),
        "num_classes": number_of_classes(),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "microbatch_size": int(microbatch_size),
        "learning_rate": float(learning_rate),
        "min_learning_rate": float(MIN_LEARNING_RATE),
        "scheduler_patience": int(PATIENCE_THRESHOLD_NUM),
        "early_stopping": "after_min_lr",
    }
    training_signature = _stable_hash(training_config)
    real_condition_signature = _stable_hash(sorted(real_condition_ids))
    synthetic_pair_signature = _stable_hash(sorted(record.condition_id for record in records))
    per_class_counts = dict(sorted(real_class_counts.items()))
    baseline_cache_root = Path(
        baseline_cache_root
        or PROJECT_ROOT / "Evaluation" / "Downstream_Replacement_Baselines" / DATASET_NAME
    )
    baseline_cache_root.mkdir(parents=True, exist_ok=True)
    real_only_path = baseline_cache_root / (
        f"real_only_conditions_{real_condition_signature[:12]}_"
        f"train_{training_signature[:12]}.json"
    )
    common_real_metadata = {
        "dataset": DATASET_NAME,
        "seed": int(seed),
        "condition_signature": real_condition_signature,
        "num_train_samples": len(real_train),
        "training_config": training_config,
        "protocol": "real_only",
    }
    real_report = _load_cached_real_report(real_only_path, common_real_metadata)
    if real_report is None:
        real_report = _fit_one_protocol(
            real_train,
            real_valid,
            real_test,
            seed,
            epochs,
            batch_size,
            microbatch_size,
            learning_rate,
        )
        real_report.update({
            **common_real_metadata,
            "per_class_counts": per_class_counts,
            "selection_path": str(selection_path),
            "selection_fraction": float(effective_fraction),
            "selection_seed": int(selection_seed),
        })
        real_only_path.write_text(
            json.dumps(real_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif "per_class_counts" not in real_report:
        real_report["per_class_counts"] = per_class_counts
        real_report["selection_path"] = str(selection_path)
        real_report["selection_fraction"] = float(effective_fraction)
        real_report["selection_seed"] = int(selection_seed)
        real_only_path.write_text(
            json.dumps(real_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    synthetic_report = _fit_one_protocol(
        synthetic_train,
        real_valid,
        real_test,
        seed,
        epochs,
        batch_size,
        microbatch_size,
        learning_rate,
    )
    synthetic_report.update({
        "dataset": DATASET_NAME,
        "seed": int(seed),
        "num_train_samples": len(synthetic_train),
        "training_config": training_config,
        "protocol": "mask_free_synthetic_only",
        "manifest": str(Path(manifest_path).resolve()),
        "per_class_counts": per_class_counts,
        "synthetic_pair_signature": synthetic_pair_signature,
        "real_reference_condition_signature": real_condition_signature,
        "prompt_classes": list(PROMPT_DICT),
    })
    synthetic_only_path = output_root / f"mask_free_synthetic_only_seed_{int(seed)}.json"
    synthetic_only_path.write_text(
        json.dumps(synthetic_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = _summarize_mask_free_reports(real_only_path, synthetic_only_path)
    summary.update({
        "dataset": DATASET_NAME,
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_validation": manifest_validation,
        "real_reference_condition_signature": real_condition_signature,
        "synthetic_pair_signature": synthetic_pair_signature,
        "selection_path": str(selection_path),
        "selection_fraction": float(effective_fraction),
        "selection_seed": int(selection_seed),
        "metric_protocol": (
            "single_seed_equal_class_count_real_only_vs_mask_free_synthetic_only_"
            "on_untouched_real_test"
        ),
    })
    summary_path = output_root / "mask_free_downstream_replacement_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether autonomous IMPGM image-mask pairs can replace an "
            "equal-size class-matched real training subset."
        )
    )
    parser.add_argument("manifest", help="Mask-free train generation_manifest.jsonl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--real-sample-fraction", type=float, default=0.15)
    parser.add_argument("--selection-seed", type=int, default=999)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--baseline-cache-root", default=None)
    args = parser.parse_args()
    summary = run_mask_free_downstream_utility_evaluation(
        args.manifest,
        args.output_root,
        real_sample_fraction=args.real_sample_fraction,
        selection_seed=args.selection_seed,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        learning_rate=args.learning_rate,
        baseline_cache_root=args.baseline_cache_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
