import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import (
    DATASET_NAME,
    DATASET_DICT,
    DATA_TRANSFER_NON_BLOCKING,
    DEVICE,
    INPUT_CHANNELS,
    MIN_LEARNING_RATE,
    NUM_WORKERS,
    PATIENCE_THRESHOLD_NUM,
    PIN_MEMORY,
    PROJECT_ROOT,
    PROMPT_DICT,
    all_train_transforms,
    transform_only_msk,
    transform_only_tif,
)
from IMPGM_Dataset import IMPGM_Dataset, build_unique_resolved_entries
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import (
    load_generation_manifest,
    summarize_replacement_reports,
    validate_replacement_manifest,
)
from IMPGM_TifReader import Tif_Read_and_Write
from IMPGM_Utils import (
    build_dataloader_generator,
    build_train_autocast,
    seed_dataloader_worker,
    set_random_seed,
    training_monitor,
)
from Quality_Evaluation.Downstream_UNet.Downstream_UNet_Config import (
    ARCHITECTURE_ID,
    CONDITION_CHANNELS,
    UNET_CHANNELS,
    number_of_classes,
)
from Quality_Evaluation.Downstream_UNet.Downstream_UNet_model import DownstreamUNet


REPLACEMENT_PROTOCOL_VERSION = 3


class RealSegmentationDataset(Dataset):
    """Expose normalized full images and masks from one real IMPGM split."""

    def __init__(self, split_name, condition_ids=None):
        suffix = {"train": "Train", "valid": "Valid", "test": "Test"}[split_name]
        self.base = IMPGM_Dataset(
            img_rootdir_list=list(DATASET_DICT[f"img_rootdir_list_for{suffix}"]),
            msk_rootdir_list=list(DATASET_DICT[f"msk_rootdir_list_for{suffix}"]),
            is_train=split_name == "train",
        )
        resolved_entries = build_unique_resolved_entries(self.base)
        entry_by_condition = {}
        for catalog_index, source_path in resolved_entries:
            condition_id = Path(source_path).stem
            if condition_id in entry_by_condition:
                raise ValueError(
                    f"Real {split_name} split contains duplicate condition_id {condition_id!r}."
                )
            entry_by_condition[condition_id] = catalog_index

        if condition_ids is None:
            self.indices = [catalog_index for catalog_index, _ in resolved_entries]
        else:
            if split_name != "train":
                raise ValueError("Condition matching is supported for the real train split only.")
            requested_ids = [str(condition_id) for condition_id in condition_ids]
            if len(requested_ids) != len(set(requested_ids)):
                raise ValueError("Replacement manifest contains duplicate condition IDs.")
            missing = sorted(set(requested_ids) - set(entry_by_condition))
            if missing:
                raise ValueError(
                    "Replacement manifest conditions are absent from the real train split: "
                    f"{missing[:10]}."
                )
            self.indices = [entry_by_condition[condition_id] for condition_id in requested_ids]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        label, _, mask, full_image, _, _ = self.base[self.indices[index]]
        if label not in PROMPT_DICT:
            raise KeyError(f"Unknown downstream real-sample label: {label!r}.")
        return full_image.float(), mask[:1].float(), int(PROMPT_DICT[label])


class SyntheticSegmentationDataset(Dataset):
    """Read primary train-split synthetic images from a generation manifest."""

    def __init__(self, records):
        self.records = [record for record in records if record.primary]
        if not self.records:
            raise ValueError("Synthetic downstream dataset has no primary samples.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, _, _ = Tif_Read_and_Write().Tif_Read(record.generated_tif)
        mask, _, _ = Tif_Read_and_Write().Tif_Read(record.condition_mask_tif)
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if image.ndim == 2:
            image = image[None]
        if mask.ndim == 2:
            mask = mask[None]
        image = np.clip(np.nan_to_num(image, nan=0.0), 0.0, 1.0)
        mask = (np.nan_to_num(mask, nan=0.0)[:1] > 0.5).astype(np.float32)
        image = np.transpose(image, (1, 2, 0))
        mask = mask[0]
        augmented = all_train_transforms(image=image, image0=mask)
        image, mask = augmented["image"], augmented["image0"]
        image = transform_only_tif(image=image)["image"]
        mask = transform_only_msk(image=mask)["image"]
        expected_label_id = PROMPT_DICT.get(record.label)
        if expected_label_id is None or int(expected_label_id) != int(record.label_id):
            raise ValueError(
                "Synthetic downstream record label metadata is inconsistent: "
                f"label={record.label!r}, label_id={record.label_id!r}."
            )
        return image.float(), mask[:1].float(), int(record.label_id)


def _dice_loss(probabilities, targets, eps=1.0e-8):
    probabilities = probabilities.float().flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _loss(logits, targets, bce):
    logits = logits.float()
    targets = targets.float()
    return 0.5 * bce(logits, targets) + 0.5 * _dice_loss(torch.sigmoid(logits), targets)


def _loader(dataset, batch_size, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(NUM_WORKERS),
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(seed),
    )


def _train_epoch(model, loader, optimizer, bce, microbatch_size):
    model.train()
    total = 0.0
    samples = 0
    for images, masks, label_ids in loader:
        batch_size = int(images.shape[0])
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, batch_size, int(microbatch_size)):
            end = min(start + int(microbatch_size), batch_size)
            images_mb = images[start:end].to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
            masks_mb = masks[start:end].to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
            label_ids_mb = label_ids[start:end].to(
                DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
            )
            with build_train_autocast():
                loss = _loss(model(images_mb, label_ids_mb), masks_mb, bce)
            (loss * ((end - start) / batch_size)).backward()
            total += float(loss.detach().item()) * (end - start)
            samples += end - start
        optimizer.step()
    return total / max(samples, 1)


@torch.no_grad()
def _evaluate(model, loader, bce):
    model.eval()
    loss_sum = 0.0
    samples = 0
    tp = fp = fn = 0.0
    for images, masks, label_ids in loader:
        images = images.to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
        masks = masks.to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
        label_ids = label_ids.to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
        with build_train_autocast():
            logits = model(images, label_ids)
        loss = _loss(logits, masks, bce)
        predictions = torch.sigmoid(logits.float()) >= 0.5
        targets = masks >= 0.5
        tp += (predictions & targets).sum().item()
        fp += (predictions & ~targets).sum().item()
        fn += (~predictions & targets).sum().item()
        batch_size = int(images.shape[0])
        loss_sum += float(loss.item()) * batch_size
        samples += batch_size
    eps = 1.0e-8
    miou = tp / (tp + fp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "loss": loss_sum / max(samples, 1),
        "miou": miou,
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
    }


def _fit_one_protocol(
    train_dataset,
    valid_dataset,
    test_dataset,
    seed,
    epochs,
    batch_size,
    microbatch_size,
    learning_rate,
):
    set_random_seed(int(seed), deterministic=False)
    model = DownstreamUNet(
        image_channels=INPUT_CHANNELS,
        num_classes=number_of_classes(),
        channels=UNET_CHANNELS,
        condition_channels=CONDITION_CHANNELS,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        threshold=0.0,
        patience=PATIENCE_THRESHOLD_NUM,
        min_lr=MIN_LEARNING_RATE,
    )
    bce = nn.BCEWithLogitsLoss()
    train_loader = _loader(train_dataset, batch_size, True, seed)
    valid_loader = _loader(valid_dataset, batch_size, False, seed)
    test_loader = _loader(test_dataset, batch_size, False, seed)
    best_state = None
    best_epoch = 0
    for epoch_idx in range(int(epochs)):
        epoch = epoch_idx + 1
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = _train_epoch(model, train_loader, optimizer, bce, microbatch_size)
        valid_metrics = _evaluate(model, valid_loader, bce)
        (
            is_break_training,
            is_savemodel,
            patience_counter,
            patience_counter_after_min_lr,
        ) = training_monitor(
            epoch_idx,
            MIN_LEARNING_RATE,
            current_lr,
            valid_metrics["loss"],
            PATIENCE_THRESHOLD_NUM,
            resume_train=False,
            load_dict=None,
        )
        scheduler.step(valid_metrics["loss"])
        print(
            f"[Downstream] Seed={seed}, Epoch={epoch}, Train={train_loss:.6f}, "
            f"Valid={valid_metrics['loss']:.6f}, mIoU={valid_metrics['miou']:.6f}, "
            f"Pc={patience_counter}, Pc_min_lr={patience_counter_after_min_lr}, "
            f"LR={current_lr:.8f}"
        )
        if is_savemodel:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if is_break_training:
            break
    if best_state is None:
        raise RuntimeError("Downstream training did not produce a valid model state.")
    model.load_state_dict(best_state)
    metrics = _evaluate(model, test_loader, bce)
    return {"seed": int(seed), "best_epoch": best_epoch, "metrics": metrics}


def _stable_hash(payload):
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_cached_real_report(path, expected_metadata):
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    mismatches = [
        f"{key}: cached={payload.get(key)!r}, expected={value!r}"
        for key, value in expected_metadata.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Cached real-only replacement baseline is incompatible: {path}\n  - "
            + "\n  - ".join(mismatches)
        )
    print(f"[Downstream Replacement] Reusing real-only baseline: {path}")
    return payload


def run_downstream_utility_evaluation(
    manifest_path,
    output_root,
    seed=17,
    epochs=100,
    batch_size=8,
    microbatch_size=1,
    learning_rate=1.0e-4,
    baseline_cache_root=None,
):
    """Compare equal-condition real-only and synthetic-only replacement training."""
    records = load_generation_manifest(manifest_path)
    manifest_datasets = {record.dataset for record in records}
    if manifest_datasets != {DATASET_NAME}:
        raise ValueError(
            "Downstream manifest dataset does not match the active dataset: "
            f"manifest={sorted(manifest_datasets)}, active={DATASET_NAME!r}."
        )
    if any(record.split != "train" for record in records):
        raise ValueError("Downstream synthetic replacement data must come from the train split.")
    real_valid = RealSegmentationDataset("valid")
    real_test = RealSegmentationDataset("test")
    heldout_paths = (
        list(real_valid.base.img_path_catalog) + list(real_test.base.img_path_catalog)
    )
    validate_replacement_manifest(records, heldout_paths)
    condition_ids = [record.condition_id for record in records]
    condition_signature = _stable_hash(sorted(condition_ids))
    real_train = RealSegmentationDataset("train", condition_ids=condition_ids)
    synthetic_train = SyntheticSegmentationDataset(records)
    if len(real_train) != len(synthetic_train):
        raise RuntimeError(
            "Real-only and synthetic-only replacement datasets must contain equal samples: "
            f"real={len(real_train)}, synthetic={len(synthetic_train)}."
        )
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
    baseline_cache_root = Path(
        baseline_cache_root
        or PROJECT_ROOT / "Evaluation" / "Downstream_Replacement_Baselines" / DATASET_NAME
    )
    baseline_cache_root.mkdir(parents=True, exist_ok=True)
    real_only_path = baseline_cache_root / (
        f"real_only_conditions_{condition_signature[:12]}_"
        f"train_{training_signature[:12]}.json"
    )
    common_metadata = {
        "dataset": DATASET_NAME,
        "seed": int(seed),
        "condition_signature": condition_signature,
        "num_train_samples": len(real_train),
        "training_config": training_config,
    }
    real_metadata = {**common_metadata, "protocol": "real_only"}
    real_report = _load_cached_real_report(real_only_path, real_metadata)
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
        real_report.update(real_metadata)
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
        **common_metadata,
        "protocol": "synthetic_only",
        "manifest": str(Path(manifest_path).resolve()),
        "prompt_classes": list(PROMPT_DICT),
    })
    synthetic_only_path = output_root / f"synthetic_only_seed_{int(seed)}.json"
    synthetic_only_path.write_text(
        json.dumps(synthetic_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = summarize_replacement_reports(real_only_path, synthetic_only_path)
    summary.update({
        "dataset": DATASET_NAME,
        "manifest": str(Path(manifest_path).resolve()),
        "condition_signature": condition_signature,
        "metric_protocol": (
            "single_seed_equal_condition_real_only_vs_synthetic_only_"
            "on_untouched_real_test"
        ),
    })
    summary_path = output_root / "downstream_replacement_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether generated data can replace matched real training data."
    )
    parser.add_argument("manifest", help="Train-split generation_manifest.jsonl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--baseline-cache-root", default=None)
    args = parser.parse_args()
    summary = run_downstream_utility_evaluation(
        args.manifest,
        args.output_root,
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
