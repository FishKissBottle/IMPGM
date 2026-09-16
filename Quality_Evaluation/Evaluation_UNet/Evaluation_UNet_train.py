import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from IMPGM_Config import (
    ADAM_BETAS,
    DATASET_DICT,
    DATASET_NAME,
    DATA_TRANSFER_NON_BLOCKING,
    DEVICE,
    DRAW_BATCH_SIZE,
    DRAW_INTERVAL_EPOCHS,
    EMA_DECAY,
    FGSEG_DRAW_MICROBATCH_SIZE,
    FGSEG_TEST_MICROBATCH_SIZE,
    FGSEG_TRAIN_MICROBATCH_SIZE,
    FGSEG_VALID_MICROBATCH_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_CHANNELS,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    NUM_WORKERS,
    PATIENCE_THRESHOLD_NUM,
    PIN_MEMORY,
    RANDOM_SEED,
    TEST_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
    TRAIN_MAX_EPOCHS,
    TRAIN_MAX_ITERATIONS,
    VALID_BATCH_SIZE,
)
from IMPGM_Utils import (
    build_dataloader_generator,
    build_ema_model,
    build_train_autocast,
    get_training_monitor_state,
    iter_microbatch_slices,
    load_model,
    load_model_for_eval,
    log_info,
    prepare_rgb_vis_tensor,
    resolve_microbatch_size,
    save_model,
    save_msk_datas,
    save_rgb_datas,
    seed_dataloader_worker,
    set_random_seed,
    training_monitor,
    update_ema_model,
)
from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_Config import (
    BASE_CHANNELS,
    BCE_WEIGHT,
    CONDITION_CHANNELS,
    DICE_WEIGHT,
    LAST_MODEL_PATH,
    LOG_DIR,
    MODEL_PATH,
    RGB_DIR,
    TEST_INFO_PATH,
    THRESHOLD,
    TRAINING_INFO_PATH,
    ensure_artifact_directories,
    number_of_classes,
)
from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_Dataset import (
    EvaluationUNetDataset,
)
from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_model import EvaluationUNet


def build_evaluation_unet():
    return EvaluationUNet(
        image_channels=INPUT_CHANNELS,
        num_classes=number_of_classes(),
        base_channels=BASE_CHANNELS,
        condition_channels=CONDITION_CHANNELS,
    )


def _split_roots(split_name):
    suffix = {
        "train": "Train",
        "valid": "Valid",
        "test": "Test",
        "draw": "Draw",
    }[split_name]
    image_key = f"img_rootdir_list_for{suffix}"
    mask_key = f"msk_rootdir_list_for{suffix}"
    if image_key not in DATASET_DICT or mask_key not in DATASET_DICT:
        raise KeyError(
            f"Evaluation_UNet requires explicit {split_name} roots: "
            f"{image_key}, {mask_key}."
        )
    return list(DATASET_DICT[image_key]), list(DATASET_DICT[mask_key])


def build_loader(split_name, batch_size, shuffle=False, seed=RANDOM_SEED):
    image_roots, mask_roots = _split_roots(split_name)
    dataset = EvaluationUNetDataset(
        image_roots,
        mask_roots,
        is_train=(split_name == "train"),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(seed),
    )


def dice_loss(probabilities, targets):
    """Compute per-sample Dice loss in FP32."""
    with torch.amp.autocast(probabilities.device.type, enabled=False):
        probabilities = probabilities.float()
        targets = targets.float()
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        return (1.0 - (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)).mean()


def compute_losses(logits, targets, bce):
    targets = (targets >= 0.5).float()
    bce_value = bce(logits.float(), targets)
    probabilities = torch.sigmoid(logits.float())
    dice_value = dice_loss(probabilities, targets)
    total = BCE_WEIGHT * bce_value + DICE_WEIGHT * dice_value
    return total, bce_value, dice_value, probabilities, targets


def _metric_state():
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0}


def _update_metrics(state, probabilities, targets):
    predictions = probabilities >= THRESHOLD
    targets = targets >= 0.5
    state["tp"] += (predictions & targets).sum().item()
    state["fp"] += (predictions & ~targets).sum().item()
    state["fn"] += (~predictions & targets).sum().item()


def _finalize_metrics(state):
    tp, fp, fn = state["tp"], state["fp"], state["fn"]
    eps = 1.0e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
    }


def train_epoch(loader, model, optimizer, bce, ema_model, training_state):
    model.train()
    totals = {"loss": 0.0, "bce_loss": 0.0, "dice_loss": 0.0}
    sample_count = 0
    with tqdm(loader, colour="#3a86ff", leave=False, ncols=120) as progress:
        for batch in progress:
            images = batch["image"]
            masks = batch["mask"]
            label_ids = batch["label_id"]
            batch_size = int(images.shape[0])
            microbatch_size = resolve_microbatch_size(
                FGSEG_TRAIN_MICROBATCH_SIZE, batch_size
            )
            optimizer.zero_grad(set_to_none=True)
            batch_totals = {key: 0.0 for key in totals}
            for start, end in iter_microbatch_slices(batch_size, microbatch_size):
                count = end - start
                images_mb = images[start:end].to(
                    DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
                )
                masks_mb = masks[start:end].to(
                    DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
                )
                labels_mb = label_ids[start:end].to(
                    DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
                )
                with build_train_autocast():
                    logits = model(images_mb, labels_mb)
                    loss, bce_value, dice_value, _, _ = compute_losses(
                        logits, masks_mb, bce
                    )
                (loss * (count / batch_size)).backward()
                batch_totals["loss"] += float(loss.item()) * count
                batch_totals["bce_loss"] += float(bce_value.item()) * count
                batch_totals["dice_loss"] += float(dice_value.item()) * count
            optimizer.step()
            update_ema_model(ema_model, model, EMA_DECAY)
            for key in totals:
                totals[key] += batch_totals[key]
            sample_count += batch_size
            training_state["iteration"] += 1
            progress.set_postfix(loss=f"{batch_totals['loss'] / batch_size:.6f}")
            if TRAIN_MAX_ITERATIONS > 0 and training_state["iteration"] >= TRAIN_MAX_ITERATIONS:
                training_state["max_iter_reached"] = True
                break
    if sample_count == 0:
        raise RuntimeError("Evaluation_UNet training loader is empty.")
    return {key: value / sample_count for key, value in totals.items()}


@torch.no_grad()
def evaluate_loader(loader, model, bce, microbatch_size):
    model.eval()
    totals = {"loss": 0.0, "bce_loss": 0.0, "dice_loss": 0.0}
    state = _metric_state()
    sample_count = 0
    for batch in loader:
        batch_size = int(batch["image"].shape[0])
        chunk = resolve_microbatch_size(microbatch_size, batch_size)
        for start, end in iter_microbatch_slices(batch_size, chunk):
            count = end - start
            images = batch["image"][start:end].to(
                DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
            )
            masks = batch["mask"][start:end].to(
                DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
            )
            labels = batch["label_id"][start:end].to(
                DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
            )
            with build_train_autocast():
                logits = model(images, labels)
            loss, bce_value, dice_value, probabilities, targets = compute_losses(
                logits, masks, bce
            )
            totals["loss"] += float(loss.item()) * count
            totals["bce_loss"] += float(bce_value.item()) * count
            totals["dice_loss"] += float(dice_value.item()) * count
            sample_count += count
            _update_metrics(state, probabilities, targets)
    if sample_count == 0:
        raise RuntimeError("Evaluation_UNet evaluation loader is empty.")
    result = {key: value / sample_count for key, value in totals.items()}
    result.update(_finalize_metrics(state))
    return result


@torch.no_grad()
def save_draw_snapshot(loader, model, epoch_num):
    batch = next(iter(loader))
    images = batch["image"].to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
    labels = batch["label_id"].to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
    with build_train_autocast():
        logits = model(images, labels)
    predictions = (torch.sigmoid(logits.float()) >= THRESHOLD).float().cpu()
    if epoch_num == 1:
        save_rgb_datas(
            prepare_rgb_vis_tensor(batch["image"]).cpu(),
            nrow=3,
            savepath=str(RGB_DIR / "Evaluation_UNet_Full_Image.png"),
        )
        save_msk_datas(
            batch["mask"],
            nrow=3,
            savepath=str(RGB_DIR / "Evaluation_UNet_Target_Mask.png"),
        )
    save_msk_datas(
        predictions,
        nrow=3,
        savepath=str(RGB_DIR / f"Evaluation_UNet_Predicted_Mask_{epoch_num}.png"),
    )


def _save_checkpoint(path, epoch_num, model, ema_model, optimizer, scheduler, results, state):
    save_model(
        path,
        epoch_num,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_dict={
            "train_loss": results["train"]["loss"],
            "valid_loss": results["valid"]["loss"],
            "min_loss": state["best_valid_loss"],
        },
        extra={
            "ema_model_state_dict": ema_model.state_dict(),
            "ema_decay": EMA_DECAY,
            "best_epoch_num": state["best_epoch_num"],
            "patience_counter": state["patience_counter"],
            "patience_counter_after_min_lr": state["patience_counter_after_min_lr"],
            "global_iteration": state["iteration"],
            "dataset_name": DATASET_NAME,
            "num_classes": number_of_classes(),
            "input_domain": "normalized_complete_image",
            "conditioning": "label_id",
        },
    )


def main(resume_train=False):
    ensure_artifact_directories()
    set_random_seed(RANDOM_SEED, deterministic=False)
    if MODEL_PATH.is_file() and not resume_train:
        raise FileExistsError(f"Evaluation_UNet checkpoint already exists: {MODEL_PATH}")

    train_loader = build_loader("train", TRAIN_BATCH_SIZE, shuffle=True, seed=RANDOM_SEED)
    valid_loader = build_loader("valid", VALID_BATCH_SIZE, seed=RANDOM_SEED + 1)
    test_loader = build_loader("test", TEST_BATCH_SIZE, seed=RANDOM_SEED + 2)
    draw_loader = build_loader("draw", DRAW_BATCH_SIZE, seed=RANDOM_SEED + 3)

    model = build_evaluation_unet().to(DEVICE)
    ema_model = build_ema_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        threshold=0.0,
        patience=PATIENCE_THRESHOLD_NUM,
        min_lr=MIN_LEARNING_RATE,
    )
    bce = nn.BCEWithLogitsLoss()
    state = {
        "iteration": 0,
        "max_iter_reached": False,
        "best_valid_loss": float("inf"),
        "best_epoch_num": 0,
        "patience_counter": 0,
        "patience_counter_after_min_lr": 0,
    }
    start_epoch = 0
    if resume_train:
        resume_path = LAST_MODEL_PATH if LAST_MODEL_PATH.is_file() else MODEL_PATH
        model, optimizer, scheduler, last_epoch, losses, extra, _ = load_model(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_training=True,
            ema_model=ema_model,
            ema_decay=EMA_DECAY,
            map_location=DEVICE,
        )
        start_epoch = int(last_epoch)
        state.update({
            "iteration": int(extra.get("global_iteration", start_epoch * len(train_loader))),
            "best_valid_loss": float(losses.get("min_loss", losses.get("valid_loss", float("inf")))),
            "best_epoch_num": int(extra.get("best_epoch_num", start_epoch)),
            "patience_counter": int(extra.get("patience_counter", 0)),
            "patience_counter_after_min_lr": int(
                extra.get("patience_counter_after_min_lr", 0)
            ),
        })
        training_monitor(
            start_epoch,
            MIN_LEARNING_RATE,
            optimizer.param_groups[0]["lr"],
            float("inf"),
            PATIENCE_THRESHOLD_NUM,
            resume_train=True,
            load_dict={
                "min_loss": state["best_valid_loss"],
                "best_epoch_idx": max(state["best_epoch_num"] - 1, 0),
                "patience_counter": state["patience_counter"],
                "patience_counter_after_min_lr": state["patience_counter_after_min_lr"],
            },
        )

    writer = SummaryWriter(str(LOG_DIR))
    for epoch_idx in range(start_epoch, TRAIN_MAX_EPOCHS):
        epoch_num = epoch_idx + 1
        current_lr = optimizer.param_groups[0]["lr"]
        train_result = train_epoch(train_loader, model, optimizer, bce, ema_model, state)
        valid_result = evaluate_loader(
            valid_loader, ema_model, bce, FGSEG_VALID_MICROBATCH_SIZE
        )
        (
            is_break_training,
            is_savemodel,
            patience_counter,
            patience_counter_after_min_lr,
        ) = training_monitor(
            epoch_idx,
            MIN_LEARNING_RATE,
            current_lr,
            valid_result["loss"],
            PATIENCE_THRESHOLD_NUM,
            resume_train=False,
            load_dict=None,
        )
        scheduler.step(valid_result["loss"])
        monitor_state = get_training_monitor_state()
        state.update({
            "best_valid_loss": monitor_state["min_loss"],
            "best_epoch_num": int(monitor_state["best_epoch_idx"]) + 1,
            "patience_counter": patience_counter,
            "patience_counter_after_min_lr": patience_counter_after_min_lr,
        })
        results = {"train": train_result, "valid": valid_result}
        if is_savemodel:
            _save_checkpoint(
                MODEL_PATH, epoch_num, model, ema_model, optimizer, scheduler, results, state
            )
        _save_checkpoint(
            LAST_MODEL_PATH, epoch_num, model, ema_model, optimizer, scheduler, results, state
        )
        for key, value in train_result.items():
            writer.add_scalar(f"Evaluation_UNet/train_{key}", value, epoch_num)
        for key, value in valid_result.items():
            writer.add_scalar(f"Evaluation_UNet/valid_{key}", value, epoch_num)
        summary = (
            f"Epoch: {epoch_num}, Iteration: {state['iteration']}, "
            f"Pc: {patience_counter}, Pc_min_lr: {patience_counter_after_min_lr}, "
            f"LR: {current_lr:.8f}, "
            f"Train_Loss: {train_result['loss']:.6f}, "
            f"Valid_Loss: {valid_result['loss']:.6f}, "
            f"Valid_IoU: {valid_result['iou']:.6f}, "
            f"Valid_Dice: {valid_result['dice']:.6f}"
        )
        print(summary)
        log_info(TRAINING_INFO_PATH, summary)
        if epoch_num == 1 or epoch_num % DRAW_INTERVAL_EPOCHS == 0:
            save_draw_snapshot(draw_loader, ema_model, epoch_num)
        if state["max_iter_reached"]:
            print(
                f"Reached TRAIN_MAX_ITERATIONS ({TRAIN_MAX_ITERATIONS}), "
                "stopping training."
            )
            break
        if is_break_training:
            break
    writer.close()

    best_model = build_evaluation_unet().to(DEVICE)
    best_model, _ = load_model_for_eval(MODEL_PATH, best_model, map_location=DEVICE)
    test_result = evaluate_loader(
        test_loader, best_model, bce, FGSEG_TEST_MICROBATCH_SIZE
    )
    TEST_INFO_PATH.write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "checkpoint": str(MODEL_PATH.resolve()),
                "best_epoch_num": state["best_epoch_num"],
                "metrics": test_result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IMPGM full-image evaluator UNet.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(resume_train=args.resume)
