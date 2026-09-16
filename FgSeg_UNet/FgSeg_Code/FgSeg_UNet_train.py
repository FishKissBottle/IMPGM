import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from osgeo import gdal
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from FgSeg_UNet.FgSeg_Code.FgSeg_UNet_model import FgSeg_UNet
from IMPGM_Config import *
from IMPGM_Dataset import IMPGM_Dataset
from IMPGM_Utils import (
    build_dataloader_generator,
    build_ema_model,
    build_scheduler,
    build_train_autocast,
    get_training_monitor_state,
    iter_microbatch_slices,
    load_model,
    load_model_for_eval,
    load_standard_vae,
    log_info,
    resolve_microbatch_size,
    restore_rng_state,
    save_model,
    save_msk_datas,
    save_rgb_datas,
    seed_dataloader_worker,
    set_random_seed,
    slice_microbatch,
    stash_rng_state,
    training_monitor,
    update_ema_model,
    prepare_rgb_vis_tensor,
)

gdal.UseExceptions()


def dice_loss_cal(msk_prob, msk_true):
    """Compute mean per-sample Dice loss entirely in FP32."""
    device_type = msk_prob.device.type
    with torch.amp.autocast(device_type, enabled=False):
        probabilities = msk_prob.to(dtype=torch.float32)
        targets = msk_true.to(dtype=torch.float32)
        smooth = 1.0e-6
        reduce_dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + targets.sum(
            dim=reduce_dims
        )
        return (
            1.0
            - (2.0 * intersection + smooth) / (denominator + smooth)
        ).mean()


def _compute_losses(logits, targets, criterion):
    targets = (targets >= 0.5).to(dtype=torch.float32)
    bce_loss = criterion(logits, targets)
    probabilities = torch.sigmoid(logits)
    dice_loss = dice_loss_cal(probabilities, targets)
    loss = 0.5 * bce_loss + 0.5 * dice_loss
    return loss, bce_loss, dice_loss, probabilities, targets


def _empty_metric_state():
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0}


def _update_metric_state(state, probabilities, targets, threshold=0.5):
    predictions = probabilities >= float(threshold)
    targets = targets >= 0.5
    state["tp"] += torch.logical_and(predictions, targets).sum().item()
    state["fp"] += torch.logical_and(predictions, ~targets).sum().item()
    state["fn"] += torch.logical_and(~predictions, targets).sum().item()


def _finalize_metrics(state):
    eps = 1.0e-8
    tp, fp, fn = state["tp"], state["fp"], state["fn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2.0 * precision * recall / (precision + recall + eps),
        "precision": precision,
        "recall": recall,
        "dice": 2.0 * tp / (2.0 * tp + fp + fn + eps),
    }


@torch.no_grad()
def _reconstruct_foreground(vae_model, fg_imgs):
    reconstructed, _, _ = vae_model(fg_imgs)
    return reconstructed


def train(
    data_loader,
    model,
    vae_model,
    criterion,
    optimizer,
    ema_model=None,
    training_state=None,
):
    model.train()
    vae_model.eval()
    loss_sum = 0.0
    bce_sum = 0.0
    dice_sum = 0.0
    sample_count = 0

    with tqdm(data_loader, colour="#ff924a", leave=False, ncols=120) as progress:
        for _, fg_imgs, msks, _, _, _ in progress:
            batch_size = int(fg_imgs.shape[0])
            microbatch_size = resolve_microbatch_size(
                FGSEG_TRAIN_MICROBATCH_SIZE, batch_size
            )
            optimizer.zero_grad(set_to_none=True)
            batch_loss_sum = 0.0
            batch_bce_sum = 0.0
            batch_dice_sum = 0.0

            for mb_start, mb_end in iter_microbatch_slices(
                batch_size, microbatch_size
            ):
                mb = int(mb_end - mb_start)
                microbatch_weight = float(mb) / float(batch_size)
                fg_imgs_mb = slice_microbatch(fg_imgs, mb_start, mb_end).to(
                    DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
                )
                msks_mb = slice_microbatch(msks, mb_start, mb_end).to(
                    DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
                )

                with build_train_autocast():
                    reconstructed = _reconstruct_foreground(vae_model, fg_imgs_mb)
                    logits = model(reconstructed)
                    loss, bce_loss, dice_loss, _, _ = _compute_losses(
                        logits, msks_mb, criterion
                    )

                (loss * microbatch_weight).backward()
                batch_loss_sum += loss.item() * mb
                batch_bce_sum += bce_loss.item() * mb
                batch_dice_sum += dice_loss.item() * mb

            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, EMA_DECAY)

            loss_sum += batch_loss_sum
            bce_sum += batch_bce_sum
            dice_sum += batch_dice_sum
            sample_count += batch_size

            progress.set_postfix(
                ordered_dict={
                    "loss": f"{batch_loss_sum / batch_size:.6f}",
                    "bce": f"{batch_bce_sum / batch_size:.6f}",
                    "dice": f"{batch_dice_sum / batch_size:.6f}",
                }
            )

            if training_state is not None:
                training_state["iteration"] += 1
                if (
                    TRAIN_MAX_ITERATIONS > 0
                    and training_state["iteration"] >= TRAIN_MAX_ITERATIONS
                ):
                    training_state["max_iter_reached"] = True
                    break

    if sample_count == 0:
        raise RuntimeError("Foreground segmentation training loader is empty.")
    return {
        "loss": loss_sum / sample_count,
        "bce_loss": bce_sum / sample_count,
        "dice_loss": dice_sum / sample_count,
    }


@torch.no_grad()
def evaluate(data_loader, model, vae_model, criterion, microbatch_size):
    loss_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    sample_count = 0
    metric_state = _empty_metric_state()

    model.eval()
    vae_model.eval()
    for _, fg_imgs, msks, _, _, _ in data_loader:
        batch_size = int(fg_imgs.shape[0])
        chunk_size = resolve_microbatch_size(microbatch_size, batch_size)
        for mb_start, mb_end in iter_microbatch_slices(batch_size, chunk_size):
            mb = int(mb_end - mb_start)
            fg_imgs_mb = slice_microbatch(fg_imgs, mb_start, mb_end).to(
                DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
            )
            msks_mb = slice_microbatch(msks, mb_start, mb_end).to(
                DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
            )
            with build_train_autocast():
                reconstructed = _reconstruct_foreground(vae_model, fg_imgs_mb)
                logits = model(reconstructed)
                loss, bce_loss, dice_loss, probabilities, targets = _compute_losses(
                    logits, msks_mb, criterion
                )

            loss_sum += loss.item() * mb
            bce_sum += bce_loss.item() * mb
            dice_loss_sum += dice_loss.item() * mb
            sample_count += mb
            _update_metric_state(metric_state, probabilities, targets)

    if sample_count == 0:
        raise RuntimeError("Foreground segmentation evaluation loader is empty.")
    result = {
        "loss": loss_sum / sample_count,
        "bce_loss": bce_sum / sample_count,
        "dice_loss": dice_loss_sum / sample_count,
    }
    result.update(_finalize_metrics(metric_state))
    return result


@torch.no_grad()
def _predict_draw_masks(model, reconstructed_imgs, threshold=0.5):
    outputs = []
    batch_size = int(reconstructed_imgs.shape[0])
    microbatch_size = resolve_microbatch_size(
        FGSEG_DRAW_MICROBATCH_SIZE, batch_size
    )
    model.eval()
    for mb_start, mb_end in iter_microbatch_slices(batch_size, microbatch_size):
        imgs_mb = reconstructed_imgs[mb_start:mb_end]
        with build_train_autocast():
            logits = model(imgs_mb)
        outputs.append((torch.sigmoid(logits) >= float(threshold)).to(torch.float32))
    return torch.cat(outputs, dim=0)


def _build_loader(dataset, batch_size, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(seed),
    )


def _get_split_roots(dataset_dict, split_name):
    suffix = {
        "train": "Train",
        "valid": "Valid",
        "test": "Test",
        "draw": "Draw",
    }[split_name]
    image_key = f"img_rootdir_list_for{suffix}"
    mask_key = f"msk_rootdir_list_for{suffix}"
    if image_key not in dataset_dict or mask_key not in dataset_dict:
        raise KeyError(
            f"FgSeg requires explicit {split_name!r} image and mask roots: "
            f"{image_key!r}, {mask_key!r}."
        )
    image_roots = list(dataset_dict[image_key])
    mask_roots = list(dataset_dict[mask_key])
    image_roots.sort(key=lambda path: "No" in path)
    return image_roots, mask_roots


def _save_checkpoint(
    save_path,
    epoch_num,
    model,
    ema_model,
    optimizer,
    scheduler,
    train_result,
    valid_result,
    training_state,
):
    monitor_state = get_training_monitor_state()
    tmp_path = f"{save_path}.tmp"
    save_model(
        tmp_path,
        epoch_num,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_dict={
            "train_loss": train_result["loss"],
            "valid_loss": valid_result["loss"],
            "min_loss": monitor_state["min_loss"],
        },
        extra={
            "ema_model_state_dict": ema_model.state_dict(),
            "ema_decay": EMA_DECAY,
            "best_epoch_num": int(monitor_state["best_epoch_idx"]) + 1,
            "patience_counter": monitor_state["patience_counter"],
            "patience_counter_after_min_lr": monitor_state[
                "patience_counter_after_min_lr"
            ],
            "global_iteration": int(training_state["iteration"]),
            "random_seed": RANDOM_SEED,
            "dataset_name": DATASET_NAME,
        },
    )
    os.replace(tmp_path, save_path)


def _format_result(prefix, result):
    return (
        f"{prefix}_Loss: {result['loss']:.6f}, "
        f"{prefix}_BCE: {result['bce_loss']:.6f}, "
        f"{prefix}_DiceLoss: {result['dice_loss']:.6f}, "
        f"{prefix}_IoU: {result['iou']:.6f}, "
        f"{prefix}_F1: {result['f1']:.6f}, "
        f"{prefix}_Precision: {result['precision']:.6f}, "
        f"{prefix}_Recall: {result['recall']:.6f}"
    )


def main(dataset_dict=None, resume_train=False):
    dataset_dict = DATASET_DICT if dataset_dict is None else dataset_dict
    if dataset_dict is None:
        raise RuntimeError("DATASET_DICT must not be None.")

    os.makedirs(FGSEG_LOG_DIR, exist_ok=True)
    os.makedirs(FGSEG_RGB_DIR, exist_ok=True)
    os.makedirs(FGSEG_TIF_DIR, exist_ok=True)
    Path(FGSEG_UNET_MODEL_SAVEPATH).parent.mkdir(parents=True, exist_ok=True)
    set_random_seed(RANDOM_SEED, deterministic=False)

    writer = SummaryWriter(FGSEG_LOG_DIR)
    last_model_path = str(
        Path(FGSEG_UNET_MODEL_SAVEPATH).with_name(
            f"{Path(FGSEG_UNET_MODEL_SAVEPATH).stem}_last.pth"
        )
    )
    if os.path.exists(FGSEG_UNET_MODEL_SAVEPATH) and not resume_train:
        raise FileExistsError(
            f"FgSeg checkpoint exists and resume_train=False: "
            f"{FGSEG_UNET_MODEL_SAVEPATH}"
        )

    model = FgSeg_UNet(
        in_channels=INPUT_CHANNELS,
        out_channels=UNET_OUTPUT_CHANNELS,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
    )
    scheduler = build_scheduler(
        optimizer, MIN_LEARNING_RATE, PATIENCE_THRESHOLD_NUM
    )
    ema_model = build_ema_model(model)
    resume_extra = {}

    if resume_train:
        resume_path = (
            last_model_path
            if os.path.exists(last_model_path)
            else FGSEG_UNET_MODEL_SAVEPATH
        )
        (
            model,
            optimizer,
            scheduler,
            last_epoch,
            last_loss_dict,
            resume_extra,
            _,
        ) = load_model(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_training=True,
            preload_only=False,
            ema_model=ema_model,
            ema_decay=EMA_DECAY,
            map_location=DEVICE,
        )
        start_epoch = max(int(last_epoch), 0)
        fallback_best_epoch_idx = max(last_epoch - 1, 0)
        best_epoch_num_raw = resume_extra.get(
            "best_epoch_num",
            last_loss_dict.get("best_epoch_num"),
        )
        resume_best_epoch_idx = (
            max(int(best_epoch_num_raw) - 1, 0)
            if best_epoch_num_raw is not None
            else fallback_best_epoch_idx
        )
        training_monitor(
            fallback_best_epoch_idx,
            MIN_LEARNING_RATE,
            optimizer.param_groups[0]["lr"],
            float("inf"),
            PATIENCE_THRESHOLD_NUM,
            resume_train=True,
            load_dict={
                "min_loss": last_loss_dict.get(
                    "min_loss",
                    last_loss_dict.get("valid_loss", float("inf")),
                ),
                "best_epoch_idx": resume_best_epoch_idx,
                "patience_counter": int(
                    resume_extra.get(
                        "patience_counter",
                        last_loss_dict.get("patience_counter", 0),
                    )
                ),
                "patience_counter_after_min_lr": int(
                    resume_extra.get(
                        "patience_counter_after_min_lr",
                        last_loss_dict.get("patience_counter_after_min_lr", 0),
                    )
                ),
            },
        )
    else:
        start_epoch = 0

    split_datasets = {}
    for split_name, is_train in (
        ("train", True),
        ("valid", False),
        ("test", False),
        ("draw", False),
    ):
        image_roots, mask_roots = _get_split_roots(dataset_dict, split_name)
        split_datasets[split_name] = IMPGM_Dataset(
            img_rootdir_list=image_roots,
            msk_rootdir_list=mask_roots,
            is_train=is_train,
        )

    train_loader = _build_loader(
        split_datasets["train"], TRAIN_BATCH_SIZE, True, None
    )
    valid_loader = _build_loader(
        split_datasets["valid"], VALID_BATCH_SIZE, False, RANDOM_SEED + 1
    )
    test_loader = _build_loader(
        split_datasets["test"], TEST_BATCH_SIZE, False, RANDOM_SEED + 2
    )
    draw_loader = _build_loader(
        split_datasets["draw"], DRAW_BATCH_SIZE, False, RANDOM_SEED + 3
    )

    try:
        _, draw_fg_imgs, draw_msks, _, _, _ = next(iter(draw_loader))
    except StopIteration as exc:
        raise RuntimeError("FgSeg draw dataset is empty.") from exc
    draw_fg_imgs = draw_fg_imgs.to(
        DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
    )
    draw_msks = draw_msks.to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)

    vae_model, _, _ = load_standard_vae(
        VAE_MODEL_SAVEPATH, device=DEVICE, load_ema=True
    )
    vae_model.eval()
    with build_train_autocast():
        draw_reconstructed = _reconstruct_foreground(vae_model, draw_fg_imgs)

    criterion = nn.BCEWithLogitsLoss()
    model.train()
    training_state = {
        "iteration": int(
            resume_extra.get(
                "global_iteration", start_epoch * len(train_loader)
            )
        ),
        "max_iter_reached": False,
    }

    for epoch_idx in range(start_epoch, TRAIN_MAX_EPOCHS):
        epoch_num = epoch_idx + 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_result = train(
            train_loader,
            model,
            vae_model,
            criterion,
            optimizer,
            ema_model=ema_model,
            training_state=training_state,
        )

        eval_rng_state = stash_rng_state()
        try:
            if DETERMINISTIC_EVAL:
                set_random_seed(RANDOM_SEED, deterministic=False)
            valid_result = evaluate(
                valid_loader,
                ema_model,
                vae_model,
                criterion,
                FGSEG_VALID_MICROBATCH_SIZE,
            )
        finally:
            restore_rng_state(eval_rng_state)
        model.train()

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

        for name, value in train_result.items():
            writer.add_scalar(f"FgSeg_UNet/train_{name}", value, epoch_num)
        for name, value in valid_result.items():
            writer.add_scalar(f"FgSeg_UNet/valid_{name}", value, epoch_num)

        summary = (
            f"Epoch: {epoch_num}, Iteration: {training_state['iteration']}, "
            f"Pc: {patience_counter}, Pc_min_lr: "
            f"{patience_counter_after_min_lr}, Lr: {current_lr:.8f}, "
            f"Train_Loss: {train_result['loss']:.6f}, "
            f"{_format_result('Valid', valid_result)}"
        )
        print(f"   {summary}")
        log_info(
            FGSEG_TRAIN_INFO_PATH,
            f">>> {summary}",
        )
        log_info(
            FGSEG_TRAIN_INFO_PATH,
            "",
        )

        if epoch_num == 1:
            save_rgb_datas(
                prepare_rgb_vis_tensor(draw_fg_imgs, masks=draw_msks).cpu(),
                nrow=3,
                savepath=os.path.join(FGSEG_RGB_DIR, "FgSeg_Target_RGB.png"),
            )
            save_rgb_datas(
                prepare_rgb_vis_tensor(
                    draw_reconstructed, masks=draw_msks
                ).cpu(),
                nrow=3,
                savepath=os.path.join(
                    FGSEG_RGB_DIR, "FgSeg_VAE_Reconstruction_RGB.png"
                ),
            )
            save_msk_datas(
                draw_msks.cpu(),
                nrow=3,
                savepath=os.path.join(FGSEG_RGB_DIR, "FgSeg_Target_Mask.png"),
            )

        if is_savemodel:
            _save_checkpoint(
                FGSEG_UNET_MODEL_SAVEPATH,
                epoch_num,
                model,
                ema_model,
                optimizer,
                scheduler,
                train_result,
                valid_result,
                training_state,
            )

        should_save_last = (
            epoch_num % TIF_INTERVAL_EPOCHS == 0
            or training_state["max_iter_reached"]
            or is_break_training
            or epoch_num == TRAIN_MAX_EPOCHS
        )
        if should_save_last:
            _save_checkpoint(
                last_model_path,
                epoch_num,
                model,
                ema_model,
                optimizer,
                scheduler,
                train_result,
                valid_result,
                training_state,
            )

        if epoch_num in EXTRA_MODEL_SAVE_EPOCH_NUM:
            extra_path = str(
                Path(FGSEG_UNET_MODEL_SAVEPATH).with_name(
                    f"{Path(FGSEG_UNET_MODEL_SAVEPATH).stem}_"
                    f"{epoch_num}e.pth"
                )
            )
            _save_checkpoint(
                extra_path,
                epoch_num,
                model,
                ema_model,
                optimizer,
                scheduler,
                train_result,
                valid_result,
                training_state,
            )

        if epoch_num % DRAW_INTERVAL_EPOCHS == 0:
            predicted_masks = _predict_draw_masks(
                ema_model, draw_reconstructed
            )
            save_msk_datas(
                predicted_masks.cpu(),
                nrow=3,
                savepath=os.path.join(
                    FGSEG_RGB_DIR,
                    f"FgSeg_Predicted_Mask_{epoch_num}.png",
                ),
            )

        if training_state["max_iter_reached"]:
            print(
                f"Reached TRAIN_MAX_ITERATIONS ({TRAIN_MAX_ITERATIONS}), "
                "stopping training."
            )
            break
        if is_break_training:
            break
        time.sleep(5)

    writer.close()

    best_model = FgSeg_UNet(
        in_channels=INPUT_CHANNELS,
        out_channels=UNET_OUTPUT_CHANNELS,
    ).to(DEVICE)
    best_model, _ = load_model_for_eval(
        FGSEG_UNET_MODEL_SAVEPATH,
        best_model,
        map_location=DEVICE,
    )
    test_result = evaluate(
        test_loader,
        best_model,
        vae_model,
        criterion,
        FGSEG_TEST_MICROBATCH_SIZE,
    )
    test_summary = _format_result("Test", test_result)
    print(f"   {test_summary}")
    log_info(
        FGSEG_TEST_INFO_PATH,
        test_summary,
    )
    return test_result


if __name__ == "__main__":
    main(dataset_dict=DATASET_DICT, resume_train=False)
