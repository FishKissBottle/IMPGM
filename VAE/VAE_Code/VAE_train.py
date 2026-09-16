import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from osgeo import gdal
from tqdm import tqdm

from IMPGM_Dataset import IMPGM_Dataset
from VAE.VAE_Code.VAE_model import AutoEncoder
from IMPGM_Config import *
from IMPGM_Utils import (
    build_dataloader_generator,
    build_ema_model,
    build_scheduler,
    build_train_autocast,
    get_training_monitor_state,
    iter_microbatch_slices,
    load_model,
    log_info,
    restore_rng_state,
    resolve_microbatch_size,
    save_model,
    save_rgb_datas,
    save_tif_datas,
    seed_dataloader_worker,
    set_random_seed,
    slice_microbatch,
    stash_rng_state,
    training_monitor,
    update_ema_model,
    denormalize_image_tensor,
    prepare_rgb_vis_tensor,
)

from Evaluation.Evaluation_Code.IMPGM_Quality_Metrics import (
    compute_FSIM,
    compute_GMSD,
    compute_LPIPS,
    compute_PSNR,
    compute_SAM,
    compute_SSIM,
)

gdal.UseExceptions()


def loss_function(imgs_d, imgs, mu, logvar):
    """Compute VAE loss matching CSUA_LDM_VAE_NoDisent's KL formulation.

    Reconstruction: MSE with mean reduction.
    KL: element-wise mean (mean over batch, channel, H, W) with logvar
        clipped to [-20, 20] for numeric stability.
    """
    recon_loss = F.mse_loss(imgs_d, imgs, reduction='mean')
    logvar_clipped = torch.clamp(logvar, min=-20.0, max=20.0)
    kl_loss = (-0.5 * (1 + logvar_clipped - mu.pow(2) - logvar_clipped.exp())).mean()
    total_loss = recon_loss + kl_loss * KL_LOSS_WEIGHT
    return total_loss, recon_loss, kl_loss


def _scalar_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def train(
    data_loader,
    VAE_model,
    optimizer,
    ema_model=None,
    training_state=None,
    benchmark_recorder=None,
):
    train_loss_sum = 0.0
    train_mse_loss_sum = 0.0
    train_kl_loss_sum = 0.0
    num_samples = 0
    with tqdm(data_loader, leave=False, colour="#ff924a", ncols=120) as pbar:
        for _, fg_imgs, _, syn_imgs, _, _ in pbar:
            if benchmark_recorder is not None:
                benchmark_recorder.before_step((fg_imgs, syn_imgs))
            imgs = torch.cat([fg_imgs, syn_imgs], dim=0)
            B = imgs.shape[0]
            microbatch_size = resolve_microbatch_size(VAE_TRAIN_MICROBATCH_SIZE, B)

            optimizer.zero_grad()
            batch_loss_sum = 0.0
            batch_mse_loss_sum = 0.0
            batch_kl_loss_sum = 0.0
            for mb_start, mb_end in iter_microbatch_slices(B, microbatch_size):
                mb = int(mb_end - mb_start)
                microbatch_weight = float(mb) / float(B)
                imgs_mb = slice_microbatch(imgs, mb_start, mb_end).to(
                    DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
                )
                with build_train_autocast():
                    imgs_d, mu, logvar = VAE_model(imgs_mb)
                    loss, mse_loss, kl_loss = loss_function(imgs_d, imgs_mb, mu, logvar)

                (loss * microbatch_weight).backward()
                batch_loss_sum += loss.item() * mb
                batch_mse_loss_sum += mse_loss.item() * mb
                batch_kl_loss_sum += kl_loss.item() * mb

            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, VAE_model, EMA_DECAY)

            batch_loss = batch_loss_sum / B
            batch_mse_loss = batch_mse_loss_sum / B
            batch_kl_loss = batch_kl_loss_sum / B

            train_loss_sum += batch_loss_sum
            train_mse_loss_sum += batch_mse_loss_sum
            train_kl_loss_sum += batch_kl_loss_sum
            num_samples += B

            if training_state is not None:
                training_state['iteration'] += 1
                if (
                    benchmark_recorder is not None
                    and benchmark_recorder.after_step()
                ):
                    return (
                        train_loss_sum / num_samples,
                        train_mse_loss_sum / num_samples,
                        train_kl_loss_sum / num_samples,
                    )
                if TRAIN_MAX_ITERATIONS > 0 and training_state['iteration'] >= TRAIN_MAX_ITERATIONS:
                    training_state['max_iter_reached'] = True
                    return train_loss_sum / num_samples, train_mse_loss_sum / num_samples, train_kl_loss_sum / num_samples

            pbar.set_postfix(ordered_dict={
                'loss': f'{batch_loss:.6f}',
                'mse' : f'{batch_mse_loss:.6f}',
                'kl'  : f'{batch_kl_loss:.6f}',
            })

    return train_loss_sum / num_samples, train_mse_loss_sum / num_samples, train_kl_loss_sum / num_samples


def benchmark_training_memory(
    *, output, warmup_steps=5, measure_steps=20, batch_size=1,
    seed=999, overwrite=False,
):
    """Measure the formal shared-VAE training path without validation or saving."""
    from Evaluation.Evaluation_Code.IMPGM_Training_Memory_Evaluation import (
        TrainingMemoryRecorder,
    )

    set_random_seed(int(seed), deterministic=False)
    VAE_model = AutoEncoder(
        img_channel=VAE_IMG_CHANNEL,
        down_channels=VAE_DOWN_CHANNELS,
        mid_inout_channels=VAE_MID_CHANNELS,
        num_down_layers=VAE_NUM_DOWN_LAYERS,
        num_mid_layers=VAE_NUM_MID_LAYERS,
        num_up_layers=VAE_NUM_UP_LAYERS,
        z_channel=VAE_Z_CHANNEL,
        norm_channels=VAE_NORM_CHANNELS,
    ).to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
    _maybe_preload_vae_weights(VAE_model)
    optimizer = torch.optim.Adam(
        VAE_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999)
    )
    ema_model = build_ema_model(VAE_model)
    dataset_dict = DATASET_DICT
    if dataset_dict is None:
        raise RuntimeError("DATASET_DICT must not be None.")
    train_images = sorted(
        dataset_dict['img_rootdir_list_forTrain'], key=lambda path: 'No' in path
    )
    train_dataset = IMPGM_Dataset(
        img_rootdir_list=train_images,
        msk_rootdir_list=dataset_dict['msk_rootdir_list_forTrain'],
        is_train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(int(seed)),
    )
    recorder = TrainingMemoryRecorder(
        method_key="impgm",
        stage="vae_training",
        output=output,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        batch_size=batch_size,
        seed=seed,
        overwrite=overwrite,
        precision="formal build_train_autocast configuration",
        notes=(
            "Each dataset item contributes one foreground image and one complete "
            "image to the shared-VAE loss."
        ),
    )
    recorder.bind_trainable_modules([VAE_model])
    train(
        train_loader,
        VAE_model,
        optimizer,
        ema_model=ema_model,
        training_state={"iteration": 0, "max_iter_reached": False},
        benchmark_recorder=recorder,
    )
    return recorder.finish()


def _eval_vae_batch(imgs, VAE_model, microbatch_size):
    batch_size = imgs.shape[0]
    microbatch_size = resolve_microbatch_size(microbatch_size, batch_size)
    loss_sum = 0.0
    mse_sum = 0.0
    kl_sum = 0.0

    for mb_start, mb_end in iter_microbatch_slices(batch_size, microbatch_size):
        mb = int(mb_end - mb_start)
        imgs_mb = slice_microbatch(imgs, mb_start, mb_end).to(
            DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
        )
        imgs_d, mu, logvar = VAE_model(imgs_mb)
        loss, mse_loss, kl_loss = loss_function(imgs_d, imgs_mb, mu, logvar)
        loss_sum += loss.item() * mb
        mse_sum += mse_loss.item() * mb
        kl_sum += kl_loss.item() * mb

    return loss_sum / batch_size, mse_sum / batch_size, kl_sum / batch_size


def valid(data_loader, VAE_model):
    valid_loss_sum = 0.0
    valid_mse_loss_sum = 0.0
    valid_kl_loss_sum = 0.0
    num_samples = 0
    with torch.no_grad():
        for _, fg_imgs, _, syn_imgs, _, _ in data_loader:
            B = fg_imgs.shape[0]
            fg_loss, fg_mse_loss, fg_kl_loss = _eval_vae_batch(
                fg_imgs, VAE_model, VAE_VALID_MICROBATCH_SIZE,
            )
            syn_loss, syn_mse_loss, syn_kl_loss = _eval_vae_batch(
                syn_imgs, VAE_model, VAE_VALID_MICROBATCH_SIZE,
            )

            loss = (fg_loss + syn_loss) / 2.0
            mse_loss = (fg_mse_loss + syn_mse_loss) / 2.0
            kl_loss = (fg_kl_loss + syn_kl_loss) / 2.0

            valid_loss_sum += _scalar_to_float(loss) * B
            valid_mse_loss_sum += _scalar_to_float(mse_loss) * B
            valid_kl_loss_sum += _scalar_to_float(kl_loss) * B
            num_samples += B

    return valid_loss_sum / num_samples, valid_mse_loss_sum / num_samples, valid_kl_loss_sum / num_samples


def generate_samples(imgs, VAE_model, microbatch_size=VAE_DRAW_MICROBATCH_SIZE):
    with torch.no_grad():
        batch_size = imgs.shape[0]
        microbatch_size = resolve_microbatch_size(microbatch_size, batch_size)
        outputs = []
        for mb_start, mb_end in iter_microbatch_slices(batch_size, microbatch_size):
            imgs_mb = slice_microbatch(imgs, mb_start, mb_end).to(
                DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING
            )
            imgs_d, _, _ = VAE_model(imgs_mb)
            outputs.append(imgs_d)
    return torch.cat(outputs, dim=0)


def _maybe_preload_vae_weights(VAE_model):
    """Initialize compatible VAE weights from an EMA checkpoint."""
    if not VAE_PRELOAD_ENABLED:
        return

    if not VAE_PRELOAD_SOURCE_PATH:
        raise ValueError(
            "vae.preload.enabled is true, but vae.preload.source_path is empty."
        )

    source_path = Path(VAE_PRELOAD_SOURCE_PATH).expanduser()
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"VAE preload checkpoint does not exist: {source_path}")

    print(f"[Preload] Loading VAE EMA weights from: {source_path}")
    load_model(
        source_path,
        model=VAE_model,
        optimizer=None,
        scheduler=None,
        resume_training=True,
        preload_only=True,
        load_ema=True,
        strict=True,
        map_location=DEVICE,
        allowed_shape_mismatch_prefixes=(
            "encoder_conv_in",
            "decoder_conv_out",
        ),
    )


def main(resume_train=False):
    set_random_seed(RANDOM_SEED, deterministic=False)
    os.makedirs(VAE_LOG_DIR, exist_ok=True)
    os.makedirs(VAE_RGB_DIR, exist_ok=True)
    os.makedirs(VAE_TIF_DIR, exist_ok=True)

    writer = SummaryWriter(VAE_LOG_DIR)

    # Check whether a checkpoint already exists at the target path.
    if os.path.exists(VAE_MODEL_SAVEPATH) and not resume_train:
        raise Exception("VAE model checkpoint already exists, but resume_train is False.")

    VAE_model = AutoEncoder(
        img_channel=VAE_IMG_CHANNEL,
        down_channels=VAE_DOWN_CHANNELS,
        mid_inout_channels=VAE_MID_CHANNELS,
        num_down_layers=VAE_NUM_DOWN_LAYERS,
        num_mid_layers=VAE_NUM_MID_LAYERS,
        num_up_layers=VAE_NUM_UP_LAYERS,
        z_channel=VAE_Z_CHANNEL,
        norm_channels=VAE_NORM_CHANNELS,
    ).to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)

    if not resume_train:
        _maybe_preload_vae_weights(VAE_model)

    optimizer = torch.optim.Adam(VAE_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    lr_scheduler = build_scheduler(optimizer, MIN_LEARNING_RATE, PATIENCE_THRESHOLD_NUM)
    ema_model = build_ema_model(VAE_model)
    resume_extra = {}

    if resume_train:
        VAE_model, optimizer, lr_scheduler, VAE_last_epoch, VAE_last_loss_dict, resume_extra, _ = load_model(
            VAE_MODEL_SAVEPATH,
            VAE_model,
            optimizer,
            scheduler=lr_scheduler,
            resume_training=True,
            preload_only=False,
            ema_model=ema_model,
            ema_decay=EMA_DECAY,
            map_location=DEVICE,
        )
        start_epoch = max(int(VAE_last_epoch), 0)
    else:
        start_epoch = 0

    dataset_dict = DATASET_DICT
    if dataset_dict is None:
        raise Exception("DATASET_DICT must not be None.")

    img_rootdir_list_forTrain = dataset_dict['img_rootdir_list_forTrain']
    img_rootdir_list_forTrain.sort(key=lambda x: 'No' in x)
    msk_rootdir_list_forTrain = dataset_dict['msk_rootdir_list_forTrain']

    img_rootdir_list_forValid = dataset_dict['img_rootdir_list_forValid']
    img_rootdir_list_forValid.sort(key=lambda x: 'No' in x)
    msk_rootdir_list_forValid = dataset_dict['msk_rootdir_list_forValid']

    img_rootdir_list_forDraw = dataset_dict['img_rootdir_list_forDraw']
    img_rootdir_list_forDraw.sort(key=lambda x: 'No' in x)
    msk_rootdir_list_forDraw = dataset_dict['msk_rootdir_list_forDraw']

    train_dataset = IMPGM_Dataset(
        img_rootdir_list=img_rootdir_list_forTrain,
        msk_rootdir_list=msk_rootdir_list_forTrain,
        is_train=True,
    )
    train_generator = build_dataloader_generator(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=train_generator,
    )

    valid_dataset = IMPGM_Dataset(
        img_rootdir_list=img_rootdir_list_forValid,
        msk_rootdir_list=msk_rootdir_list_forValid,
        is_train=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(RANDOM_SEED + 1),
    )

    draw_dataset = IMPGM_Dataset(
        img_rootdir_list=img_rootdir_list_forDraw,
        msk_rootdir_list=msk_rootdir_list_forDraw,
        is_train=False,
    )
    draw_loader = DataLoader(
        draw_dataset,
        batch_size=DRAW_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(RANDOM_SEED + 2),
    )

    _, draw_fg_imgs, draw_msks, draw_syn_imgs, draw_projs, draw_geos = next(iter(draw_loader))

    draw_pair_count = min(3, draw_fg_imgs.shape[0], draw_syn_imgs.shape[0])
    draw_fg_imgs = draw_fg_imgs[:draw_pair_count]
    draw_syn_imgs = draw_syn_imgs[:draw_pair_count]
    draw_fg_msks = draw_msks[:draw_pair_count]
    draw_imgs = torch.cat([draw_fg_imgs, draw_syn_imgs], dim=0)
    draw_save_msks = torch.cat([draw_fg_msks, torch.ones_like(draw_fg_msks)], dim=0)
    draw_imgs = draw_imgs.to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)
    draw_save_msks = draw_save_msks.to(DEVICE, non_blocking=DATA_TRANSFER_NON_BLOCKING)

    draw_geos = np.array([draw_geo.numpy() for draw_geo in draw_geos]).T
    draw_source_geos = draw_geos[:draw_pair_count]
    draw_save_geos = np.concatenate([draw_source_geos, draw_source_geos], axis=0)
    draw_source_projs = list(draw_projs[:draw_pair_count])
    draw_save_projs = draw_source_projs + draw_source_projs
    if any(
        not isinstance(projection, str) or not projection.strip()
        for projection in draw_source_projs
    ):
        print(
            "[VAE] Draw samples contain no projection WKT; snapshot TIFs "
            "will use the configured placeholder georeference."
        )
        draw_save_projs = None
        draw_save_geos = None

    VAE_model.train()
    training_state = {
        "iteration": int(
            resume_extra.get("global_iteration", start_epoch * len(train_loader))
        ),
        "max_iter_reached": False,
    }

    for epoch_idx in range(start_epoch, TRAIN_MAX_EPOCHS):
        epoch_num = epoch_idx + 1
        train_generator.manual_seed(RANDOM_SEED + epoch_idx)
        global current_lr
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        if resume_train and epoch_idx == start_epoch:
            fallback_best_epoch_idx = max(VAE_last_epoch - 1, 0)
            best_epoch_num_raw = resume_extra.get(
                "best_epoch_num",
                VAE_last_loss_dict.get("best_epoch_num"),
            )
            resume_best_epoch_idx = (
                max(int(best_epoch_num_raw) - 1, 0)
                if best_epoch_num_raw is not None
                else fallback_best_epoch_idx
            )
            load_dict = {
                'min_loss': VAE_last_loss_dict.get(
                    'min_loss',
                    VAE_last_loss_dict.get('valid_loss', float('inf')),
                ),
                'best_epoch_idx': resume_best_epoch_idx,
                'patience_counter': int(
                    resume_extra.get(
                        'patience_counter',
                        VAE_last_loss_dict.get('patience_counter', 0),
                    )
                ),
                'patience_counter_after_min_lr': int(
                    resume_extra.get(
                        'patience_counter_after_min_lr',
                        VAE_last_loss_dict.get('patience_counter_after_min_lr', 0),
                    )
                ),
            }
            is_break_training, is_savemodel, patience_counter, patience_counter_after_min_lr = training_monitor(
                epoch_idx, MIN_LEARNING_RATE, current_lr, float('inf'),
                PATIENCE_THRESHOLD_NUM, resume_train=resume_train, load_dict=load_dict,
            )

            train_loss, train_mse_loss, train_kl_loss = train(
                train_loader, VAE_model, optimizer,
                ema_model=ema_model, training_state=training_state,
            )
            eval_rng_state = stash_rng_state()
            if DETERMINISTIC_EVAL:
                set_random_seed(RANDOM_SEED, deterministic=False)
            ema_model.eval()
            valid_loss, valid_mse_loss, valid_kl_loss = valid(valid_loader, ema_model)
            restore_rng_state(eval_rng_state)
            VAE_model.train()
            is_break_training, is_savemodel, patience_counter, patience_counter_after_min_lr = training_monitor(
                epoch_idx, MIN_LEARNING_RATE, current_lr, valid_loss,
                PATIENCE_THRESHOLD_NUM, resume_train=False, load_dict=None,
            )

        else:
            train_loss, train_mse_loss, train_kl_loss = train(
                train_loader, VAE_model, optimizer,
                ema_model=ema_model, training_state=training_state,
            )
            eval_rng_state = stash_rng_state()
            if DETERMINISTIC_EVAL:
                set_random_seed(RANDOM_SEED, deterministic=False)
            ema_model.eval()
            valid_loss, valid_mse_loss, valid_kl_loss = valid(valid_loader, ema_model)
            restore_rng_state(eval_rng_state)
            VAE_model.train()
            is_break_training, is_savemodel, patience_counter, patience_counter_after_min_lr = training_monitor(
                epoch_idx, MIN_LEARNING_RATE, current_lr, valid_loss,
                PATIENCE_THRESHOLD_NUM, resume_train=False, load_dict=None,
            )

        lr_scheduler.step(valid_loss)
        monitor_state = get_training_monitor_state()

        writer.add_scalars('VAE/train_loss'    , {f"train_loss"    : train_loss}    , epoch_num)
        writer.add_scalars('VAE/train_mse_loss', {f"train_mse_loss": train_mse_loss}, epoch_num)
        writer.add_scalars('VAE/train_kl_loss' , {f"train_kl_loss" : train_kl_loss} , epoch_num)
        writer.add_scalars('VAE/valid_loss'    , {f"valid_loss"    : valid_loss }    , epoch_num)
        writer.add_scalars('VAE/valid_mse_loss', {f"valid_mse_loss": valid_mse_loss} , epoch_num)
        writer.add_scalars('VAE/valid_kl_loss' , {f"valid_kl_loss" : valid_kl_loss}  , epoch_num)

        print(f'    Epoch: {epoch_num}, Iteration: {training_state["iteration"]}, Pc: {patience_counter}, Pc_min_lr: {patience_counter_after_min_lr}, Lr: {current_lr:.8f}, Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}, Valid_Mse_loss: {valid_mse_loss:.6f}, Valid_Kl_loss: {valid_kl_loss:.6f}')

        log_info(
            log_path=VAE_TRAIN_INFO_PATH,
            text=f'>>> Epoch: {epoch_num}, Pc: {patience_counter}, Pc_min_lr: {patience_counter_after_min_lr}, Lr: {current_lr:.8f}, Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}, Valid_Mse_loss: {valid_mse_loss:.6f}, Valid_Kl_loss: {valid_kl_loss:.6f}',
        )
        log_info(
            log_path=VAE_TRAIN_INFO_PATH,
            text='',
        )   # Insert a blank line between epochs in the log file.

        if epoch_num == 1:
            save_rgb_datas(
                prepare_rgb_vis_tensor(draw_imgs, masks=draw_save_msks).cpu(), nrow=3,
                savepath=os.path.join(VAE_RGB_DIR, 'VAE_RGB_0.png'),
                is_showminmax=True,
            )
            save_tif_datas(
                draw_imgs.cpu(), draw_save_projs, draw_save_geos,
                savepath=os.path.join(VAE_TIF_DIR, 'VAE_TIF_0.tif'),
                is_showminmax=True,
                masks=draw_save_msks.cpu(),
            )

        if is_savemodel or epoch_num % 25 == 0:
            loss_dict = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'min_loss': monitor_state['min_loss'],
            }
            save_model(
                VAE_MODEL_SAVEPATH,
                epoch_num,
                model=VAE_model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                loss_dict=loss_dict,
                extra={
                    **VAE_PRELOAD_METADATA,
                    'ema_model_state_dict': ema_model.state_dict(),
                    'ema_decay': EMA_DECAY,
                    'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                    'patience_counter': monitor_state['patience_counter'],
                    'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                    'latent_scaling_factor': LATENT_SCALING_FACTOR,
                    'global_iteration': int(training_state['iteration']),
                    'random_seed': RANDOM_SEED,
                },
            )

        if epoch_num in PRETRAIN_EXTRA_MODEL_SAVE_EPOCH_NUM:
            loss_dict = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'min_loss': monitor_state['min_loss'],
            }
            save_model(
                VAE_MODEL_SAVEPATH.replace('.pth', '') + f'_{epoch_num}e.pth',
                epoch_num,
                model=VAE_model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                loss_dict=loss_dict,
                extra={
                    **VAE_PRELOAD_METADATA,
                    'ema_model_state_dict': ema_model.state_dict(),
                    'ema_decay': EMA_DECAY,
                    'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                    'patience_counter': monitor_state['patience_counter'],
                    'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                    'latent_scaling_factor': LATENT_SCALING_FACTOR,
                    'global_iteration': int(training_state['iteration']),
                    'random_seed': RANDOM_SEED,
                },
            )

        # Evaluate metrics and save an RGB snapshot every five epochs.
        eval_freq = 5

        if epoch_num % eval_freq == 0:
            eval_rng_state = stash_rng_state()
            if DETERMINISTIC_EVAL:
                set_random_seed(RANDOM_SEED, deterministic=False)
            ema_model.eval()
            recon_imgs = generate_samples(draw_imgs, ema_model)
            restore_rng_state(eval_rng_state)
            VAE_model.train()

            # Denormalize to real pixel space and clamp to [0, 1] for metric evaluation.
            recon_imgs_normalized = torch.clamp(denormalize_image_tensor(recon_imgs), min=0.0, max=1.0)
            draw_imgs_normalized = torch.clamp(denormalize_image_tensor(draw_imgs), min=0.0, max=1.0)

            psnr_val = compute_PSNR(recon_imgs_normalized, draw_imgs_normalized)
            ssim_val = compute_SSIM(recon_imgs_normalized, draw_imgs_normalized)
            rgb_gmsd_val, nir_gmsd_val = compute_GMSD(recon_imgs_normalized, draw_imgs_normalized)
            rgb_fsim_val, nir_fsim_val = compute_FSIM(recon_imgs_normalized, draw_imgs_normalized)
            rgb_lpips_val, nir_lpips_val = compute_LPIPS(recon_imgs_normalized, draw_imgs_normalized)
            sam_val = compute_SAM(recon_imgs_normalized, draw_imgs_normalized)

            writer.add_scalars('VAE/psnr', {f"psnr": psnr_val}, epoch_num)
            writer.add_scalars('VAE/ssim', {f"ssim": ssim_val}, epoch_num)
            writer.add_scalars('VAE/rgb_gmsd', {f"rgb_gmsd": rgb_gmsd_val}, epoch_num)
            writer.add_scalars('VAE/rgb_fsim', {f"rgb_fsim": rgb_fsim_val}, epoch_num)
            if nir_gmsd_val is not None:
                writer.add_scalars('VAE/nir_gmsd', {"nir_gmsd": nir_gmsd_val}, epoch_num)
            if nir_fsim_val is not None:
                writer.add_scalars('VAE/nir_fsim', {"nir_fsim": nir_fsim_val}, epoch_num)
            if rgb_lpips_val is not None:
                writer.add_scalars('VAE/rgb_lpips', {"rgb_lpips": rgb_lpips_val}, epoch_num)
            if nir_lpips_val is not None:
                writer.add_scalars('VAE/nir_lpips', {"nir_lpips": nir_lpips_val}, epoch_num)
            writer.add_scalars('VAE/sam', {f"sam": sam_val}, epoch_num)

            nir_gmsd_text = f"{nir_gmsd_val:.6f}" if nir_gmsd_val is not None else "N/A"
            nir_fsim_text = f"{nir_fsim_val:.6f}" if nir_fsim_val is not None else "N/A"
            rgb_lpips_text = f"{rgb_lpips_val:.6f}" if rgb_lpips_val is not None else "N/A"
            nir_lpips_text = f"{nir_lpips_val:.6f}" if nir_lpips_val is not None else "N/A"
            metric_text = (
                f"    Epoch: {epoch_num}, =====PSNR: {psnr_val:.6f}, SSIM: {ssim_val:.6f}, "
                f"RGB_GMSD: {rgb_gmsd_val:.6f}, NIR_GMSD: {nir_gmsd_text}, "
                f"RGB_FSIM: {rgb_fsim_val:.6f}, NIR_FSIM: {nir_fsim_text}, "
                f"RGB_LPIPS: {rgb_lpips_text}, NIR_LPIPS: {nir_lpips_text}, "
                f"SAM: {sam_val:.6f}====="
            )
            print(metric_text)

            log_info(
                log_path=VAE_TRAIN_INFO_PATH,
                text=f">>> {metric_text.strip()}",
            )
            log_info(
                log_path=VAE_TRAIN_INFO_PATH,
                text='',
            )   # Insert a blank line between epochs in the log file.

            save_rgb_datas(
                prepare_rgb_vis_tensor(recon_imgs, masks=draw_save_msks).cpu(), nrow=3,
                savepath=os.path.join(VAE_RGB_DIR, f'VAE_RGB_{epoch_num}.png'),
                is_showminmax=True,
            )

            if epoch_num % 25 == 0:
                save_tif_datas(
                    recon_imgs.cpu(), draw_save_projs, draw_save_geos,
                    savepath=os.path.join(VAE_TIF_DIR, f'VAE_TIF_{epoch_num}.tif'),
                    is_showminmax=True,
                    masks=draw_save_msks.cpu(),
                )

        if training_state['max_iter_reached']:
            print(f"Reached TRAIN_MAX_ITERATIONS ({TRAIN_MAX_ITERATIONS}), stopping training.")
            break

        if is_break_training:
            break

        time.sleep(5)

    writer.close()


if __name__ == '__main__':
    
    main(resume_train=False)
