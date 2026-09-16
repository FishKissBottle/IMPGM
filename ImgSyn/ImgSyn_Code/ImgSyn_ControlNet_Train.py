import sys
import os
from contextlib import contextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time
import numpy as np

from IMPGM_Config import *
apply_task_config(globals(), IMGSYN_CONTROLNET_CONFIG)
from torch.utils.data import DataLoader
from IMPGM_Dataset import IMPGM_Dataset
from ImgSyn.ImgSyn_Code.ImgSyn_ControlNet import ControlNet_on_ImgSyn_Diffusion
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from ImgSyn.ImgSyn_Code.ImgSyn_ControlNet_Denoise import ImgSyn_ControlNet_Denoise
from osgeo import gdal
from IMPGM_Scheduler import build_beta_schedule
from IMPGM_Pixel_Losses import (
    compute_timestep_gate,
    sobel_edge_loss_pixel,
    prepare_pixel_tensors,
)
from IMPGM_Utils import (
    build_dataloader_generator,
    build_ema_model,
    build_scheduler,
    build_torch_generator,
    build_train_autocast,
    encode_to_scaled_latent,
    extract_diffusion_coefficient,
    extract_high_frequency,
    get_training_monitor_state,
    iter_microbatch_slices,
    load_controlnet_model_for_eval,
    load_model,
    load_standard_vae,
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
    prepare_rgb_vis_tensor,
    randn,
)

from torch.utils.tensorboard import SummaryWriter

gdal.UseExceptions()


@contextmanager
def use_controlnet_branch(model, controlnet_branch):
    original_branch = model.ControlNet_model
    was_training = model.training
    model.ControlNet_model = controlnet_branch
    try:
        yield model
    finally:
        model.ControlNet_model = original_branch
        model.train(was_training)


def _maybe_preload_controlnet_weights(model):
    """Optionally initialize the ImgSyn ControlNet branch from EMA weights."""
    if not PRELOAD_ENABLED:
        return
    if not PRELOAD_SOURCE_PATH:
        raise ValueError("ImgSyn ControlNet preloading is enabled but source_path is empty.")
    if not Path(PRELOAD_SOURCE_PATH).is_file():
        raise FileNotFoundError(f"ImgSyn ControlNet preload source not found: {PRELOAD_SOURCE_PATH}")

    load_controlnet_model_for_eval(
        PRELOAD_SOURCE_PATH,
        model,
        branch_attr="ControlNet_model",
        map_location=DEVICE,
        skip_prefix=PRELOAD_SKIP_PREFIX or None,
        strict=not bool(PRELOAD_SKIP_PREFIX),
    )
    model.train()
    print(f"Preloading ControlNet EMA weights from: {Path(PRELOAD_SOURCE_PATH).stem}")


@torch.no_grad()
def _draw_imgsyn_controlnet_microbatched(
    draw_labs,
    draw_fg_imgs_e,
    draw_syn_imgs_c,
    controlnet_model,
    vae_model,
    draw_projs=None,
    draw_geos=None,
):
    """Generate fixed-noise draw samples from the in-memory EMA model."""
    batch_size = len(draw_labs)
    microbatch_size = resolve_microbatch_size(DIFFUSION_DRAW_MICROBATCH_SIZE, batch_size)
    draw_outputs = []
    rng_state = stash_rng_state()
    try:
        set_random_seed(DRAW_RANDOM_SEED, deterministic=False)
        generator = build_torch_generator(DRAW_RANDOM_SEED, DEVICE)
        initial_noise = randn(
            (batch_size, LATENT_HIDDENCHANNEL, LATENT_RESOLUTION, LATENT_RESOLUTION),
            device=DEVICE,
            generator=generator,
        )
        for mb_start, mb_end in iter_microbatch_slices(batch_size, microbatch_size):
            draw_outputs.append(
                ImgSyn_ControlNet_Denoise(
                    sampler_mode=DRAW_SAMPLER_MODE,
                    prompt_str=slice_microbatch(draw_labs, mb_start, mb_end),
                    fg_imgs_e=slice_microbatch(draw_fg_imgs_e, mb_start, mb_end),
                    conditional_element=slice_microbatch(draw_syn_imgs_c, mb_start, mb_end),
                    projs=slice_microbatch(draw_projs, mb_start, mb_end),
                    geos=slice_microbatch(draw_geos, mb_start, mb_end),
                    seed=DRAW_RANDOM_SEED + mb_start,
                    controlnet_model=controlnet_model,
                    vae_model=vae_model,
                    initial_noise=initial_noise[mb_start:mb_end],
                )
            )
    finally:
        restore_rng_state(rng_state)
    return torch.cat(draw_outputs, dim=0)


class ControlNetLossCalculator(nn.Module):
    """Compute the training loss for the ImgSyn ControlNet."""
    def __init__(self, model, beta_t, vae_model=None, latent_scaling_factor=1.0):
        super().__init__()
        self.model = model
        self.T = len(beta_t)

        # Store VAE in a list to avoid registering it as a submodule of the loss calculator.
        self._vae_model = [vae_model]
        self._latent_scaling_factor = latent_scaling_factor

        # Register the beta schedule for all T diffusion steps.
        self.register_buffer("beta_t", beta_t)

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # Calculate and store the two coefficients of q(x_t | x_0).
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def _compute_pixel_losses(self, pred_x_0_latent, x_0_latent, t):
        vae_model = self._vae_model[0]
        if vae_model is None:
            raise ValueError("VAE model is required for pixel-space auxiliary losses.")

        pred_pixel, target_pixel = prepare_pixel_tensors(
            vae_model,
            pred_x_0_latent,
            x_0_latent,
            self._latent_scaling_factor,
        )

        loss_edge_per = sobel_edge_loss_pixel(pred_pixel, target_pixel)

        gate = compute_timestep_gate(t, self.T)
        loss_edge = (loss_edge_per * gate).mean()

        return loss_edge

    def forward(self, x_0, prompt_str, fg_imgs_e, conditional_element):
        # Sample t uniformly from {0, ..., T-1}
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # Construct x_t directly from x_0 via q_sample; the model predicts v_t (v-parameterization)
        x_t = (extract_diffusion_coefficient(self.signal_rate, t, x_0.shape) * x_0 + extract_diffusion_coefficient(self.noise_rate, t, x_0.shape) * epsilon)

        v_t = extract_diffusion_coefficient(self.signal_rate, t, x_0.shape) * epsilon - extract_diffusion_coefficient(self.noise_rate, t, x_0.shape) * x_0

        pred_v_t = self.model(x_t, t, prompt_str, fg_imgs_e, conditional_element)

        pred_x_0 = (extract_diffusion_coefficient(self.signal_rate, t, x_0.shape) * x_t - extract_diffusion_coefficient(self.noise_rate, t, x_0.shape) * pred_v_t) / (extract_diffusion_coefficient(self.signal_rate, t, x_0.shape) ** 2 + extract_diffusion_coefficient(self.noise_rate, t, x_0.shape) ** 2)

        loss_edge = self._compute_pixel_losses(pred_x_0, x_0, t)

        sp_loss = F.mse_loss(pred_v_t, v_t)
        loss = sp_loss + loss_edge * EDGE_LOSS_WEIGHT

        return loss, sp_loss, loss_edge


def train(
    data_loader,
    Latent_model,
    optimizer,
    loss_calculator,
    latent_scaling_factor,
    ema_control_model=None,
    training_state=None,
):

    train_loss_sum = 0.
    train_sp_loss_sum = 0.
    train_edge_loss_sum = 0.
    train_sample_count = 0
    with tqdm(data_loader, dynamic_ncols=False, colour="#ff924a", leave=False, ncols=120) as data:
        for labs, fg_imgs, msks, syn_imgs, _, _ in data:
            batch_size = syn_imgs.shape[0]
            microbatch_size = resolve_microbatch_size(DIFFUSION_TRAIN_MICROBATCH_SIZE, batch_size)
            optimizer.zero_grad(set_to_none=True)
            batch_loss_sum = 0.0
            batch_sp_loss_sum = 0.0
            batch_edge_loss_sum = 0.0
            for mb_start, mb_end in iter_microbatch_slices(batch_size, microbatch_size):
                mb = int(mb_end - mb_start)
                microbatch_weight = float(mb) / float(batch_size)
                fg_imgs_mb = slice_microbatch(fg_imgs, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                syn_imgs_mb = slice_microbatch(syn_imgs, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                msks_mb = slice_microbatch(msks, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                syn_imgs_high_freq_mb = extract_high_frequency(
                    syn_imgs_mb, CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE
                )
                labs_mb = slice_microbatch(labs, mb_start, mb_end)
                with build_train_autocast():
                    fg_imgs_e = encode_to_scaled_latent(Latent_model, fg_imgs_mb, latent_scaling_factor)
                    syn_imgs_e = encode_to_scaled_latent(Latent_model, syn_imgs_mb, latent_scaling_factor)
                    syn_imgs_c = syn_imgs_high_freq_mb * (1.0 - msks_mb)
                    loss, sp_loss, edge_loss = loss_calculator(syn_imgs_e, labs_mb, fg_imgs_e, syn_imgs_c)

                (loss * microbatch_weight).backward()
                batch_loss_sum += loss.item() * mb
                batch_sp_loss_sum += sp_loss.item() * mb
                batch_edge_loss_sum += edge_loss.item() * mb
            optimizer.step()
            if ema_control_model is not None:
                update_ema_model(
                    ema_control_model,
                    loss_calculator.model.ControlNet_model,
                    EMA_DECAY,
                )

            batch_loss = batch_loss_sum / batch_size
            batch_sp_loss = batch_sp_loss_sum / batch_size
            batch_edge_loss = batch_edge_loss_sum / batch_size

            train_loss_sum += batch_loss_sum
            train_sp_loss_sum += batch_sp_loss_sum
            train_edge_loss_sum += batch_edge_loss_sum
            train_sample_count += batch_size

            if training_state is not None:
                training_state['iteration'] += 1
                if TRAIN_MAX_ITERATIONS > 0 and training_state['iteration'] >= TRAIN_MAX_ITERATIONS:
                    training_state['max_iter_reached'] = True
                    return train_loss_sum / train_sample_count, train_sp_loss_sum / train_sample_count, train_edge_loss_sum / train_sample_count

            data.set_postfix(ordered_dict={
                'loss':      f'{batch_loss:.6f}',
                'sp_loss':   f'{batch_sp_loss:.6f}',
                'edge_loss': f'{batch_edge_loss:.6f}'
            })

    return train_loss_sum / train_sample_count, train_sp_loss_sum / train_sample_count, train_edge_loss_sum / train_sample_count


def valid(data_loader, Latent_model, loss_calculator, latent_scaling_factor):

    valid_loss_sum = 0.
    valid_sp_loss_sum = 0.
    valid_edge_loss_sum = 0.
    valid_sample_count = 0
    with torch.no_grad():
        for labs, fg_imgs, msks, syn_imgs, _, _ in data_loader:
            batch_size = syn_imgs.shape[0]
            microbatch_size = resolve_microbatch_size(DIFFUSION_VALID_MICROBATCH_SIZE, batch_size)
            for mb_start, mb_end in iter_microbatch_slices(batch_size, microbatch_size):
                mb = int(mb_end - mb_start)
                fg_imgs_mb = slice_microbatch(fg_imgs, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                syn_imgs_mb = slice_microbatch(syn_imgs, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                msks_mb = slice_microbatch(msks, mb_start, mb_end).to(
                    DEVICE,
                    non_blocking=DATA_TRANSFER_NON_BLOCKING,
                )
                syn_imgs_high_freq_mb = extract_high_frequency(
                    syn_imgs_mb, CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE
                )
                labs_mb = slice_microbatch(labs, mb_start, mb_end)

                with build_train_autocast():
                    fg_imgs_e = encode_to_scaled_latent(Latent_model, fg_imgs_mb, latent_scaling_factor)
                    syn_imgs_e = encode_to_scaled_latent(Latent_model, syn_imgs_mb, latent_scaling_factor)
                    syn_imgs_c = syn_imgs_high_freq_mb * (1.0 - msks_mb)
                    loss, sp_loss, edge_loss = loss_calculator(
                        syn_imgs_e,
                        labs_mb,
                        fg_imgs_e,
                        syn_imgs_c,
                    )

                valid_loss_sum += loss.item() * mb
                valid_sp_loss_sum += sp_loss.item() * mb
                valid_edge_loss_sum += edge_loss.item() * mb
                valid_sample_count += mb

    return valid_loss_sum / valid_sample_count, valid_sp_loss_sum / valid_sample_count, valid_edge_loss_sum / valid_sample_count


def main(resume_train=False):
    if TASK_NAME != "imgsyn_controlnet":
        raise RuntimeError(
            f"ImgSyn_ControlNet_Train requires imgsyn_controlnet.yaml, got {TASK_NAME!r}"
        )
    set_random_seed(RANDOM_SEED, deterministic=False)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(TIF_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    is_exist = os.path.exists(MODEL_SAVEPATH)
    if is_exist and not resume_train:
        raise Exception('IMGSYN_CONTROLNET_MODEL exists and resume_train is False ^_^')

    ImgSyn_ControlNet_model = ControlNet_on_ImgSyn_Diffusion(
                                                             ImgSyn_Diffusion_model_savepath=IMGSYN_BASE_DIFFUSION_MODEL_SAVEPATH,
                                                             ch=MODEL_CH,
                                                             out_ch=MODEL_OUT_CH,
                                                             ch_mult=MODEL_CH_MULT,
                                                             attn_resolutions=MODEL_ATTN_RESOLUTIONS,
                                                             dropout=MODEL_DROPOUT,
                                                             resamp_with_conv=MODEL_RESAMP_WITH_CONV,
                                                             in_channels=MODEL_IN_CHANNELS,
                                                             resolution=MODEL_RESOLUTION,
                                                             conditional_ch=MODEL_CONDITIONAL_CH,
                                                             ControlNet_weight=MODEL_CONTROLNET_WEIGHT,
                                                             prompt_dict=PROMPT_DICT
                                                             ).to(DEVICE)

    if not resume_train:
        _maybe_preload_controlnet_weights(ImgSyn_ControlNet_model)

    trainable_params = [
        param for param in ImgSyn_ControlNet_model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
    )

    beta_t = build_beta_schedule(
        scheduler_type=SCHEDULER_TYPE,
        timesteps=STEPS,
        power_val=SCHEDULER_POWER_VAL,
        min_beta=SCHEDULER_MIN_BETA,
        max_beta=SCHEDULER_MAX_BETA,
        cosine_s=SCHEDULER_COSINE_S,
        sigmoid_start=SCHEDULER_SIGMOID_START,
        sigmoid_end=SCHEDULER_SIGMOID_END,
    )

    Latent_model, latent_scaling_factor, _ = load_standard_vae(VAE_MODEL_SAVEPATH, device=DEVICE, load_ema=True)
    Latent_model.eval()
    for param in Latent_model.parameters():
        param.requires_grad = False

    loss_calculator = ControlNetLossCalculator(
        ImgSyn_ControlNet_model, beta_t, vae_model=Latent_model, latent_scaling_factor=latent_scaling_factor
    ).to(DEVICE)
    lr_scheduler = build_scheduler(optimizer, MIN_LEARNING_RATE, PATIENCE_THRESHOLD_NUM)
    ema_control_model = build_ema_model(ImgSyn_ControlNet_model.ControlNet_model)
    resume_extra = {}

    if resume_train:
        ImgSyn_ControlNet_model, optimizer, lr_scheduler, last_epoch, last_loss_dict, resume_extra, _ = load_model(
            MODEL_SAVEPATH,
            ImgSyn_ControlNet_model,
            optimizer,
            scheduler=lr_scheduler,
            resume_training=True,
            preload_only=False,
            map_location=DEVICE,
        )
        ema_state_dict = resume_extra.get("controlnet_ema_state_dict")
        if ema_state_dict is None:
            raise KeyError(
                f"`controlnet_ema_state_dict` was not found in checkpoint: {MODEL_SAVEPATH}"
            )
        ema_control_model.load_state_dict(ema_state_dict, strict=True)
        start_epoch = max(int(last_epoch), 0)
    else:
        start_epoch = 0


    # Load the dataset.
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


    train_dataset = IMPGM_Dataset(img_rootdir_list=img_rootdir_list_forTrain, msk_rootdir_list=msk_rootdir_list_forTrain, is_train=True)
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

    valid_dataset = IMPGM_Dataset(img_rootdir_list=img_rootdir_list_forValid, msk_rootdir_list=msk_rootdir_list_forValid, is_train=False)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(RANDOM_SEED + 1),
    )

    draw_dataset = IMPGM_Dataset(img_rootdir_list=img_rootdir_list_forDraw, msk_rootdir_list=msk_rootdir_list_forDraw, is_train=False)
    draw_loader = DataLoader(
        draw_dataset,
        batch_size=DRAW_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(RANDOM_SEED + 2),
    )

    draw_labs, draw_fg_imgs, draw_msks, draw_syn_imgs, draw_projs, draw_geos = next(iter(draw_loader))
    draw_fg_imgs = draw_fg_imgs.to(
        DEVICE,
        non_blocking=DATA_TRANSFER_NON_BLOCKING,
    )
    draw_msks = draw_msks.to(
        DEVICE,
        non_blocking=DATA_TRANSFER_NON_BLOCKING,
    )
    draw_syn_imgs = draw_syn_imgs.to(
        DEVICE,
        non_blocking=DATA_TRANSFER_NON_BLOCKING,
    )
    draw_syn_imgs_high_freq = extract_high_frequency(
        draw_syn_imgs, CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE
    )
    draw_syn_imgs_c = draw_syn_imgs_high_freq * (1.0 - draw_msks)
    draw_geos = np.array([geo.numpy() for geo in draw_geos]).T


    draw_fg_imgs_e = Latent_model.encode(draw_fg_imgs)
    draw_fg_imgs_e, _, _ = Latent_model.reparameterize(draw_fg_imgs_e)

    ImgSyn_ControlNet_model.train()
    training_state = {
        "iteration": int(
            resume_extra.get("global_iteration", start_epoch * len(train_loader))
        ),
        "max_iter_reached": False,
    }

    for epoch_idx in range(start_epoch, TRAIN_MAX_EPOCHS):
        epoch_num = epoch_idx + 1
        train_generator.manual_seed(RANDOM_SEED + epoch_idx)

        # Get the current learning rate.
        global current_lr
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        if resume_train and epoch_idx == start_epoch:
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
            load_dict = {
                'min_loss': last_loss_dict.get(
                    'min_loss',
                    last_loss_dict.get('valid_loss', float('inf')),
                ),
                'best_epoch_idx': resume_best_epoch_idx,
                'patience_counter': int(
                    resume_extra.get(
                        'patience_counter',
                        last_loss_dict.get('patience_counter', 0),
                    )
                ),
                'patience_counter_after_min_lr': int(
                    resume_extra.get(
                        'patience_counter_after_min_lr',
                        last_loss_dict.get('patience_counter_after_min_lr', 0),
                    )
                ),
            }
            is_break_training, is_savemodel, patience_counter, patience_counter_after_min_lr = training_monitor(epoch_idx, MIN_LEARNING_RATE, current_lr, float('inf'), PATIENCE_THRESHOLD_NUM, resume_train=resume_train, load_dict=load_dict)
            train_loss, train_sp_loss, train_edge_loss = train(
                train_loader,
                Latent_model,
                optimizer,
                loss_calculator,
                latent_scaling_factor,
                ema_control_model=ema_control_model,
                training_state=training_state,
            )
            eval_rng_state = stash_rng_state()
            if DETERMINISTIC_EVAL:
                set_random_seed(RANDOM_SEED, deterministic=False)
            with use_controlnet_branch(ImgSyn_ControlNet_model, ema_control_model):
                ImgSyn_ControlNet_model.eval()
                valid_loss, valid_sp_loss, valid_edge_loss = valid(
                    valid_loader,
                    Latent_model,
                    loss_calculator,
                    latent_scaling_factor,
                )
            restore_rng_state(eval_rng_state)
            ImgSyn_ControlNet_model.train()
            is_break_training, is_savemodel, patience_counter, patience_counter_after_min_lr = training_monitor(epoch_idx, MIN_LEARNING_RATE, current_lr, valid_loss, PATIENCE_THRESHOLD_NUM, resume_train=False, load_dict=None)
        else:
            train_loss, train_sp_loss, train_edge_loss = train(
                train_loader,
                Latent_model,
                optimizer,
                loss_calculator,
                latent_scaling_factor,
                ema_control_model=ema_control_model,
                training_state=training_state,
            )
            eval_rng_state = stash_rng_state()
            if DETERMINISTIC_EVAL:
                set_random_seed(RANDOM_SEED, deterministic=False)
            with use_controlnet_branch(ImgSyn_ControlNet_model, ema_control_model):
                ImgSyn_ControlNet_model.eval()
                valid_loss, valid_sp_loss, valid_edge_loss = valid(
                    valid_loader,
                    Latent_model,
                    loss_calculator,
                    latent_scaling_factor,
                )
            restore_rng_state(eval_rng_state)
            ImgSyn_ControlNet_model.train()
            is_break_training, is_savemodel, patience_counter, patience_counter_after_min_lr = training_monitor(epoch_idx, MIN_LEARNING_RATE, current_lr, valid_loss, PATIENCE_THRESHOLD_NUM, resume_train=False, load_dict=None)

        # Update the learning rate scheduler.
        lr_scheduler.step(valid_loss)
        monitor_state = get_training_monitor_state()

        _tag_prefix = EXP_NAME
        writer.add_scalars('ImgSyn_ControlNet/train_loss',     {f"{_tag_prefix}_train_loss": train_loss},         epoch_num)
        writer.add_scalars('ImgSyn_ControlNet/train_sp_loss',  {f"{_tag_prefix}_train_sp_loss": train_sp_loss},   epoch_num)
        writer.add_scalars('ImgSyn_ControlNet/train_edge_loss', {f"{_tag_prefix}_train_edge_loss": train_edge_loss}, epoch_num)
        writer.add_scalars('ImgSyn_ControlNet/valid_loss',      {f"{_tag_prefix}_valid_loss": valid_loss},           epoch_num)
        writer.add_scalars('ImgSyn_ControlNet/valid_sp_loss',   {f"{_tag_prefix}_valid_sp_loss": valid_sp_loss},     epoch_num)
        writer.add_scalars('ImgSyn_ControlNet/valid_edge_loss', {f"{_tag_prefix}_valid_edge_loss": valid_edge_loss}, epoch_num)

        print(f'   Epoch: {epoch_num}, Iteration: {training_state["iteration"]}, Pc: {patience_counter}, Pc_min_lr: {patience_counter_after_min_lr}, Lr: {current_lr:.8f}, Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}, Valid_Sp_loss: {valid_sp_loss:.6f}, Valid_Edge_loss: {valid_edge_loss:.6f}')

        log_info(log_path=TRAIN_INFO_PATH,
                 text=f'>>> Epoch: {epoch_num}, Pc: {patience_counter}, Pc_min_lr: {patience_counter_after_min_lr}, Lr: {current_lr:.8f}, Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}, Valid_Sp_loss: {valid_sp_loss:.6f}, Valid_Edge_loss: {valid_edge_loss:.6f}')
        if epoch_num % DRAW_INTERVAL_EPOCHS != 0:
            log_info(log_path=TRAIN_INFO_PATH,
                    text = '')     # Insert a blank line between epochs in the log file.
            
        if epoch_num == 1:
            rgb_save_path = os.path.join(RGB_DIR, 'ImgSyn_ControlNet_RGB_1.png')
            tif_save_path = os.path.join(TIF_DIR, 'ImgSyn_ControlNet_TIF_1.tif')
            con_save_path = os.path.join(RGB_DIR, 'ImgSyn_ControlNet_Condition_1.png')
            save_rgb_datas(prepare_rgb_vis_tensor(draw_syn_imgs), nrow=3, savepath=rgb_save_path)
            save_rgb_datas(prepare_rgb_vis_tensor(draw_syn_imgs_c), nrow=3, savepath=con_save_path)
            if SAVE_TIF_IMAGES:
                save_tif_datas(
                    draw_syn_imgs,
                    projections=draw_projs,
                    geotransforms=draw_geos,
                    savepath=tif_save_path,
                )

        if is_savemodel or epoch_num % TIF_INTERVAL_EPOCHS == 0:
            loss_dict = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'min_loss': monitor_state['min_loss'],
            }
            save_model(
                MODEL_SAVEPATH,
                epoch_num,
                model=ImgSyn_ControlNet_model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                loss_dict=loss_dict,
                extra={
                    **PRELOAD_METADATA,
                    'controlnet_ema_state_dict': ema_control_model.state_dict(),
                    'ema_decay': EMA_DECAY,
                    'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                    'patience_counter': monitor_state['patience_counter'],
                    'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                    'latent_scaling_factor': latent_scaling_factor,
                    'global_iteration': int(training_state['iteration']),
                    'random_seed': RANDOM_SEED,
                },
            )

        if epoch_num in EXTRA_MODEL_SAVE_EPOCH_NUM:
            loss_dict = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'min_loss': monitor_state['min_loss'],
            }
            save_model(
                MODEL_SAVEPATH.replace('.pth', '') + f'_{epoch_num}e.pth',
                epoch_num,
                model=ImgSyn_ControlNet_model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                loss_dict=loss_dict,
                extra={
                    **PRELOAD_METADATA,
                    'controlnet_ema_state_dict': ema_control_model.state_dict(),
                    'ema_decay': EMA_DECAY,
                    'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                    'patience_counter': monitor_state['patience_counter'],
                    'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                    'latent_scaling_factor': latent_scaling_factor,
                    'global_iteration': int(training_state['iteration']),
                    'random_seed': RANDOM_SEED,
                },
            )

        if epoch_num % DRAW_INTERVAL_EPOCHS == 0:
            print(f'   ControlNet_ImgSyn at epoch: {epoch_num} --- labels: {draw_labs}')
            with use_controlnet_branch(ImgSyn_ControlNet_model, ema_control_model):
                ImgSyn_ControlNet_model.eval()
                Denoised_Syn_Imgs = _draw_imgsyn_controlnet_microbatched(
                    draw_labs,
                    draw_fg_imgs_e,
                    draw_syn_imgs_c,
                    controlnet_model=ImgSyn_ControlNet_model,
                    vae_model=Latent_model,
                    draw_projs=draw_projs,
                    draw_geos=draw_geos,
                )
            save_rgb_datas(prepare_rgb_vis_tensor(Denoised_Syn_Imgs), nrow=3, 
                           savepath=os.path.join(RGB_DIR, f'ImgSyn_ControlNet_RGB_{epoch_num}.png'),
                           is_showminmax=True)
            if SAVE_TIF_IMAGES and epoch_num % TIF_INTERVAL_EPOCHS == 0:
                save_tif_datas(
                    Denoised_Syn_Imgs,
                    projections=draw_projs,
                    geotransforms=draw_geos,
                    savepath=os.path.join(TIF_DIR, f'ImgSyn_ControlNet_TIF_{epoch_num}.tif'),
                    is_showminmax=True,
                )
                
        if training_state['max_iter_reached']:
            print(f"Reached TRAIN_MAX_ITERATIONS ({TRAIN_MAX_ITERATIONS}), stopping training.")
            break

        if is_break_training:
            break

        time.sleep(5)

    writer.close()


if __name__ == "__main__":
    main(resume_train=False)
