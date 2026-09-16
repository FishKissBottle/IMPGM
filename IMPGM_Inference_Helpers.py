import torch

from Diffusion_Sampler import (
    DEFAULT_DDIM_ETA,
    DEFAULT_DDIM_STEPS,
    DDIMSampler,
    DDPMSampler,
)
from IMPGM_Config import DEVICE, IMG_SIZE, LATENT_HIDDENCHANNEL, RANDOM_SEED, SIDELENGTH_SCALE_FACTOR
from IMPGM_Utils import (
    build_torch_generator,
    decode_from_scaled_latent,
    load_controlnet_model_for_eval,
    load_model_for_eval,
    load_standard_vae,
    normalize_latent_scaling_factor,
    randn,
    scale_latent,
    set_random_seed,
)


def _build_sampler(sampler_mode, model, beta_t, is_imgsyn=False, is_with_controlnet=False, generator=None):
    """Build the configured DDPM or DDIM sampler."""
    sampler_mode = str(sampler_mode).lower()
    if sampler_mode == "ddpm":
        sampler = DDPMSampler(
            model,
            beta_t,
            is_ImgSyn=is_imgsyn,
            is_with_ControlNet=is_with_controlnet,
            generator=generator,
        ).to(DEVICE)
    elif sampler_mode == "ddim":
        sampler = DDIMSampler(
            model,
            beta_t,
            is_ImgSyn=is_imgsyn,
            is_with_ControlNet=is_with_controlnet,
            generator=generator,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unsupported sampler_mode: {sampler_mode}")
    return sampler


def _ddim_forward_kwargs(sampler_mode, ddim_steps=None, ddim_eta=None):
    """Resolve optional DDIM arguments without changing the DDPM call path."""
    sampler_mode = str(sampler_mode).lower()
    if sampler_mode != "ddim":
        if ddim_steps is not None or ddim_eta is not None:
            raise ValueError("DDIM steps and eta may only be set when sampler_mode='ddim'.")
        return {}

    steps = DEFAULT_DDIM_STEPS if ddim_steps is None else int(ddim_steps)
    eta = DEFAULT_DDIM_ETA if ddim_eta is None else float(ddim_eta)
    if steps <= 0:
        raise ValueError(f"DDIM steps must be positive, got {steps}.")
    if eta < 0.0:
        raise ValueError(f"DDIM eta must be non-negative, got {eta}.")
    return {"steps": steps, "eta": eta}


def _randn_batch(shape, generators):
    if len(generators) != shape[0]:
        raise ValueError(
            "The number of random generators must match the batch size: "
            f"{len(generators)} vs {shape[0]}."
        )
    sample_shape = (1, *shape[1:])
    return torch.cat(
        [randn(sample_shape, device=DEVICE, generator=generator) for generator in generators],
        dim=0,
    )


@torch.no_grad()
def run_fggen_diffusion_denoise(
    sampler_mode,
    prompt_str,
    model_builder,
    model_savepath,
    beta_t,
    rgb_save_path=None,
    tif_save_path=None,
    need_to_decode=True,
    seed=RANDOM_SEED,
    diffusion_model=None,
    vae_model=None,
    initial_noise=None,
    generators=None,
    ddim_steps=None,
    ddim_eta=None,
):
    set_random_seed(seed, deterministic=False)
    generator = (
        list(generators)
        if generators is not None
        else build_torch_generator(seed, DEVICE)
    )

    if diffusion_model is None:
        diffusion_model = model_builder()
        diffusion_model, _ = load_model_for_eval(
            model_savepath,
            diffusion_model,
            map_location=DEVICE,
        )
    diffusion_model = diffusion_model.eval()

    batch_size = len(prompt_str)
    latent_size = IMG_SIZE // SIDELENGTH_SCALE_FACTOR
    expected_shape = (batch_size, LATENT_HIDDENCHANNEL, latent_size, latent_size)
    if initial_noise is None:
        z_t = (
            _randn_batch(expected_shape, generator)
            if isinstance(generator, list)
            else randn(expected_shape, device=DEVICE, generator=generator)
        )
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(
                f"initial_noise shape must be {expected_shape}, got {tuple(initial_noise.shape)}"
            )
        z_t = initial_noise.to(device=DEVICE, dtype=torch.float32)
    if isinstance(generator, list):
        generator = [
            build_torch_generator(item.initial_seed(), DEVICE)
            for item in generator
        ]
    sampler = _build_sampler(sampler_mode, diffusion_model, beta_t, is_imgsyn=False, is_with_controlnet=False, generator=generator)
    sampler_kwargs = {"seed": seed}
    if isinstance(generator, list):
        sampler_kwargs["generator"] = generator
    sampler_kwargs.update(_ddim_forward_kwargs(sampler_mode, ddim_steps, ddim_eta))
    latent = sampler(z_t, prompt_str, is_record_process=False, **sampler_kwargs)

    if not need_to_decode:
        return None, latent

    if vae_model is None and need_to_decode:
        vae_model, latent_scaling_factor, _ = load_standard_vae(device=DEVICE, load_ema=True)
    else:
        if vae_model is not None:
            vae_model = vae_model.eval()
        latent_scaling_factor = normalize_latent_scaling_factor()
    decoded = decode_from_scaled_latent(vae_model, latent, latent_scaling_factor)

    # Low-level helpers always expose both tensors; public wrappers decide
    # whether callers receive only the decoded image or the pair.
    return decoded, latent


@torch.no_grad()
def run_fggen_controlnet_denoise(
    sampler_mode,
    prompt_str,
    conditional_element,
    model_builder,
    model_savepath,
    beta_t,
    rgb_save_path=None,
    tif_save_path=None,
    need_to_decode=True,
    seed=RANDOM_SEED,
    controlnet_model=None,
    vae_model=None,
    initial_noise=None,
    generators=None,
    ddim_steps=None,
    ddim_eta=None,
):
    set_random_seed(seed, deterministic=False)
    generator = (
        list(generators)
        if generators is not None
        else build_torch_generator(seed, DEVICE)
    )

    if controlnet_model is None:
        controlnet_model = model_builder()
        controlnet_model, _ = load_controlnet_model_for_eval(
            model_savepath,
            controlnet_model,
            map_location=DEVICE,
        )
    controlnet_model = controlnet_model.eval()

    batch_size = len(prompt_str)
    latent_size = IMG_SIZE // SIDELENGTH_SCALE_FACTOR
    expected_shape = (batch_size, LATENT_HIDDENCHANNEL, latent_size, latent_size)
    if initial_noise is None:
        z_t = (
            _randn_batch(expected_shape, generator)
            if isinstance(generator, list)
            else randn(expected_shape, device=DEVICE, generator=generator)
        )
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(
                f"initial_noise shape must be {expected_shape}, got {tuple(initial_noise.shape)}"
            )
        z_t = initial_noise.to(device=DEVICE, dtype=torch.float32)
    if isinstance(generator, list):
        generator = [
            build_torch_generator(item.initial_seed(), DEVICE)
            for item in generator
        ]
    sampler = _build_sampler(sampler_mode, controlnet_model, beta_t, is_imgsyn=False, is_with_controlnet=True, generator=generator)
    sampler_kwargs = {
        "conditional_element": conditional_element,
        "seed": seed,
    }
    if isinstance(generator, list):
        sampler_kwargs["generator"] = generator
    sampler_kwargs.update(_ddim_forward_kwargs(sampler_mode, ddim_steps, ddim_eta))
    latent = sampler(z_t, prompt_str, is_record_process=False, **sampler_kwargs)

    if not need_to_decode:
        return None, latent

    if vae_model is None and need_to_decode:
        vae_model, latent_scaling_factor, _ = load_standard_vae(device=DEVICE, load_ema=True)
    else:
        if vae_model is not None:
            vae_model = vae_model.eval()
        latent_scaling_factor = normalize_latent_scaling_factor()
    decoded = decode_from_scaled_latent(vae_model, latent, latent_scaling_factor)

    return decoded, latent


@torch.no_grad()
def run_imgsyn_diffusion_denoise(
    sampler_mode,
    prompt_str,
    fg_imgs_e,
    model_builder,
    model_savepath,
    beta_t,
    rgb_save_path=None,
    tif_save_path=None,
    need_to_decode=True,
    seed=RANDOM_SEED,
    diffusion_model=None,
    vae_model=None,
    initial_noise=None,
    generators=None,
    ddim_steps=None,
    ddim_eta=None,
):
    set_random_seed(seed, deterministic=False)
    generator = (
        list(generators)
        if generators is not None
        else build_torch_generator(seed, DEVICE)
    )

    if diffusion_model is None:
        diffusion_model = model_builder()
        diffusion_model, _ = load_model_for_eval(
            model_savepath,
            diffusion_model,
            map_location=DEVICE,
        )
    diffusion_model = diffusion_model.eval()

    if vae_model is None and need_to_decode:
        vae_model, latent_scaling_factor, _ = load_standard_vae(device=DEVICE, load_ema=True)
    else:
        if vae_model is not None:
            vae_model = vae_model.eval()
        latent_scaling_factor = normalize_latent_scaling_factor()
    fg_imgs_e_scaled = scale_latent(fg_imgs_e.to(DEVICE), latent_scaling_factor)

    batch_size = fg_imgs_e_scaled.shape[0]
    latent_size = IMG_SIZE // SIDELENGTH_SCALE_FACTOR
    expected_shape = (batch_size, LATENT_HIDDENCHANNEL, latent_size, latent_size)
    if initial_noise is None:
        z_t = (
            _randn_batch(expected_shape, generator)
            if isinstance(generator, list)
            else randn(expected_shape, device=DEVICE, generator=generator)
        )
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(
                f"initial_noise shape must be {expected_shape}, got {tuple(initial_noise.shape)}"
            )
        z_t = initial_noise.to(device=DEVICE, dtype=torch.float32)
    if isinstance(generator, list):
        generator = [
            build_torch_generator(item.initial_seed(), DEVICE)
            for item in generator
        ]
    sampler = _build_sampler(sampler_mode, diffusion_model, beta_t, is_imgsyn=True, is_with_controlnet=False, generator=generator)
    sampler_kwargs = {
        "fg_imgs_e": fg_imgs_e_scaled,
        "seed": seed,
    }
    if isinstance(generator, list):
        sampler_kwargs["generator"] = generator
    sampler_kwargs.update(_ddim_forward_kwargs(sampler_mode, ddim_steps, ddim_eta))
    latent = sampler(z_t, prompt_str, is_record_process=False, **sampler_kwargs)

    if not need_to_decode:
        return None, latent

    decoded = decode_from_scaled_latent(vae_model, latent, latent_scaling_factor)

    return decoded, latent


@torch.no_grad()
def run_imgsyn_controlnet_denoise(
    sampler_mode,
    prompt_str,
    fg_imgs_e,
    conditional_element,
    model_builder,
    model_savepath,
    beta_t,
    rgb_save_path=None,
    tif_save_path=None,
    need_to_decode=True,
    seed=RANDOM_SEED,
    controlnet_model=None,
    vae_model=None,
    initial_noise=None,
    ddim_steps=None,
    ddim_eta=None,
):
    set_random_seed(seed, deterministic=False)
    generator = build_torch_generator(seed, DEVICE)

    if controlnet_model is None:
        controlnet_model = model_builder()
        controlnet_model, _ = load_controlnet_model_for_eval(
            model_savepath,
            controlnet_model,
            map_location=DEVICE,
        )
    controlnet_model = controlnet_model.eval()

    if vae_model is None and need_to_decode:
        vae_model, latent_scaling_factor, _ = load_standard_vae(device=DEVICE, load_ema=True)
    else:
        if vae_model is not None:
            vae_model = vae_model.eval()
        latent_scaling_factor = normalize_latent_scaling_factor()
    fg_imgs_e_scaled = scale_latent(fg_imgs_e.to(DEVICE), latent_scaling_factor)

    batch_size = fg_imgs_e_scaled.shape[0]
    latent_size = IMG_SIZE // SIDELENGTH_SCALE_FACTOR
    expected_shape = (batch_size, LATENT_HIDDENCHANNEL, latent_size, latent_size)
    if initial_noise is None:
        z_t = randn(expected_shape, device=DEVICE, generator=generator)
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(
                f"initial_noise shape must be {expected_shape}, got {tuple(initial_noise.shape)}"
            )
        z_t = initial_noise.to(device=DEVICE, dtype=torch.float32)
    sampler = _build_sampler(sampler_mode, controlnet_model, beta_t, is_imgsyn=True, is_with_controlnet=True, generator=generator)
    sampler_kwargs = _ddim_forward_kwargs(sampler_mode, ddim_steps, ddim_eta)
    latent = sampler(
        z_t,
        prompt_str,
        is_record_process=False,
        fg_imgs_e=fg_imgs_e_scaled,
        conditional_element=conditional_element,
        seed=seed,
        **sampler_kwargs,
    )

    if not need_to_decode:
        return None, latent

    decoded = decode_from_scaled_latent(vae_model, latent, latent_scaling_factor)

    return decoded, latent
