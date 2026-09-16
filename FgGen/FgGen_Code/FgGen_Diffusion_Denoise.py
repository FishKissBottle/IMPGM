import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import *
apply_task_config(globals(), FGGEN_DIFFUSION_CONFIG)
from FgGen.FgGen_Code.FgGen_Diffusion import FgGen_Diffusion_UNet
from IMPGM_Inference_Helpers import run_fggen_diffusion_denoise
from IMPGM_Scheduler import build_beta_schedule
from IMPGM_Utils import (
    save_rgb_datas,
    save_tif_datas,
    prepare_rgb_vis_tensor,
)


@torch.no_grad()
def FgGen_Diffusion_Denoise(
    sampler_mode,
    prompt_str,
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
    decoded, latent = run_fggen_diffusion_denoise(
        sampler_mode=sampler_mode,
        prompt_str=prompt_str,
        model_builder=lambda: FgGen_Diffusion_UNet(
            ch=MODEL_CH,
            out_ch=MODEL_OUT_CH,
            ch_mult=MODEL_CH_MULT,
            attn_resolutions=MODEL_ATTN_RESOLUTIONS,
            dropout=MODEL_DROPOUT,
            resamp_with_conv=MODEL_RESAMP_WITH_CONV,
            in_channels=MODEL_IN_CHANNELS,
            resolution=MODEL_RESOLUTION,
            prompt_dict=PROMPT_DICT,
        ).to(DEVICE),
        model_savepath=MODEL_SAVEPATH,
        beta_t=build_beta_schedule(
            scheduler_type=SCHEDULER_TYPE,
            timesteps=STEPS,
            power_val=SCHEDULER_POWER_VAL,
            min_beta=SCHEDULER_MIN_BETA,
            max_beta=SCHEDULER_MAX_BETA,
            cosine_s=SCHEDULER_COSINE_S,
            sigmoid_start=SCHEDULER_SIGMOID_START,
            sigmoid_end=SCHEDULER_SIGMOID_END,
        ),
        rgb_save_path=rgb_save_path,
        tif_save_path=tif_save_path,
        need_to_decode=need_to_decode,
        seed=seed,
        diffusion_model=diffusion_model,
        vae_model=vae_model,
        initial_noise=initial_noise,
        generators=generators,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
    )

    if rgb_save_path is not None and need_to_decode:
        rgb_save_path = str(Path(rgb_save_path).with_suffix('.png'))
        save_rgb_datas(
            prepare_rgb_vis_tensor(decoded),
            nrow=3,
            savepath=rgb_save_path,
            is_showminmax=True,
        )
    if tif_save_path is not None and need_to_decode:
        save_tif_datas(
            decoded,
            projections=None,
            geotransforms=None,
            savepath=tif_save_path,
            is_showminmax=True,
        )

    if rgb_save_path is None and tif_save_path is None and need_to_decode:
        return decoded
    if rgb_save_path is None and tif_save_path is None and not need_to_decode:
        return decoded, latent
    return decoded
