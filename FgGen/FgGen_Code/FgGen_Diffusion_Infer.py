import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IMPGM_Config import *
apply_task_config(globals(), FGGEN_DIFFUSION_CONFIG)
from FgGen.FgGen_Code.FgGen_Diffusion_Denoise import FgGen_Diffusion_Denoise
from FgGen.FgGen_Code.FgGen_Diffusion import FgGen_Diffusion_UNet
from IMPGM_Dataset import IMPGM_Dataset
from IMPGM_Utils import (
    build_dataloader_generator,
    load_model_for_eval,
    load_standard_vae,
    prepare_rgb_vis_tensor,
    save_rgb_datas,
    save_tif_datas,
    seed_dataloader_worker,
    set_random_seed,
)
from osgeo import gdal
from torch.utils.data import DataLoader

gdal.UseExceptions()


@torch.no_grad()
def FgGen_Diffusion_Batch_Infer(sampler_mode, data_loader, rgb_save_path, tif_save_path, base_seed=RANDOM_SEED):
    rgb_path = Path(rgb_save_path)
    tif_path = Path(tif_save_path)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    tif_path.parent.mkdir(parents=True, exist_ok=True)

    set_random_seed(base_seed, deterministic=False)
    diffusion_model = FgGen_Diffusion_UNet(
        ch=MODEL_CH,
        out_ch=MODEL_OUT_CH,
        ch_mult=MODEL_CH_MULT,
        attn_resolutions=MODEL_ATTN_RESOLUTIONS,
        dropout=MODEL_DROPOUT,
        resamp_with_conv=MODEL_RESAMP_WITH_CONV,
        in_channels=MODEL_IN_CHANNELS,
        resolution=MODEL_RESOLUTION,
        prompt_dict=PROMPT_DICT,
    ).to(DEVICE)
    diffusion_model, _ = load_model_for_eval(
        MODEL_SAVEPATH,
        diffusion_model,
        map_location=DEVICE,
    )
    vae_model, _, _ = load_standard_vae(
        VAE_MODEL_SAVEPATH,
        device=DEVICE,
        load_ema=True,
    )
    for batch_idx, (labs, _, _, _, _, _) in enumerate(data_loader):
        batch_seed = base_seed + batch_idx
        decoded = FgGen_Diffusion_Denoise(
            sampler_mode=sampler_mode,
            prompt_str=labs,
            rgb_save_path=None,
            tif_save_path=None,
            seed=batch_seed,
            diffusion_model=diffusion_model,
            vae_model=vae_model,
        )
        edited_rgb_save_path = str(rgb_path.with_name(f'{rgb_path.stem}_b{batch_idx + 1}.png'))
        edited_tif_save_path = str(
            tif_path.with_name(f'{tif_path.stem}_b{batch_idx + 1}{tif_path.suffix}')
        )
        save_rgb_datas(prepare_rgb_vis_tensor(decoded), nrow=3, savepath=edited_rgb_save_path, is_makegrid=False, prompt_strs=labs)
        save_tif_datas(
            decoded,
            projections=None,
            geotransforms=None,
            savepath=edited_tif_save_path,
            prompt_strs=labs,
        )


if __name__ == '__main__':
    if TASK_NAME != "fggen_diffusion":
        raise RuntimeError(
            f"FgGen_Diffusion_Infer requires fggen_diffusion.yaml, got {TASK_NAME!r}"
        )
    if DATASET_DICT is None:
        raise RuntimeError("DATASET_DICT must not be None.")

    set_random_seed(RANDOM_SEED, deterministic=False)

    test_dataset = IMPGM_Dataset(
        img_rootdir_list=DATASET_DICT["img_rootdir_list_forTest"],
        msk_rootdir_list=DATASET_DICT["msk_rootdir_list_forTest"],
        is_train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(RANDOM_SEED),
    )

    infer_root = PROJECT_ROOT / "FgGen" / "FgGen_Diffusion_Infer"
    rgb_save_path = str(infer_root / "FgGen_Diffusion_Infer_RGBs" / f"{EXP_NAME}.png")
    tif_save_path = str(infer_root / "FgGen_Diffusion_Infer_TIFs" / f"{EXP_NAME}.tif")
    FgGen_Diffusion_Batch_Infer(
        DRAW_SAMPLER_MODE,
        test_loader,
        rgb_save_path,
        tif_save_path,
        base_seed=DRAW_RANDOM_SEED,
    )

