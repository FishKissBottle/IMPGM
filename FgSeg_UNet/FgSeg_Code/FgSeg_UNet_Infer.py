import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from osgeo import gdal

from FgSeg_UNet.FgSeg_Code.FgSeg_UNet_model import FgSeg_UNet
from IMPGM_Config import (
    DEVICE,
    FGSEG_RGB_DIR,
    FGSEG_TIF_DIR,
    FGSEG_UNET_MODEL_SAVEPATH,
    INPUT_CHANNELS,
    UNET_OUTPUT_CHANNELS,
    transform_only_tif,
)
from IMPGM_TifReader import Tif_Read_and_Write
from IMPGM_Utils import build_train_autocast, load_model_for_eval, save_msk_datas

gdal.UseExceptions()


def _load_normalized_tif(input_path):
    image, projection, geotransform = Tif_Read_and_Write().Tif_Read(
        str(input_path)
    )
    if not isinstance(projection, str) or not projection.strip():
        raise ValueError(
            f"Input TIF has an empty projection string: {input_path}"
        )
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    if image.ndim != 3:
        raise ValueError(
            f"Expected a band-first TIF array, got shape {image.shape}."
        )
    if image.shape[0] != INPUT_CHANNELS:
        raise ValueError(
            f"Expected {INPUT_CHANNELS} input bands, got {image.shape[0]} "
            f"from {input_path}."
        )
    height, width = image.shape[-2:]
    assert height % 8 == 0 and width % 8 == 0, (
        "FgSeg_UNet requires input height and width to be divisible by 8, "
        f"got H={height}, W={width}."
    )

    image_hwc = np.transpose(image.astype(np.float32, copy=False), (1, 2, 0))
    normalized = transform_only_tif(image=image_hwc)["image"].unsqueeze(0)
    if not torch.isfinite(normalized).all():
        non_finite_count = int((~torch.isfinite(normalized)).sum().item())
        normalized = torch.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        print(
            f"[FgSeg] Replaced {non_finite_count} non-finite input value(s) "
            "with model-space background value 0."
        )
    return normalized, projection, geotransform


def _build_model():
    """Build FgSeg and load its EMA weights for inference."""
    checkpoint_path = Path(FGSEG_UNET_MODEL_SAVEPATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"FgSeg checkpoint does not exist: {checkpoint_path}"
        )

    model = FgSeg_UNet(
        in_channels=INPUT_CHANNELS,
        out_channels=UNET_OUTPUT_CHANNELS,
    ).to(DEVICE)
    model, _ = load_model_for_eval(
        checkpoint_path,
        model,
        map_location=DEVICE,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def infer_tif(input_path, png_path=None, tif_path=None, threshold=0.5):
    """Predict a binary foreground mask and save PNG and georeferenced TIF."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input TIF does not exist: {input_path}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"`threshold` must be in [0, 1], got {threshold}.")

    default_name = f"{input_path.stem}_FgSeg_Mask"
    png_path = (
        Path(png_path)
        if png_path is not None
        else Path(FGSEG_RGB_DIR) / "Infer" / f"{default_name}.png"
    )
    tif_path = (
        Path(tif_path)
        if tif_path is not None
        else Path(FGSEG_TIF_DIR) / "Infer" / f"{default_name}.tif"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tif_path.parent.mkdir(parents=True, exist_ok=True)

    image, projection, geotransform = _load_normalized_tif(input_path)
    image = image.to(DEVICE)
    model = _build_model()

    with build_train_autocast():
        logits = model(image)
    predicted_mask = (
        torch.sigmoid(logits.float()) >= float(threshold)
    ).to(torch.float32)

    save_msk_datas(
        predicted_mask.cpu(),
        savepath=str(png_path),
        is_makegrid=True,
        nrow=1,
    )
    save_msk_datas(
        predicted_mask.cpu(),
        savepath=str(tif_path),
        is_makegrid=False,
        projections=[projection],
        geotransforms=[geotransform],
    )
    print(f"Saved FgSeg PNG: {png_path}")
    print(f"Saved FgSeg TIF: {tif_path}")
    return predicted_mask


def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Run FgSeg inference on one georeferenced TIF."
    )
    parser.add_argument("input_tif", help="Path to the input foreground TIF.")
    parser.add_argument("--png-path", default=None, help="Optional PNG output path.")
    parser.add_argument("--tif-path", default=None, help="Optional TIF output path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary-mask probability threshold.",
    )
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    infer_tif(
        args.input_tif,
        png_path=args.png_path,
        tif_path=args.tif_path,
        threshold=args.threshold,
    )
