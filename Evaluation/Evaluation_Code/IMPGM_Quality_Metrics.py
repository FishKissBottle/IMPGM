import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from piqa import FID, FSIM, GMSD, SSIM

try:
    import lpips as _lpips_backend
except ImportError as exc:
    _lpips_backend = None
    _LPIPS_IMPORT_ERROR = exc
else:
    _LPIPS_IMPORT_ERROR = None

from IMPGM_Config import DEVICE, VIS_BAND_ORDER
from IMPGM_Utils import denormalize_image_tensor
from Quality_Evaluation.swd import swd


QUALITY_EVALUATION_DIR = _PROJECT_ROOT / "Quality_Evaluation"
CLIP_MODEL_DIR = QUALITY_EVALUATION_DIR / "CLIP_vit_base_patch16"
INCEPTION_V3_HUB_DIR = QUALITY_EVALUATION_DIR / "Inception_V3"
INCEPTION_V3_WEIGHTS_PATH = (
    INCEPTION_V3_HUB_DIR / "checkpoints" / "inception_v3_google-0cc3c7bd.pth"
)
_LPIPS_CACHE = {}


def _build_project_fid(device):
    """Build PIQA FID from the project-local Torch Hub checkpoint."""
    if not INCEPTION_V3_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            "Missing project-local Inception V3 weights: "
            f"{INCEPTION_V3_WEIGHTS_PATH}"
        )

    previous_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(INCEPTION_V3_HUB_DIR))
    try:
        metric = FID()
    finally:
        torch.hub.set_dir(previous_hub_dir)

    return metric.to(device)


def require_lpips():
    if _lpips_backend is None:
        raise ImportError(
            "LPIPS evaluation requires the `lpips` package. Install the project "
            "requirements or explicitly run generation evaluation with --skip-lpips."
        ) from _LPIPS_IMPORT_ERROR


def _get_lpips_metric(device, net="alex"):
    """Return a cached LPIPS metric instance on the requested device."""
    require_lpips()

    key = (str(device), str(net))
    if key not in _LPIPS_CACHE:
        _LPIPS_CACHE[key] = _lpips_backend.LPIPS(net=net).to(device).eval()
    return _LPIPS_CACHE[key]


def _validate_image_pair(rebuild_imgs, real_imgs, require_same_batch=True):
    if rebuild_imgs.dim() != 4 or real_imgs.dim() != 4:
        raise ValueError(
            f"Expected 4D tensors (B, C, H, W), got {tuple(rebuild_imgs.shape)} and {tuple(real_imgs.shape)}."
        )

    if rebuild_imgs.shape[1:] != real_imgs.shape[1:]:
        raise ValueError(
            f"Input shape mismatch beyond batch dimension: rebuild={tuple(rebuild_imgs.shape)}, real={tuple(real_imgs.shape)}."
        )

    if require_same_batch and rebuild_imgs.shape[0] != real_imgs.shape[0]:
        raise ValueError(
            f"Batch size mismatch: rebuild batch={rebuild_imgs.shape[0]}, real batch={real_imgs.shape[0]}."
        )


def _validate_4d_images(imgs, name="imgs"):
    if imgs.dim() != 4:
        raise ValueError(f"{name} must be a 4D tensor (B, C, H, W), got {tuple(imgs.shape)}.")


def _validate_3_or_4_channels(imgs):
    if imgs.shape[1] not in (3, 4):
        raise ValueError(f"Only 3-channel or 4-channel inputs are supported, got C={imgs.shape[1]}.")


def _prepare_metric_pair(rebuild_imgs, real_imgs, require_same_batch=True):
    _validate_image_pair(rebuild_imgs, real_imgs, require_same_batch=require_same_batch)
    _validate_3_or_4_channels(rebuild_imgs)

    device = rebuild_imgs.device
    rebuild_imgs = rebuild_imgs.to(device=device, dtype=torch.float32)
    real_imgs = real_imgs.to(device=device, dtype=torch.float32)
    return rebuild_imgs, real_imgs


def _mean_metric_per_sample(metric_output, batch_size):
    if not isinstance(metric_output, torch.Tensor):
        metric_output = torch.as_tensor(metric_output)

    metric_output = metric_output.to(dtype=torch.float32)
    if metric_output.ndim == 0:
        if int(batch_size) != 1:
            raise ValueError(
                f"Expected a batch-aware metric output for batch_size={batch_size}, got a scalar tensor."
            )
        return metric_output.reshape(1)

    if int(metric_output.shape[0]) != int(batch_size):
        if int(metric_output.numel()) == int(batch_size):
            return metric_output.reshape(batch_size)
        raise ValueError(
            f"Metric output shape {tuple(metric_output.shape)} is incompatible with batch_size={batch_size}."
        )

    return metric_output.reshape(batch_size, -1).mean(dim=1)


def _extract_rgb_nir_views(imgs, vis_band_order):
    rgb_imgs = imgs[:, vis_band_order, :, :]
    if imgs.shape[1] == 4:
        remaining = [
            index for index in range(int(imgs.shape[1]))
            if index not in {int(value) for value in vis_band_order}
        ]
        if len(remaining) != 1:
            raise ValueError(
                "NIR metrics require exactly one band outside vis_band_order; "
                f"got vis_band_order={list(vis_band_order)}, remaining={remaining}."
            )
        nir_index = remaining[0]
        nir_imgs = imgs[:, [nir_index, nir_index, nir_index], :, :]
    else:
        nir_imgs = None
    return rgb_imgs, nir_imgs


def _extract_fid_features(fid_metric, imgs, vis_band_order):
    rgb_imgs, nir_imgs = _extract_rgb_nir_views(imgs, vis_band_order)
    rgb_feats = fid_metric.features(rgb_imgs)
    nir_feats = fid_metric.features(nir_imgs) if nir_imgs is not None else None
    return rgb_feats, nir_feats


def _prepare_clip_inputs(clip_processor, imgs, device):
    return clip_processor(images=imgs, return_tensors="pt", do_rescale=False).to(device)


def prepare_inception_band_images(images, band_mode, vis_band_order=None):
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER
    _validate_4d_images(images, name="images")
    _validate_3_or_4_channels(images)
    rgb_indices = [int(index) for index in vis_band_order]
    if band_mode == "rgb":
        if len(rgb_indices) != 3:
            raise ValueError("Inception RGB input requires exactly three visible bands.")
        return images[:, rgb_indices]
    if band_mode == "nir":
        remaining = [
            index for index in range(int(images.shape[1]))
            if index not in set(rgb_indices)
        ]
        if len(remaining) != 1:
            raise ValueError(
                "Inception NIR input requires exactly one band outside "
                f"vis_band_order; got remaining={remaining}."
            )
        return images[:, remaining[0]:remaining[0] + 1].repeat(1, 3, 1, 1)
    raise ValueError(f"Unsupported Inception band mode: {band_mode!r}.")


def build_inception_feature_extractor(device):
    return _build_project_fid(device).eval()


@torch.no_grad()
def extract_inception_features(
    images,
    band_mode="rgb",
    vis_band_order=None,
    feature_extractor=None,
):
    images = images.to(dtype=torch.float32).clamp(0.0, 1.0)
    inception_images = prepare_inception_band_images(
        images,
        band_mode,
        vis_band_order=vis_band_order,
    )
    if feature_extractor is None:
        feature_extractor = build_inception_feature_extractor(images.device)
    return feature_extractor.features(inception_images).float().reshape(images.shape[0], -1)


def compute_KID_from_features(features_a, features_b):
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("KID features must have shape [N, D].")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError("KID feature dimensions must match.")
    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        return None
    features_a = features_a.float()
    features_b = features_b.float()
    dimension = float(features_a.shape[1])
    kernel_aa = (features_a @ features_a.T / dimension + 1.0).pow(3)
    kernel_bb = (features_b @ features_b.T / dimension + 1.0).pow(3)
    kernel_ab = (features_a @ features_b.T / dimension + 1.0).pow(3)
    n = features_a.shape[0]
    m = features_b.shape[0]
    within_a = (kernel_aa.sum() - kernel_aa.diagonal().sum()) / (n * (n - 1))
    within_b = (kernel_bb.sum() - kernel_bb.diagonal().sum()) / (m * (m - 1))
    value = float((within_a + within_b - 2.0 * kernel_ab.mean()).item())
    return value if math.isfinite(value) else None


def compute_FID_from_features(features_a, features_b, device=None):
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("FID features must have shape [N, D].")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError("FID feature dimensions must match.")
    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        return None
    device = torch.device(device or features_a.device)
    metric = _build_project_fid(device)
    value = metric(features_a.to(device), features_b.to(device))
    value = float(value.detach().float().cpu().item())
    return value if math.isfinite(value) else None


def compute_mean_spectrum_SAM_degrees(vector_a, vector_b, eps=1.0e-8):
    """Compute the angle in degrees between two mean-spectrum vectors."""
    vector_a = torch.as_tensor(vector_a, dtype=torch.float32)
    vector_b = torch.as_tensor(vector_b, dtype=torch.float32, device=vector_a.device)
    if vector_a.ndim != 1 or vector_b.ndim != 1 or vector_a.shape != vector_b.shape:
        raise ValueError("Mean-spectrum SAM requires matching one-dimensional vectors.")
    denominator = float(torch.linalg.norm(vector_a) * torch.linalg.norm(vector_b))
    if denominator <= float(eps):
        return None
    cosine = float(torch.dot(vector_a, vector_b) / denominator)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def compute_histogram_W1(hist_a, hist_b, bins=256):
    hist_a = torch.as_tensor(hist_a, dtype=torch.float32)
    hist_b = torch.as_tensor(hist_b, dtype=torch.float32, device=hist_a.device)
    if hist_a.ndim != 1 or hist_b.ndim != 1 or hist_a.shape != hist_b.shape:
        raise ValueError("Histogram W1 requires matching one-dimensional histograms.")
    if int(bins) <= 0 or int(hist_a.numel()) != int(bins):
        raise ValueError("Histogram W1 bins must match the histogram length.")
    cdf_a = hist_a.cumsum(0) / hist_a.sum().clamp_min(1.0)
    cdf_b = hist_b.cumsum(0) / hist_b.sum().clamp_min(1.0)
    return float(torch.abs(cdf_a - cdf_b).sum().item() / int(bins))


def compute_PSNR(rebuild_imgs, real_imgs, max_val=1.0, eps=1e-8):
    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )

    mse_per_sample = torch.mean((rebuild_imgs - real_imgs) ** 2, dim=(1, 2, 3))
    mse_per_sample = torch.clamp(mse_per_sample, min=eps)
    psnr_per_sample = 10 * torch.log10((max_val ** 2) / mse_per_sample)
    return psnr_per_sample.mean()


def compute_SSIM(rebuild_imgs, real_imgs):
    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )

    n_channels = rebuild_imgs.shape[1]
    batch_size = int(rebuild_imgs.shape[0])
    ssim_metric = SSIM(n_channels=n_channels, reduction="none").to(rebuild_imgs.device)
    ssim_per_sample = _mean_metric_per_sample(
        ssim_metric(rebuild_imgs, real_imgs),
        batch_size=batch_size,
    )
    return ssim_per_sample.mean()


def compute_GMSD(rebuild_imgs, real_imgs, vis_band_order=None):
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )

    batch_size = int(rebuild_imgs.shape[0])
    gmsd_metric = GMSD(reduction="none").to(rebuild_imgs.device)

    rebuild_rgb_imgs, rebuild_nir_imgs = _extract_rgb_nir_views(rebuild_imgs, vis_band_order)
    real_rgb_imgs, real_nir_imgs = _extract_rgb_nir_views(real_imgs, vis_band_order)
    rgb_gmsd_val = _mean_metric_per_sample(
        gmsd_metric(rebuild_rgb_imgs, real_rgb_imgs),
        batch_size=batch_size,
    ).mean()

    if rebuild_imgs.shape[1] == 3:
        nir_gmsd_val = None
    elif rebuild_imgs.shape[1] == 4:
        nir_gmsd_val = _mean_metric_per_sample(
            gmsd_metric(rebuild_nir_imgs, real_nir_imgs),
            batch_size=batch_size,
        ).mean()
    else:
        raise Exception("Unexpected number of channels.")

    return rgb_gmsd_val, nir_gmsd_val


def compute_FSIM(rebuild_imgs, real_imgs, vis_band_order=None):
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )

    rebuild_rgb_imgs, rebuild_nir_imgs = _extract_rgb_nir_views(rebuild_imgs, vis_band_order)
    real_rgb_imgs, real_nir_imgs = _extract_rgb_nir_views(real_imgs, vis_band_order)

    batch_size = int(rebuild_imgs.shape[0])
    fsim_rgb_metric = FSIM(chromatic=True, reduction="none").to(rebuild_imgs.device)
    rgb_fsim_val = _mean_metric_per_sample(
        fsim_rgb_metric(rebuild_rgb_imgs, real_rgb_imgs),
        batch_size=batch_size,
    ).mean()

    if rebuild_imgs.shape[1] == 3:
        nir_fsim_val = None
    elif rebuild_imgs.shape[1] == 4:
        fsim_nir_metric = FSIM(chromatic=False, reduction="none").to(rebuild_imgs.device)
        nir_fsim_val = _mean_metric_per_sample(
            fsim_nir_metric(rebuild_nir_imgs, real_nir_imgs),
            batch_size=batch_size,
        ).mean()
    else:
        raise Exception("Unexpected number of channels.")

    return rgb_fsim_val, nir_fsim_val


@torch.no_grad()
def compute_LPIPS(rebuild_imgs, real_imgs, vis_band_order=None, net="alex"):
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )

    lpips_metric = _get_lpips_metric(rebuild_imgs.device, net=net)

    rebuild_rgb_imgs, rebuild_nir_imgs = _extract_rgb_nir_views(rebuild_imgs, vis_band_order)
    real_rgb_imgs, real_nir_imgs = _extract_rgb_nir_views(real_imgs, vis_band_order)

    batch_size = int(rebuild_imgs.shape[0])
    rebuild_rgb_imgs = rebuild_rgb_imgs * 2.0 - 1.0
    real_rgb_imgs = real_rgb_imgs * 2.0 - 1.0
    rgb_lpips_val = _mean_metric_per_sample(
        lpips_metric(rebuild_rgb_imgs, real_rgb_imgs),
        batch_size=batch_size,
    ).mean()

    if rebuild_imgs.shape[1] == 3:
        nir_lpips_val = None
    elif rebuild_imgs.shape[1] == 4:
        rebuild_nir_imgs = rebuild_nir_imgs * 2.0 - 1.0
        real_nir_imgs = real_nir_imgs * 2.0 - 1.0
        nir_lpips_val = _mean_metric_per_sample(
            lpips_metric(rebuild_nir_imgs, real_nir_imgs),
            batch_size=batch_size,
        ).mean()
    else:
        raise Exception("Unexpected number of channels.")

    return rgb_lpips_val, nir_lpips_val


@torch.no_grad()
def compute_FID_by_imgs(rebuild_imgs, real_imgs, vis_band_order=None):
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=False,
    )

    fid_metric = _build_project_fid(rebuild_imgs.device)

    rebuild_rgb_feats, rebuild_nir_feats = _extract_fid_features(
        fid_metric,
        rebuild_imgs,
        vis_band_order,
    )
    real_rgb_feats, real_nir_feats = _extract_fid_features(
        fid_metric,
        real_imgs,
        vis_band_order,
    )

    rgb_fid_val = fid_metric(rebuild_rgb_feats, real_rgb_feats)

    if rebuild_imgs.shape[1] == 3:
        nir_fid_val = None
    elif rebuild_imgs.shape[1] == 4:
        nir_fid_val = fid_metric(rebuild_nir_feats, real_nir_feats)
    else:
        raise Exception("Unexpected number of channels.")

    return rgb_fid_val, nir_fid_val


@torch.no_grad()
def compute_FID_by_datasets(rebuild_imgs, real_imgs_loader, is_ImgSyn=False, vis_band_order=None):
    """
    Compute FID against a real-image DataLoader.

    Inputs are expected in [0, 1]. `rebuild_imgs` is treated as the generated
    sample set used to estimate one FID distribution. If it only contains a
    small batch, the result is still computable but will be statistically noisy.
    """
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    _validate_4d_images(rebuild_imgs, name="rebuild_imgs")
    _validate_3_or_4_channels(rebuild_imgs)

    device = rebuild_imgs.device
    rebuild_imgs = rebuild_imgs.to(device=device, dtype=torch.float32)
    fid_metric = _build_project_fid(device)

    has_nir = rebuild_imgs.shape[1] == 4

    rebuild_rgb_feats, rebuild_nir_feats = _extract_fid_features(
        fid_metric,
        rebuild_imgs,
        vis_band_order,
    )

    real_rgb_feats_list = []
    real_nir_feats_list = [] if has_nir else None

    for _, real_fg_imgs, _, real_base_imgs, _, _ in real_imgs_loader:
        real_imgs = real_base_imgs if is_ImgSyn else real_fg_imgs
        real_imgs = torch.clamp(denormalize_image_tensor(real_imgs), min=0.0, max=1.0)
        real_imgs = real_imgs.to(device=device, dtype=torch.float32)

        if real_imgs.dim() != 4 or real_imgs.shape[1:] != rebuild_imgs.shape[1:]:
            raise ValueError(
                f"DataLoader yielded images with shape {tuple(real_imgs.shape)}; expected (?, {rebuild_imgs.shape[1]}, {rebuild_imgs.shape[2]}, {rebuild_imgs.shape[3]})."
            )

        rgb_feats, nir_feats = _extract_fid_features(
            fid_metric,
            real_imgs,
            vis_band_order,
        )
        real_rgb_feats_list.append(rgb_feats.cpu())

        if has_nir:
            real_nir_feats_list.append(nir_feats.cpu())

    real_rgb_feats = torch.cat(real_rgb_feats_list, dim=0).to(device)
    rgb_fid_val = fid_metric(rebuild_rgb_feats, real_rgb_feats)

    if has_nir:
        real_nir_feats = torch.cat(real_nir_feats_list, dim=0).to(device)
        nir_fid_val = fid_metric(rebuild_nir_feats, real_nir_feats)
    else:
        nir_fid_val = None

    return rgb_fid_val, nir_fid_val


@torch.no_grad()
def compute_CLIP_Cosine_Similarity(
    rebuild_imgs,
    real_imgs,
    vis_band_order=None,
    CLIPmodel_folder_path=CLIP_MODEL_DIR,
):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "CLIP metrics require compatible `transformers` and "
            "`huggingface-hub` installations."
        ) from exc

    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )

    device = rebuild_imgs.device
    CLIPmodel_folder_path = str(CLIPmodel_folder_path)
    clip_model = CLIPModel.from_pretrained(CLIPmodel_folder_path).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIPmodel_folder_path)

    rebuild_rgb_imgs, rebuild_nir_imgs = _extract_rgb_nir_views(rebuild_imgs, vis_band_order)
    real_rgb_imgs, real_nir_imgs = _extract_rgb_nir_views(real_imgs, vis_band_order)

    rebuild_rgb_inputs = _prepare_clip_inputs(clip_processor, rebuild_rgb_imgs, device)
    real_rgb_inputs = _prepare_clip_inputs(clip_processor, real_rgb_imgs, device)
    rebuild_rgb_feat = clip_model.get_image_features(**rebuild_rgb_inputs)
    real_rgb_feat = clip_model.get_image_features(**real_rgb_inputs)
    rgb_clip_cos_sim = torch.nn.functional.cosine_similarity(rebuild_rgb_feat, real_rgb_feat).mean()

    if rebuild_imgs.shape[1] == 3:
        nir_clip_cos_sim = None
    elif rebuild_imgs.shape[1] == 4:
        rebuild_nir_inputs = _prepare_clip_inputs(clip_processor, rebuild_nir_imgs, device)
        real_nir_inputs = _prepare_clip_inputs(clip_processor, real_nir_imgs, device)
        rebuild_nir_feat = clip_model.get_image_features(**rebuild_nir_inputs)
        real_nir_feat = clip_model.get_image_features(**real_nir_inputs)
        nir_clip_cos_sim = torch.nn.functional.cosine_similarity(rebuild_nir_feat, real_nir_feat).mean()
    else:
        raise Exception("Unexpected number of channels.")

    return rgb_clip_cos_sim, nir_clip_cos_sim


def compute_SAM(rebuild_imgs, real_imgs, eps=1e-8):
    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )

    pred = rebuild_imgs.permute(0, 2, 3, 1).reshape(rebuild_imgs.shape[0], -1, rebuild_imgs.shape[1])
    gt = real_imgs.permute(0, 2, 3, 1).reshape(real_imgs.shape[0], -1, real_imgs.shape[1])

    dot = (pred * gt).sum(dim=2)
    pred_norm = torch.linalg.norm(pred, dim=2)
    gt_norm = torch.linalg.norm(gt, dim=2)
    den = pred_norm * gt_norm

    valid = den > eps
    if not torch.any(valid):
        return torch.tensor(float("nan"), device=rebuild_imgs.device, dtype=torch.float32)

    cos_theta = torch.zeros_like(dot)
    cos_theta[valid] = dot[valid] / den[valid]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    sam_deg = torch.rad2deg(torch.arccos(cos_theta))
    sam_deg = torch.where(valid, sam_deg, torch.zeros_like(sam_deg))

    valid_counts = valid.sum(dim=1)
    sam_sum_per_sample = sam_deg.sum(dim=1)
    valid_counts_f = valid_counts.to(dtype=torch.float32).clamp_min(1.0)
    sam_mean_per_sample = sam_sum_per_sample / valid_counts_f
    valid_samples = valid_counts > 0
    if not torch.any(valid_samples):
        return torch.tensor(float("nan"), device=rebuild_imgs.device, dtype=torch.float32)
    return sam_mean_per_sample[valid_samples].mean()


def compute_SWD(rebuild_imgs, real_imgs, vis_band_order=None):
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER

    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=False,
    )

    rebuild_rgb_imgs, rebuild_nir_imgs = _extract_rgb_nir_views(rebuild_imgs, vis_band_order)
    real_rgb_imgs, real_nir_imgs = _extract_rgb_nir_views(real_imgs, vis_band_order)

    rgb_swd_val = swd(rebuild_rgb_imgs, real_rgb_imgs, device=rebuild_imgs.device)

    if rebuild_imgs.shape[1] == 3:
        nir_swd_val = None
    elif rebuild_imgs.shape[1] == 4:
        nir_swd_val = swd(rebuild_nir_imgs, real_nir_imgs, device=rebuild_imgs.device)
    else:
        raise Exception("Unexpected number of channels.")

    return rgb_swd_val, nir_swd_val


def compute_multiband_SWD(rebuild_imgs, real_imgs, seed=0, **swd_kwargs):
    """Compute deterministic SWD jointly over every available image band."""
    rebuild_imgs, real_imgs = _prepare_metric_pair(
        rebuild_imgs,
        real_imgs,
        require_same_batch=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = swd(
        rebuild_imgs,
        real_imgs,
        device=rebuild_imgs.device,
        generator=generator,
        **swd_kwargs,
    )
    return value if torch.isfinite(value).all() else None


@torch.no_grad()
def compute_LPIPS_diversity(sample_group, vis_band_order=None, net="alex"):
    """Measure RGB/NIR perceptual and all-band pixel diversity per condition."""
    if vis_band_order is None:
        vis_band_order = VIS_BAND_ORDER
    _validate_4d_images(sample_group, name="sample_group")
    _validate_3_or_4_channels(sample_group)
    if sample_group.shape[0] < 2:
        raise ValueError("LPIPS diversity requires at least two samples per condition.")

    samples = sample_group.to(dtype=torch.float32).clamp(0.0, 1.0)
    lpips_metric = _get_lpips_metric(samples.device, net=net)
    rgb, nir = _extract_rgb_nir_views(samples, vis_band_order)
    rgb = rgb * 2.0 - 1.0
    nir = None if nir is None else nir * 2.0 - 1.0
    rgb_lpips_values = []
    nir_lpips_values = []
    pixel_values = []
    for left in range(samples.shape[0] - 1):
        for right in range(left + 1, samples.shape[0]):
            pixel_values.append(
                torch.sqrt(torch.mean((samples[left] - samples[right]) ** 2))
            )
            value = lpips_metric(
                rgb[left:left + 1],
                rgb[right:right + 1],
            )
            rgb_lpips_values.append(value.float().mean())
            if nir is not None:
                nir_value = lpips_metric(
                    nir[left:left + 1],
                    nir[right:right + 1],
                )
                nir_lpips_values.append(nir_value.float().mean())

    pixel_distance = torch.stack(pixel_values).mean()
    rgb_lpips_distance = (
        torch.stack(rgb_lpips_values).mean() if rgb_lpips_values else None
    )
    nir_lpips_distance = (
        torch.stack(nir_lpips_values).mean() if nir_lpips_values else None
    )
    return rgb_lpips_distance, nir_lpips_distance, pixel_distance


if __name__ == "__main__":
    device = torch.device(DEVICE)

    from torch.utils.data import DataLoader

    from IMPGM_Dataset import IMPGM_Dataset

    img_rootdir_list_forDraw = [
        r"./data/main/draw/images/Water",
        r"./data/main/draw/images/Cloud",
        # r"./data/main/draw/images/NoCloud",
        # r"./data/main/draw/images/NoObj",
    ]
    msk_rootdir_list_forDraw = [
        r"./data/main/draw/masks/Water",
        r"./data/main/draw/masks/Cloud",
    ]

    img_rootdir_list_forTest = [
        r"./data/main/test/images/Water",
        # r"./data/main/test/images/FewCloud",
        # r"./data/main/test/images/LessCloud",
        # r"./data/main/test/images/MoreCloud",
        # r"./data/main/test/images/ManyCloud",
        # r"./data/main/test/images/NoCloud",
        # r"./data/main/test/images/NoObj",
    ]
    msk_rootdir_list_forTest = [
        r"./data/main/test/masks/Water",
        # r"./data/main/test/masks/FewCloud",
        # r"./data/main/test/masks/LessCloud",
        # r"./data/main/test/masks/MoreCloud",
        # r"./data/main/test/masks/ManyCloud",
    ]

    test_dataset = IMPGM_Dataset(
        img_rootdir_list=img_rootdir_list_forTest,
        msk_rootdir_list=msk_rootdir_list_forTest,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=2)

    draw_dataset = IMPGM_Dataset(
        img_rootdir_list=img_rootdir_list_forDraw,
        msk_rootdir_list=msk_rootdir_list_forDraw,
        is_train=False,
    )
    draw_loader = DataLoader(draw_dataset, batch_size=6, shuffle=False, num_workers=2)

    _, draw_fg_imgs, _, draw_syn_imgs, draw_projs, draw_geos = next(iter(draw_loader))

    reference_imgs = draw_fg_imgs.to(device)

    rebuild_imgs = reference_imgs * 0.995 + torch.randn_like(reference_imgs) * 0.005
    rebuild_imgs = rebuild_imgs.to(device)

    reference_imgs = torch.clamp(denormalize_image_tensor(reference_imgs), 0.0, 1.0)
    rebuild_imgs = torch.clamp(denormalize_image_tensor(rebuild_imgs), 0.0, 1.0)

    print("reference_imgs.shape: ", reference_imgs.shape)
    print("rebuild_imgs.shape: ", rebuild_imgs.shape)

    fid_val_rgbx, fid_val_nirx = compute_FID_by_imgs(rebuild_imgs[:3], reference_imgs[:6])
    print("fid_val_rgbx: ", fid_val_rgbx)
    print("fid_val_nirx: ", fid_val_nirx)
