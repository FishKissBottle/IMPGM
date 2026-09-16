"""Pixel-space auxiliary losses for diffusion training.

Provides gradient-enabled VAE decoding, time-step gating, and pixel-space
frequency / Sobel edge losses for FgGen and ImgSyn diffusion trainers.
"""

import torch
import torch.nn.functional as F

from IMPGM_Utils import denormalize_image_tensor, descale_latent


_PIXEL_LOSS_EPS = 1.0e-8


def decode_latent_to_pixel_with_grad(vae_model, latent, latent_scaling_factor=1.0):
    """Decode a scaled latent tensor to pixel space, keeping gradients.

    This is a training-time helper: VAE parameters must stay frozen, but
    gradients are allowed to flow from the decoded output back to `latent`.

    Args:
        vae_model: VAE AutoEncoder instance (parameters frozen, eval mode).
        latent:    Tensor of shape (B, C_latent, H_latent, W_latent).
        latent_scaling_factor: Scaling factor applied to the latent.

    Returns:
        Decoded pixel tensor of shape (B, C_img, H_img, W_img).
    """
    latent_descaled = descale_latent(latent, latent_scaling_factor)
    return vae_model.decode(latent_descaled)


def compute_timestep_gate(t, T):
    """Compute the mandatory linear time-step gate for auxiliary losses.

    Args:
        t: Long tensor of shape (B,) containing time steps in [0, T-1].
        T: Total number of diffusion steps.
    Returns:
        Float tensor of shape (B,) with values in [0, 1].
    """
    if T <= 1:
        return torch.ones_like(t, dtype=torch.float32)
    t_float = t.float() / (T - 1)
    gate = (1.0 - t_float).clamp(0.0, 1.0)
    return gate


def fft_weighted_loss_pixel(pred, target, freq_power=1.0):
    """High-frequency-weighted complex spectrum loss computed in pixel space.

    The squared magnitude of the complex spectrum difference constrains both
    magnitude and phase; the weight is the normalized frequency radius raised
    to `freq_power`, emphasizing high frequencies.

    Args:
        pred:   Tensor of shape (B, C, H, W).
        target: Tensor of shape (B, C, H, W).
        freq_power: Power applied to normalized frequency radius.
    Returns:
        Per-sample loss tensor of shape (B,).
    """
    # XPU FFT does not support fp64 and may not support bf16 kernels. Keep this
    # numerically sensitive auxiliary loss explicitly in fp32 outside AMP.
    with torch.amp.autocast(pred.device.type, enabled=False):
        pred = pred.float()
        target = target.float()
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        B, C, H, W2 = pred_fft.shape
        W = pred.shape[-1]

        freq_y = torch.fft.fftfreq(H, d=1.0, dtype=torch.float32, device=pred.device)[:, None]
        freq_x = torch.fft.rfftfreq(W, d=1.0, dtype=torch.float32, device=pred.device)[None, :]

        freq_radius = torch.sqrt(freq_y ** 2 + freq_x ** 2)
        max_radius = (
            torch.sqrt((freq_y.abs().max()) ** 2 + (freq_x.abs().max()) ** 2)
            + _PIXEL_LOSS_EPS
        )
        freq_radius = freq_radius / max_radius

        weight = freq_radius ** freq_power
        weight = weight.unsqueeze(0).unsqueeze(0)

        diff2 = (pred_fft - target_fft).abs().pow(2)
        loss_fft_per = (weight * diff2).mean(dim=(1, 2, 3))

    return loss_fft_per


def sobel_edge_map(img):
    """Compute Sobel edge magnitude map independently per channel.

    Args:
        img: Tensor of shape (B, C, H, W).
    Returns:
        Edge magnitude tensor of shape (B, C, H, W).
    """
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=dtype, device=device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        dtype=dtype, device=device,
    ).view(1, 1, 3, 3)

    # Repeat kernels for each channel, then use grouped convolution.
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)

    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + _PIXEL_LOSS_EPS)
    return magnitude


def sobel_edge_loss_pixel(pred, target):
    """Pixel-space Sobel edge L2 loss computed per channel then averaged.

    Args:
        pred:   Tensor of shape (B, C, H, W).
        target: Tensor of shape (B, C, H, W).
    Returns:
        Per-sample loss tensor of shape (B,).
    """
    with torch.amp.autocast(pred.device.type, enabled=False):
        pred_edge = sobel_edge_map(pred.float())
        target_edge = sobel_edge_map(target.float())
        diff2 = (pred_edge - target_edge).pow(2)
        loss_edge_per = diff2.mean(dim=(1, 2, 3))
    return loss_edge_per


def prepare_pixel_tensors(
    vae_model,
    pred_x_0_latent,
    x_0_latent,
    latent_scaling_factor,
):
    """Decode latent predictions/targets and return denormalized pixel tensors.

    Args:
        vae_model: VAE AutoEncoder instance (frozen, eval mode).
        pred_x_0_latent: Predicted clean latent (B, C_latent, H_latent, W_latent).
        x_0_latent:      Target clean latent (B, C_latent, H_latent, W_latent).
        latent_scaling_factor: Latent scaling factor.

    Returns:
        Tuple of (pred_pixel, target_pixel) tensors in pixel space.
    """
    pred_pixel = decode_latent_to_pixel_with_grad(
        vae_model, pred_x_0_latent, latent_scaling_factor
    )
    target_pixel = decode_latent_to_pixel_with_grad(
        vae_model, x_0_latent, latent_scaling_factor
    )

    pred_pixel = denormalize_image_tensor(pred_pixel)
    target_pixel = denormalize_image_tensor(target_pixel)

    return pred_pixel, target_pixel
