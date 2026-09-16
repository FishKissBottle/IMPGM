import torch
from torch import nn


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Apply two normalized convolutions without batch-size-dependent state."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class UpBlock(nn.Module):
    """Upsample decoder features and fuse an explicitly sized skip tensor."""

    def __init__(self, decoder_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            decoder_channels, out_channels, kernel_size=2, stride=2
        )
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs, skip):
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"EvaluationUNet skip mismatch: {inputs.shape[-2:]} vs {skip.shape[-2:]}."
            )
        return self.fuse(torch.cat((inputs, skip), dim=1))


class EvaluationUNet(nn.Module):
    """Segment a requested class from a complete multispectral image."""

    def __init__(
        self,
        image_channels,
        num_classes,
        base_channels=64,
        condition_channels=16,
    ):
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.label_embedding = nn.Embedding(self.num_classes, int(condition_channels))

        base = int(base_channels)
        input_channels = int(image_channels) + int(condition_channels)
        self.enc1 = ConvBlock(input_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = UpBlock(base * 8, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2, base * 2)
        self.up1 = UpBlock(base * 2, base, base)
        self.output = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, images, label_ids):
        if images.ndim != 4:
            raise ValueError(f"Expected images [B,C,H,W], got {tuple(images.shape)}.")
        if images.shape[-2] % 8 != 0 or images.shape[-1] % 8 != 0:
            raise ValueError("EvaluationUNet input height and width must be divisible by 8.")
        label_ids = label_ids.to(device=images.device, dtype=torch.long).reshape(-1)
        if label_ids.shape[0] != images.shape[0]:
            raise ValueError("One label_id is required for every input image.")
        if torch.any(label_ids < 0) or torch.any(label_ids >= self.num_classes):
            raise ValueError("label_ids contain values outside the configured prompt map.")

        condition = self.label_embedding(label_ids).to(dtype=images.dtype)
        condition = condition[:, :, None, None].expand(
            -1, -1, images.shape[-2], images.shape[-1]
        )
        inputs = torch.cat((images, condition), dim=1)
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        center = self.bottleneck(self.pool(enc3))
        decoded = self.up3(center, enc3)
        decoded = self.up2(decoded, enc2)
        decoded = self.up1(decoded, enc1)
        return self.output(decoded)
