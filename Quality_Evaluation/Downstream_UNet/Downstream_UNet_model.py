import torch
from torch import nn


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Apply two batch-size-independent convolution blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class UpBlock(nn.Module):
    """Upsample decoder features and fuse an explicit encoder skip tensor."""

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
                "DownstreamUNet skip mismatch: "
                f"{inputs.shape[-2:]} vs {skip.shape[-2:]}."
            )
        return self.fuse(torch.cat((inputs, skip), dim=1))


class DownstreamUNet(nn.Module):
    """Segment the requested class for real-only versus synthetic-only tests.

    The five U-Net stages follow the channel capacity reported for the auxiliary
    segmentation network in Konz et al. (2024). IMPGM uses class-conditioned
    binary masks, so a label embedding identifies the requested foreground class.
    """

    def __init__(
        self,
        image_channels,
        num_classes,
        channels=(16, 32, 64, 128, 256),
        condition_channels=16,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        if len(channels) != 5 or any(value <= 0 for value in channels):
            raise ValueError("channels must contain five positive stage widths.")
        if int(image_channels) <= 0:
            raise ValueError("image_channels must be positive.")
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be positive.")
        if int(condition_channels) <= 0:
            raise ValueError("condition_channels must be positive.")

        self.image_channels = int(image_channels)
        self.num_classes = int(num_classes)
        self.label_embedding = nn.Embedding(self.num_classes, int(condition_channels))

        c1, c2, c3, c4, c5 = channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(self.image_channels + int(condition_channels), c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)
        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.output = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, images, label_ids):
        if images.ndim != 4:
            raise ValueError(f"Expected images [B,C,H,W], got {tuple(images.shape)}.")
        if images.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} image channels, got {images.shape[1]}."
            )
        if images.shape[-2] % 16 != 0 or images.shape[-1] % 16 != 0:
            raise ValueError(
                "DownstreamUNet input height and width must be divisible by 16."
            )

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
        enc4 = self.enc4(self.pool(enc3))
        center = self.bottleneck(self.pool(enc4))
        decoded = self.up4(center, enc4)
        decoded = self.up3(decoded, enc3)
        decoded = self.up2(decoded, enc2)
        decoded = self.up1(decoded, enc1)
        return self.output(decoded)
