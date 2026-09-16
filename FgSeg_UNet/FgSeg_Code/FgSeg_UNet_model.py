import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Apply two convolutional layers for segmentation feature extraction."""
    def __init__(self, in_channels, out_channels, use_bn=False):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batch_norm2(x)
        x = self.relu(x)

        return x


class DownBlock(nn.Module):
    """Downsample segmentation features and apply a convolutional block."""
    def __init__(self, in_channels, out_channels, use_maxpool=False, use_bn=False):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, use_bn=use_bn)

        if use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        x_pool = self.pool(x)
        return x, x_pool


class UpBlock(nn.Module):
    """Upsample and fuse segmentation features with a skip connection."""
    def __init__(self, in_channels, out_channels, use_bn=False):
        super(UpBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, use_bn=use_bn)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x


class FgSeg_UNet(nn.Module):
    """Predict foreground masks with a UNet segmentation model."""
    def __init__(self, in_channels, out_channels):
        super(FgSeg_UNet, self).__init__()
        self.down1 = DownBlock(in_channels, 64, use_maxpool=False, use_bn=True)             # 256 -> 128
        self.down2 = DownBlock(64, 128, use_maxpool=False, use_bn=True)                     # 128 -> 64
        self.down3 = DownBlock(128, 256, use_maxpool=False, use_bn=True)                    # 64 -> 32

        self.mid = ConvBlock(256, 512, use_bn=True)                                         # 32 -> 32

        self.up3 = UpBlock(512, 256, use_bn=False)                                          # 32 -> 64
        self.up2 = UpBlock(256, 128, use_bn=False)                                          # 64 -> 128
        self.up1 = UpBlock(128, 64, use_bn=False)                                           # 128 -> 256

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.down1(x)
        d2, p2 = self.down2(p1)
        d3, p3 = self.down3(p2)

        mid = self.mid(p3)

        u3 = self.up3(mid, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)

        out = self.final_conv(u1)

        return out


if __name__ == "__main__":

    in_channels = 4
    out_channels = 1

    model = FgSeg_UNet(in_channels, out_channels)

    Fg_imgs_e = torch.randn(6, 4, 256, 256)
    output_tensor = model(Fg_imgs_e)
    print(output_tensor.shape)

