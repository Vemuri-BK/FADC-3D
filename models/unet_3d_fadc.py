import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d.adaptive_dilated_conv_3d import AdaptiveDilatedConv3D


# k_list=[2] is safe across all spatial sizes, including the bottleneck
# (8x8x4 after 4 downsamples on a 128x128x64 patch). k_list=[2,4,8] from
# the 2D paper assumes 512x512+ images and breaks at small depths.
_FS_CFG = dict(
    k_list=[2],
    lowfreq_att=False,
    lp_type='freq',
    act='sigmoid',
    spatial_group=1,
)


class FADCConvBlock(nn.Module):
    """
    Two consecutive AdaptiveDilatedConv3D → BN → ReLU layers.
    Drop-in replacement for the standard ConvBlock in unet_3d.py.
    bias=False because BatchNorm absorbs any additive constant.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = AdaptiveDilatedConv3D(in_ch, out_ch, kernel_size=3, bias=False, fs_cfg=_FS_CFG)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = AdaptiveDilatedConv3D(out_ch, out_ch, kernel_size=3, bias=False, fs_cfg=_FS_CFG)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = FADCConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = FADCConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3DFADC(nn.Module):
    """
    3D U-Net with Frequency-Adaptive Dilated Convolution at every
    encoder, bottleneck, and decoder block (FADC-Full variant).

    Architecture is identical to UNet3D (unet_3d.py) — same depth,
    same channel widths, same skip connections — with ConvBlock
    replaced by FADCConvBlock throughout.

    For the ablation study:
      - This file  →  FADC-Full  (primary model)
      - Bottleneck-only variant  →  swap enc/dec blocks back to ConvBlock
    """
    def __init__(self, in_channels=1, out_channels=2, base_filters=32):
        super().__init__()

        f = base_filters
        # Encoder
        self.enc1 = DownBlock(in_channels, f)        # 128x128x64 → skip(f),   pool → 64x64x32
        self.enc2 = DownBlock(f,      f * 2)          # 64x64x32   → skip(2f),  pool → 32x32x16
        self.enc3 = DownBlock(f * 2,  f * 4)          # 32x32x16   → skip(4f),  pool → 16x16x8
        self.enc4 = DownBlock(f * 4,  f * 8)          # 16x16x8    → skip(8f),  pool → 8x8x4

        # Bottleneck
        self.bottleneck = FADCConvBlock(f * 8, f * 16)  # 8x8x4 → 16f channels

        # Decoder
        self.dec4 = UpBlock(f * 16, f * 8)
        self.dec3 = UpBlock(f * 8,  f * 4)
        self.dec2 = UpBlock(f * 4,  f * 2)
        self.dec1 = UpBlock(f * 2,  f)

        # Segmentation head — plain 1x1 conv, no FADC needed
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)


if __name__ == '__main__':
    model = UNet3DFADC(in_channels=1, out_channels=2, base_filters=32)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    x = torch.randn(1, 1, 128, 128, 64)
    print(f"Input:  {x.shape}")

    with torch.no_grad():
        y = model(x)

    print(f"Output: {y.shape}")
    assert y.shape == (1, 2, 128, 128, 64), "Shape mismatch!"
    print("Forward pass OK.")
