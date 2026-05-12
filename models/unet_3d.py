import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd spatial dims
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Standard 3D U-Net with 4 encoder levels.
    in_channels: number of input MRI phases (1 or 2)
    out_channels: number of output classes (2 for binary: background + tumor)
    base_filters: feature maps at first level, doubles each level
    """
    def __init__(self, in_channels=1, out_channels=2, base_filters=32):
        super().__init__()

        f = base_filters
        self.enc1 = DownBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)

        self.bottleneck = ConvBlock(f * 8, f * 16)

        self.dec4 = UpBlock(f * 16, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2)
        self.dec1 = UpBlock(f * 2, f)

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
    model = UNet3D(in_channels=1, out_channels=2, base_filters=32)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Typical training patch size: 128x128x64
    x = torch.randn(1, 1, 128, 128, 64)
    print(f"Input:  {x.shape}")

    with torch.no_grad():
        y = model(x)

    print(f"Output: {y.shape}")
    assert y.shape == (1, 2, 128, 128, 64), "Shape mismatch!"
    print("Forward pass OK.")
