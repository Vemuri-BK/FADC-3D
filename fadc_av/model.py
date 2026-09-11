"""Matched legacy 3D U-Net with one full discrete FADC replacement."""
from models.unet_3d import UNet3D
from .adakern import AdaKernConfig
from .dilation import AdaptiveDilatedConv3D, DilationConfig


class FADCAVUNet3D(UNet3D):
    def __init__(self, in_channels=2, out_channels=2, base_filters=32):
        super().__init__(in_channels, out_channels, base_filters)
        # Keep both surrounding BN/ReLU layers and every other backbone block.
        self.enc3.conv.block[3] = AdaptiveDilatedConv3D(
            base_filters * 4, base_filters * 4,
            DilationConfig(dilations=(1, 2, 3), adaptive_kernel=AdaKernConfig()),
        )


def build_model(config):
    cls = FADCAVUNet3D if config["variant"] == "fadc_enc3" else UNet3D
    if config["variant"] not in ("fadc_enc3", "baseline"):
        raise ValueError("unknown variant")
    return cls(in_channels=2, out_channels=2, base_filters=config["base_filters"])
