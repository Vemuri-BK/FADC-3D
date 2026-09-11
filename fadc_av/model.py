"""Matched legacy 3D U-Net with one full discrete FADC replacement."""
from models.unet_3d import UNet3D
from .adakern import AdaKernConfig
from .dilation import AdaptiveDilatedConv3D, DilationConfig


class FADCAVUNet3D(UNet3D):
    def __init__(self, in_channels=2, out_channels=2, base_filters=32, placement="enc3"):
        super().__init__(in_channels, out_channels, base_filters)
        if placement not in ("enc3", "all_encoders"):
            raise ValueError("unknown FADC placement")
        stages = ("enc3",) if placement == "enc3" else ("enc1", "enc2", "enc3", "enc4")
        indices = (3,) if placement == "enc3" else (0, 3)
        for stage in stages:
            block = getattr(self, stage).conv.block
            for index in indices:
                original = block[index]
                block[index] = AdaptiveDilatedConv3D(
                    original.in_channels, original.out_channels,
                    DilationConfig(dilations=(1, 2, 3), adaptive_kernel=AdaKernConfig()),
                )


def build_model(config):
    if config["variant"] not in ("fadc_enc3", "fadc_all_encoders", "baseline"):
        raise ValueError("unknown variant")
    kwargs = dict(in_channels=2, out_channels=2, base_filters=config["base_filters"])
    if config["variant"] == "baseline":
        return UNet3D(**kwargs)
    return FADCAVUNet3D(**kwargs, placement="all_encoders" if config["variant"] == "fadc_all_encoders" else "enc3")
