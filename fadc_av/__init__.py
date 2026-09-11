"""Verified building blocks for the discrete FADC-AV project."""

from .frequency import FrequencyConfig, FrequencyDecomposition3D, FrequencyBands
from .selection import SelectionConfig, SelectionResult, FrequencySelection3D
from .adakern import AdaKernConfig, KernelGains, AdaKern3D
from .dilation import (
    DilationConfig, DilationResult, SharedKernelConv3D,
    VoxelwiseDilationSelector, AdaptiveDilatedConv3D,
)

__all__ = [
    "FrequencyConfig", "FrequencyDecomposition3D", "FrequencyBands",
    "SelectionConfig", "SelectionResult", "FrequencySelection3D",
    "DilationConfig", "DilationResult", "SharedKernelConv3D",
    "VoxelwiseDilationSelector", "AdaptiveDilatedConv3D",
    "AdaKernConfig", "KernelGains", "AdaKern3D",
]
