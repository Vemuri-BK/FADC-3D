"""Spatial frequency gates on top of the verified frequency decomposition."""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import nn

from .frequency import FrequencyConfig, FrequencyDecomposition3D


@dataclass(frozen=True)
class SelectionConfig:
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    spatial_groups: int = 1
    kernel_size: int = 3
    low_frequency_attention: bool = False

    def __post_init__(self):
        if not isinstance(self.frequency, FrequencyConfig):
            raise TypeError("frequency must be a FrequencyConfig")
        if type(self.spatial_groups) is not int or self.spatial_groups < 1:
            raise ValueError("spatial_groups must be a positive integer")
        if type(self.kernel_size) is not int or self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if type(self.low_frequency_attention) is not bool:
            raise TypeError("low_frequency_attention must be bool")


class SelectionResult(NamedTuple):
    output: torch.Tensor
    high_gains: tuple[torch.Tensor | None, ...]
    low_gain: torch.Tensor | None
    active: tuple[bool, ...]


class FrequencySelection3D(nn.Module):
    """Learn spatial band gains 2*sigmoid(Conv3D(x)).

    Each group shares a gain across its contiguous subset of input channels.
    Zero-initialized gate heads give unit gains. The low residual passes through
    unless explicitly enabled. Inactive bands skip their gate computation.
    Output follows the decomposition precision policy (at least float32).
    Ordinary forward returns a tensor; forward_with_gates exposes diagnostics
    without storing activation tensors on the module between calls.
    """

    def __init__(self, in_channels: int, config: SelectionConfig | None = None):
        super().__init__()
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        self.config = config if config is not None else SelectionConfig()
        if in_channels % self.config.spatial_groups:
            raise ValueError("spatial_groups must divide in_channels")
        self.in_channels = in_channels
        self.decomposition = FrequencyDecomposition3D(self.config.frequency)
        self.high_gates = nn.ModuleList([
            self._new_gate() for _ in self.config.frequency.cutoffs
        ])
        self.low_gate = self._new_gate() if self.config.low_frequency_attention else None

    def _new_gate(self):
        gate = nn.Conv3d(
            self.in_channels, self.config.spatial_groups,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size // 2,
            groups=self.config.spatial_groups,
        )
        nn.init.zeros_(gate.weight)
        nn.init.zeros_(gate.bias)
        return gate

    @staticmethod
    def _gain(gate, features):
        logits = gate(features)
        # Keep sigmoid and band multiplication out of half precision.
        logits = logits if logits.dtype == torch.float64 else logits.float()
        return 2 * torch.sigmoid(logits)

    def _weight_band(self, band, gain):
        b, c, d, h, w = band.shape
        groups = self.config.spatial_groups
        return (band.reshape(b, groups, c // groups, d, h, w)
                * gain.unsqueeze(2)).reshape(b, c, d, h, w)

    def forward_with_gates(self, x: torch.Tensor) -> SelectionResult:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B,{self.in_channels},D,H,W) input")
        parts = self.decomposition(x)
        features = x.to(dtype=self.high_gates[0].weight.dtype)
        low_gain = None
        output = parts.low
        if self.low_gate is not None:
            low_gain = self._gain(self.low_gate, features)
            output = self._weight_band(parts.low, low_gain)
        gains = []
        for active, band, gate in zip(parts.active, parts.high, self.high_gates):
            if not active:
                gains.append(None)
                continue
            gain = self._gain(gate, features)
            output = output + self._weight_band(band, gain)
            gains.append(gain)
        return SelectionResult(output, tuple(gains), low_gain, parts.active)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_gates(x).output
