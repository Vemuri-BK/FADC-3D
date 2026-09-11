"""Per-sample low/high kernel adaptation; no independent dilation kernels."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn


@dataclass(frozen=True)
class AdaKernConfig:
    hidden_channels: int = 16

    def __post_init__(self):
        if type(self.hidden_channels) is not int or self.hidden_channels < 1:
            raise ValueError("hidden_channels must be a positive integer")


class KernelGains(NamedTuple):
    low_input: torch.Tensor
    low_output: torch.Tensor
    high_input: torch.Tensor
    high_output: torch.Tensor


class AdaKern3D(nn.Module):
    """Predict four channel/filter gain vectors from globally pooled features.

    Gains use 2*sigmoid and initialize to one. The shared descriptor trunk is
    Linear+ReLU; no batch-dependent normalization or position attention is used.
    The caller owns the single base kernel. Output is (B,O,I,3,3,3).
    """

    def __init__(self, in_channels, out_channels, config: AdaKernConfig | None = None):
        super().__init__()
        for value in (in_channels, out_channels):
            if type(value) is not int or value < 1:
                raise ValueError("channel counts must be positive integers")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.config = config if config is not None else AdaKernConfig()
        hidden = self.config.hidden_channels
        self.trunk = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU())
        self.heads = nn.ModuleList(nn.Linear(hidden, count) for count in
                                   (in_channels, out_channels, in_channels, out_channels))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def predict_gains(self, x):
        if x.ndim != 5 or x.shape[1] != self.in_channels or any(n < 1 for n in x.shape):
            raise ValueError(f"expected nonempty (B,{self.in_channels},D,H,W) features")
        if not x.is_floating_point():
            raise TypeError("features must be real floating point")
        # Accumulate global means at least in float32, even under outer AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x if x.dtype == torch.float64 else x.float()
            pooled = work.mean(dim=(-3,-2,-1))
        hidden = self.trunk(pooled.to(self.trunk[0].weight.dtype))
        values = []
        for head in self.heads:
            logits = head(hidden)
            logits = logits if logits.dtype == torch.float64 else logits.float()
            values.append(2 * torch.sigmoid(logits))
        return KernelGains(*values)

    def apply_gains(self, weight, gains: KernelGains):
        if tuple(weight.shape) != (self.out_channels, self.in_channels, 3,3,3):
            raise ValueError("base weight must match (out_channels,in_channels,3,3,3)")
        if not weight.is_floating_point():
            raise TypeError("base weight must be real floating point")
        batch = gains.low_input.shape[0]
        for gain, count in zip(gains, (self.in_channels, self.out_channels,
                                      self.in_channels, self.out_channels)):
            if tuple(gain.shape) != (batch, count) or batch < 1:
                raise ValueError("gain shapes must be (B,input_channels) or (B,output_channels)")
        with torch.autocast(device_type=weight.device.type, enabled=False):
            work = weight if weight.dtype == torch.float64 else weight.float()
            low = work.mean(dim=(-3,-2,-1), keepdim=True)
            high = work - low
            li, lo, hi, ho = (g.to(dtype=work.dtype) for g in gains)
            low_scale = lo[:, :, None] * li[:, None, :]
            high_scale = ho[:, :, None] * hi[:, None, :]
            result = (low[None] * low_scale[...,None,None,None]
                      + high[None] * high_scale[...,None,None,None])
        return result.to(weight.dtype)

    def forward(self, x, weight):
        return self.apply_gains(weight, self.predict_gains(x))
