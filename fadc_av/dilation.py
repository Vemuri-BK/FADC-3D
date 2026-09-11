"""Shared-kernel discrete dilation with voxelwise branch selection.

This operator returns mixed features only. Normalization, activation, and
residual connections belong to the enclosing network block. Uniform initial
attention produces a branch average, not an ordinary dilation-one convolution.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .selection import FrequencySelection3D, SelectionConfig
from .adakern import AdaKernConfig, AdaKern3D


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _temperature(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class DilationConfig:
    dilations: tuple[int, ...] = (1, 2, 3)
    attention_channels: int = 16
    initial_temperature: float = 1.0
    selection: SelectionConfig | None = field(default_factory=SelectionConfig)
    bias: bool = False
    adaptive_kernel: AdaKernConfig | None = None

    def __post_init__(self):
        rates = tuple(self.dilations)
        if not rates:
            raise ValueError("dilations must be nonempty")
        for rate in rates:
            _positive_integer(rate, "dilation")
        if any(a >= b for a, b in zip(rates, rates[1:])):
            raise ValueError("dilations must be strictly increasing")
        _positive_integer(self.attention_channels, "attention_channels")
        _temperature(self.initial_temperature)
        if self.selection is not None and not isinstance(self.selection, SelectionConfig):
            raise TypeError("selection must be SelectionConfig or None")
        if type(self.bias) is not bool:
            raise TypeError("bias must be bool")
        if self.adaptive_kernel is not None and not isinstance(self.adaptive_kernel, AdaKernConfig):
            raise TypeError("adaptive_kernel must be AdaKernConfig or None")
        object.__setattr__(self, "dilations", rates)


class SharedKernelConv3D(nn.Module):
    """One trainable 3x3x3 kernel; any positive integer dilation preserves size."""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        _positive_integer(in_channels, "in_channels")
        _positive_integer(out_channels, "out_channels")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, dilation, sample_weight=None):
        _positive_integer(dilation, "dilation")
        if x.ndim != 5 or x.shape[1] != self.in_channels or any(n < 1 for n in x.shape):
            raise ValueError(f"expected nonempty (B,{self.in_channels},D,H,W) input")
        if not x.is_floating_point():
            raise TypeError("input must be real floating point")
        if sample_weight is None:
            return F.conv3d(x, self.weight, self.bias, padding=dilation, dilation=dilation)
        batch, channels, d, h, w = x.shape
        expected = (batch, self.out_channels, channels, 3,3,3)
        if tuple(sample_weight.shape) != expected:
            raise ValueError(f"sample_weight must have shape {expected}")
        # Pack independent samples into convolution groups. Every sample still
        # uses a groups=1 base convolution; no channels cross between patients.
        packed = x.reshape(1, batch*channels, d,h,w)
        weights = sample_weight.reshape(batch*self.out_channels, channels, 3,3,3)
        bias = self.bias.repeat(batch) if self.bias is not None else None
        result = F.conv3d(packed, weights, bias, padding=dilation,
                          dilation=dilation, groups=batch)
        return result.reshape(batch,self.out_channels,d,h,w)


class VoxelwiseDilationSelector(nn.Module):
    """Spatial softmax probabilities with checkpointed runtime temperature."""

    def __init__(self, in_channels, num_branches, hidden_channels=16, temperature=1.0):
        super().__init__()
        for value, name in [(in_channels, "in_channels"), (num_branches, "num_branches"),
                            (hidden_channels, "hidden_channels")]:
            _positive_integer(value, name)
        self.trunk = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 3, padding=1), nn.ReLU()
        )
        self.head = nn.Conv3d(hidden_channels, num_branches, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("temperature", torch.tensor(_temperature(temperature)))

    def set_temperature(self, temperature):
        self.temperature.fill_(_temperature(temperature))

    def forward(self, x):
        logits = self.head(self.trunk(x))
        logits = logits if logits.dtype == torch.float64 else logits.float()
        return torch.softmax(logits / self.temperature.to(dtype=logits.dtype), dim=1)


class DilationResult(NamedTuple):
    output: torch.Tensor
    probabilities: torch.Tensor
    expected_dilation: torch.Tensor


class AdaptiveDilatedConv3D(nn.Module):
    """Optional frequency selection followed by shared-kernel dilation mixing.

    Optional AdaKern constructs one per-sample kernel reused by all branches.
    No branch stack or persistent activation cache is allocated. Autograd still
    retains intermediates needed by backward. Mixing uses at least float32.
    """

    def __init__(self, in_channels, out_channels, config: DilationConfig | None = None):
        super().__init__()
        self.config = config if config is not None else DilationConfig()
        self.convolution = SharedKernelConv3D(in_channels, out_channels, self.config.bias)
        self.frequency_selection = (
            FrequencySelection3D(in_channels, self.config.selection)
            if self.config.selection is not None else nn.Identity()
        )
        self.selector = VoxelwiseDilationSelector(
            in_channels, len(self.config.dilations), self.config.attention_channels,
            self.config.initial_temperature,
        )
        self.adakern = (AdaKern3D(in_channels, out_channels, self.config.adaptive_kernel)
                        if self.config.adaptive_kernel is not None else None)

    def get_extra_state(self):
        config = asdict(self.config)
        # Runtime temperature lives in the buffer, independently of initialization.
        config.pop("initial_temperature")
        return {"operator_version": 2, "in_channels": self.convolution.in_channels,
                "out_channels": self.convolution.out_channels, "config": config}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("FADC-AV architecture/configuration mismatch in state_dict")

    def set_temperature(self, temperature):
        self.selector.set_temperature(temperature)

    def forward_with_attention(self, x):
        if x.ndim != 5 or x.shape[1] != self.convolution.in_channels or any(n < 1 for n in x.shape):
            raise ValueError("input must match the configured nonempty (B,C,D,H,W) layout")
        if not x.is_floating_point():
            raise TypeError("input must be real floating point")
        selected = self.frequency_selection(x)
        # Standard parameter precision is retained under autocast; AMP chooses
        # convolution precision while FFT decomposition and mixing stay float32.
        features = selected.to(dtype=self.convolution.weight.dtype)
        probabilities = self.selector(features)
        sample_weight = (self.adakern(features, self.convolution.weight)
                         if self.adakern is not None else None)
        output = None
        expected = torch.zeros_like(probabilities[:, 0])
        for index, rate in enumerate(self.config.dilations):
            branch = self.convolution(features, rate, sample_weight=sample_weight)
            weighted = branch.to(probabilities.dtype) * probabilities[:, index:index+1]
            output = weighted if output is None else output + weighted
            expected = expected + rate * probabilities[:, index]
        return DilationResult(output, probabilities, expected)

    def forward(self, x):
        return self.forward_with_attention(x).output
