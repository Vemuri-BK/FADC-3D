"""Real-valued 3D frequency decomposition with symmetric cuboid passbands.

Masks use unshifted FFT coordinates and the strict cutoff abs(f) < 1/(2*k).
DC always belongs to the final low-pass residual. Parameters are not created
in forward. Half/bfloat16 inputs are promoted to float32; float64 is preserved.
"""

from dataclasses import dataclass
from math import ceil
from typing import NamedTuple

import torch
from torch import nn


@dataclass(frozen=True)
class FrequencyConfig:
    cutoffs: tuple[int, ...] = (2, 4, 8)

    def __post_init__(self):
        values = tuple(self.cutoffs)
        if not values or any(type(k) is not int or k < 2 for k in values):
            raise ValueError("cutoffs must be nonempty integer denominators >= 2")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("cutoffs must be strictly increasing")
        object.__setattr__(self, "cutoffs", values)


class FrequencyBands(NamedTuple):
    high: tuple[torch.Tensor, ...]
    low: torch.Tensor
    active: tuple[bool, ...]


class FrequencyDecomposition3D(nn.Module):
    """Split (B,C,D,H,W) features into ordered high bands and a low residual.

    Output band slots always match config.cutoffs. Duplicate low-pass masks
    produce an inactive zero band. `active` describes frequency support, not
    whether a particular input happens to contain energy in that band.
    """

    def __init__(self, config: FrequencyConfig | None = None):
        super().__init__()
        self.config = config if config is not None else FrequencyConfig()

    def masks(self, shape, device, dtype=torch.float32):
        if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("shape must contain three positive integers")
        fz = torch.fft.fftfreq(shape[0], device=device, dtype=dtype)[:, None, None]
        fy = torch.fft.fftfreq(shape[1], device=device, dtype=dtype)[None, :, None]
        fx = torch.fft.fftfreq(shape[2], device=device, dtype=dtype)[None, None, :]
        return tuple(
            ((fz.abs() < 1 / (2 * k))
             & (fy.abs() < 1 / (2 * k))
             & (fx.abs() < 1 / (2 * k)))[None, None]
            for k in self.config.cutoffs
        )

    def forward(self, x: torch.Tensor) -> FrequencyBands:
        if x.ndim != 5 or any(n < 1 for n in x.shape):
            raise ValueError("expected nonempty (B,C,D,H,W) input")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("input must be a real floating-point tensor")
        # Explicit precision policy keeps FFTs safe inside an outer AMP context.
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x if x.dtype == torch.float64 else x.float()
            shape = tuple(work.shape[-3:])
            spectrum = torch.fft.fftn(work, dim=(-3, -2, -1), norm="ortho")
            masks = self.masks(shape, work.device, work.dtype)
            previous = work
            # Largest absolute frequency index present in the full spectrum.
            previous_support = tuple(n // 2 for n in shape)
            bands, active = [], []
            for k, mask in zip(self.config.cutoffs, masks):
                support = tuple(ceil(n / (2 * k)) - 1 for n in shape)
                changed = support != previous_support
                if changed:
                    low = torch.fft.ifftn(
                        spectrum * mask, dim=(-3, -2, -1), norm="ortho"
                    ).real
                    bands.append(previous - low)
                else:
                    low = previous
                    bands.append(torch.zeros_like(previous))
                active.append(changed)
                previous, previous_support = low, support
            return FrequencyBands(tuple(bands), previous, tuple(active))
