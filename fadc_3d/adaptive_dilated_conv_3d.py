import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d.omni_attention_3d import OmniAttention3D
from fadc_3d.freq_select_3d import FrequencySelection3D


class AdaptiveDilatedConv3D(nn.Module):
    """
    3D Frequency-Adaptive Dilated Convolution.

    The 2D original (FADC, CVPR 2024) uses ModulatedDeformConv2d (mmcv) to
    produce a single spatially-adaptive dilation per location.  That op has no
    3D equivalent, so we instead run MULTIPLE explicit 3D dilated convolutions
    in parallel and let OmniAttention3D learn softmax weights over them.

    Pipeline per forward pass:
      1. FrequencySelection3D  — decomposes x into freq bands, re-weights them.
                                 This filters the signal before the conv sees it,
                                 letting each dilation branch focus on the right band.
      2. Channel attention      — from OmniAttention3D; rescales input channels.
      3. N dilated Conv3d branches (dilation_list=[1,2,4] by default).
         dilation=1 → high-freq details (tumour boundary, texture).
         dilation=2 → mid-range context.
         dilation=4 → low-freq structure (overall tumour shape).
      4. Kernel attention       — softmax weights over the N branch outputs.
         The network learns which dilation is most useful per location / feature map.
      5. Filter attention       — rescales output channels.

    Drop-in replacement for nn.Conv3d when kernel_size=3 and stride=1.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True,
                 dilation_list=None,
                 reduction=0.0625,
                 min_channel=16,
                 fs_cfg=None):
        super().__init__()
        if dilation_list is None:
            dilation_list = [1, 2, 4]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_list = dilation_list

        # One dilated conv branch per dilation rate
        self.conv_branches = nn.ModuleList()
        for dil in dilation_list:
            pad = dil * (kernel_size - 1) // 2
            self.conv_branches.append(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=pad,
                          dilation=dil,
                          groups=groups,
                          bias=bias))

        # OmniAttention produces:
        #   c_att  (b, in_channels,  1,1,1)  channel gate on input
        #   f_att  (b, out_channels, 1,1,1)  filter gate on output
        #   k_att  (b, num_branches, 1,1,1)  softmax branch weights
        self.omni_att = OmniAttention3D(
            in_planes=in_channels,
            out_planes=out_channels,
            kernel_size=1,          # spatial att unused — skip it
            kernel_num=len(dilation_list),
            reduction=reduction,
            min_channel=min_channel)

        # Optional frequency pre-selection (default config mirrors 2D FADC)
        if fs_cfg is None:
            fs_cfg = dict(
                k_list=[2, 4, 8],
                lowfreq_att=False,
                lp_type='freq',
                act='sigmoid',
                spatial_group=1)
        self.fs = FrequencySelection3D(in_channels, **fs_cfg)

        self._initialize_weights()

    def _initialize_weights(self):
        for conv in self.conv_branches:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        # Step 1 — frequency pre-selection
        x_fs = self.fs(x)

        # Step 2 — attention signals from the freq-selected feature
        c_att, f_att, _, k_att = self.omni_att(x_fs)
        # c_att: (b, in_channels,  1,1,1)
        # f_att: (b, out_channels, 1,1,1)
        # k_att: (b, num_branches, 1,1,1)

        # Step 3 — channel-gate the input, then run all dilation branches
        x_in = x_fs * c_att
        branch_outs = torch.stack([conv(x_in) for conv in self.conv_branches], dim=1)
        # branch_outs: (b, num_branches, out_channels, d, h, w)

        # Step 4 — weighted sum over branches (k_att broadcasts over c,d,h,w)
        out = (branch_outs * k_att.unsqueeze(2)).sum(dim=1)
        # out: (b, out_channels, d, h, w)

        # Step 5 — filter-gate the output
        out = out * f_att
        return out


if __name__ == '__main__':
    x = torch.rand(1, 32, 16, 32, 32)
    m = AdaptiveDilatedConv3D(in_channels=32, out_channels=64)
    y = m(x)
    print('Input :', x.shape)
    print('Output:', y.shape)  # (1, 64, 16, 32, 32)
