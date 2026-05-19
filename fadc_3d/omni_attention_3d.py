import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniAttention3D(nn.Module):
    """
    3D extension of OmniAttention (AdaKern) from FADC (CVPR 2024).
    Generates four attention signals from a global channel descriptor:
      - channel attention  : reweights input feature channels
      - filter attention   : reweights output feature channels
      - spatial attention  : reweights spatial positions within the kernel (k^3)
      - kernel attention   : softmax weights over multiple kernel variants / dilation branches
    All 2D ops (AdaptiveAvgPool2d, Conv2d, BatchNorm2d) → 3D equivalents.
    """

    def __init__(self, in_planes, out_planes, kernel_size,
                 groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm3d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:
            self.func_spatial = self.skip
        else:
            # k^3 positions for a 3D kernel of side kernel_size
            self.spatial_fc = nn.Conv3d(attention_channel, kernel_size ** 3, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        # (b, in_planes, 1, 1, 1)
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        # (b, out_planes, 1, 1, 1)
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        # (b, 1, 1, 1, 1, k, k, k) — broadcasts with 3D kernel weight (b, c_out, c_in, k, k, k)
        k = self.kernel_size
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, 1, k, k, k)
        return torch.sigmoid(spatial_attention / self.temperature)

    def get_kernel_attention(self, x):
        # (b, kernel_num, 1, 1, 1) — softmax weights, one per dilation branch
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1)
        return F.softmax(kernel_attention / self.temperature, dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
