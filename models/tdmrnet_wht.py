"""TDMRNet-WHT：原始容量 + WHT 频域分支。

在原始 TDMRNet（channels=16）基础上仅增加 WHT 并联分支，
不改动通道数和 DCM 结构，用于消融验证 WHT 的独立贡献。
"""

from __future__ import annotations

import torch
from torch import nn

from .common import SamePadConv2d, SEBlock, fwht
from .tdmrnet import ICFEM, DCM


class WHTBranchLight(nn.Module):
    """轻量 WHT 频域分支（与原始 TDMRNet 通道数一致）。"""

    def __init__(self, in_channels: int = 1, channels: int = 16):
        super().__init__()
        self.conv = SamePadConv2d(in_channels, channels, kernel_size=3, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.flatten(1)
        transformed = fwht(flat)
        x = transformed.view(b, c, h, w)
        return self.act(self.norm(self.conv(x)))


class TDMRNetWHT(nn.Module):
    """原始 TDMRNet + WHT 并联分支，双分支 late-fusion。"""

    def __init__(self, num_classes: int = 18, channels: int = 16, num_dcms: int = 2, **_):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(1)
        self.icfem = ICFEM(1, channels)
        self.wht_branch = WHTBranchLight(1, channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dcms = nn.Sequential(*(DCM(channels) for _ in range(num_dcms)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        spatial_feat = self.icfem(x)
        freq_feat = self.wht_branch(x)
        x = self.fuse(torch.cat([spatial_feat, freq_feat], dim=1))
        x = self.dcms(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
