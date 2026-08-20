"""TDMRNet-CA：容量扩展 + SE 通道注意力（不含 WHT 频域分支）。

与 TDMRNet++ 的区别：不引入 WHT 并联分支，仅做空域特征增强。
用于消融对比，验证 WHT 频域分支的独立贡献。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import SamePadConv2d, SEBlock
from .tdmrnet_plus import ICFEMPlus, SEResidualBlock, DCMPlus


class TDMRNetCA(nn.Module):
    """ICFEM+ + 渐进 DCM + 双重池化 MLP 分类头（无 WHT 分支）。"""

    def __init__(self, num_classes: int = 18, channels: int = 16, num_dcms: int = 2, **_):
        super().__init__()
        base = max(channels * 2, 32)
        dcm_channels = [base * 2, base * 4]

        self.input_norm = nn.BatchNorm2d(1)
        self.icfem = ICFEMPlus(1, base)
        self.dcms = nn.Sequential(
            DCMPlus(base, dcm_channels[0]),
            DCMPlus(dcm_channels[0], dcm_channels[1]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        feat_dim = dcm_channels[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.icfem(x)
        x = self.dcms(x)
        pooled = torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1)
        return self.classifier(pooled)
