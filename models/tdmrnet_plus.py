"""TDMRNet++：容量扩展 + SE 通道注意力 + WHT 频域分支。

在原始 TDMRNet 基础上引入三项改进：
  A1 容量扩展——ICFEM 通道数 16→64，DCM 渐进扩展 64→128→256，分类头换为两层 MLP；
  A2 注意力机制——ICFEM 后和每个 DCM 残差块内嵌入 SE 模块；
  A3 WHT 频域分支——并联 Walsh-Hadamard 变换分支，利用 Polar 码的
      Kronecker 代数结构在频域暴露码长/码率特征，与空域 ICFEM 特征 late-fusion。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import SamePadConv2d, SEBlock, fwht


class ICFEMPlus(nn.Module):
    """增强版码间特征提取模块：通道数扩展 + SE 通道注意力。"""

    def __init__(self, in_channels: int = 1, channels: int = 64):
        super().__init__()
        self.local_conv = SamePadConv2d(in_channels, channels, kernel_size=(3, 1), bias=False)
        self.reducers = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False) for _ in range(3)
        )
        self.multiscale_convs = nn.ModuleList(
            SamePadConv2d(channels, channels, kernel_size=(h, 2), bias=False) for h in (8, 16, 32)
        )
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=8)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.local_conv(x))
        branches = [
            self.act(conv(self.act(reducer(x))))
            for reducer, conv in zip(self.reducers, self.multiscale_convs)
        ]
        x = self.act(self.norm(self.fuse(torch.cat(branches, dim=1))))
        return self.se(x)


class WHTBranch(nn.Module):
    """Walsh-Hadamard 变换频域特征提取分支。

    将 LLR 序列投影到 Hadamard 域，使 Polar 码 Kronecker 结构对应的
    频域模式更容易被卷积捕获。
    """

    def __init__(self, in_channels: int = 1, channels: int = 64, matrix_rows: int = 256):
        super().__init__()
        self.matrix_rows = matrix_rows
        self.conv = SamePadConv2d(in_channels, channels, kernel_size=3, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=8)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.flatten(1)                       # [B, H*W]
        transformed = fwht(flat)                  # Hadamard 域表示
        x = transformed.view(b, c, h, w)
        x = self.act(self.norm(self.conv(x)))
        return self.se(x)


class SEResidualBlock(nn.Module):
    """内嵌 SE 注意力的预激活残差块，支持步长降维和通道扩展。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = SamePadConv2d(in_channels, out_channels, kernel_size=(3, 2), stride=stride, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = SamePadConv2d(out_channels, out_channels, kernel_size=(3, 2), bias=False)
        self.se = SEBlock(out_channels, reduction=8)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = SamePadConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(F.relu(self.norm1(x), inplace=True))
        x = self.conv2(F.relu(self.norm2(x), inplace=True))
        x = self.se(x)
        return x + residual


class DCMPlus(nn.Module):
    """增强版数据耦合模块：降维残差块（通道扩展+步长2）+ 标准残差块。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.reduction_block = SEResidualBlock(in_channels, out_channels, stride=2)
        self.standard_block = SEResidualBlock(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.standard_block(self.reduction_block(x))


class TDMRNetPlus(nn.Module):
    """TDMRNet++：ICFEM+ + WHT 并联分支 + 渐进 DCM + 双重池化 MLP 分类头。"""

    def __init__(self, num_classes: int = 18, channels: int = 16, num_dcms: int = 2, **_):
        super().__init__()
        base = max(channels * 2, 32)              # 16→32
        dcm_channels = [base * 2, base * 4]       # 32→64, 64→128

        self.input_norm = nn.BatchNorm2d(1)
        self.icfem = ICFEMPlus(1, base)            # 64 通道
        self.wht_branch = WHTBranch(1, base)       # 64 通道
        self.fuse = nn.Sequential(
            nn.Conv2d(base * 2, base, kernel_size=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.dcms = nn.Sequential(
            DCMPlus(base, dcm_channels[0]),
            DCMPlus(dcm_channels[0], dcm_channels[1]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        feat_dim = dcm_channels[-1] * 2            # 256*2=512
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        spatial_feat = self.icfem(x)
        freq_feat = self.wht_branch(x)
        x = self.fuse(torch.cat([spatial_feat, freq_feat], dim=1))
        x = self.dcms(x)
        pooled = torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1)
        return self.classifier(pooled)
