"""新架构模型共享的公共组件。

包含 SAME 填充卷积、通道注意力 (SE)、空间+通道注意力 (CBAM)、
快速 Walsh-Hadamard 变换 (FWHT) 以及通用工具函数。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# ── SAME 填充卷积 ──────────────────────────────────────────
class SamePadConv2d(nn.Conv2d):
    """为任意卷积核提供 TensorFlow 风格的 SAME 非对称填充。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]
        stride_h, stride_w = self.stride
        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.dilation
        output_h = (input_h + stride_h - 1) // stride_h
        output_w = (input_w + stride_w - 1) // stride_w
        pad_h = max((output_h - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - input_w, 0)
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class SamePadConv1d(nn.Conv1d):
    """一维版本的 SAME 填充卷积。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_w = x.shape[-1]
        stride_w = self.stride[0]
        kernel_w = self.kernel_size[0]
        dilation_w = self.dilation[0]
        output_w = (input_w + stride_w - 1) // stride_w
        pad_w = max((output_w - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - input_w, 0)
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2))
        return F.conv1d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


# ── 通道注意力 (Squeeze-and-Excitation) ─────────────────────
class SEBlock(nn.Module):
    """通道注意力模块：通过全局池化 → FC → 缩放因子重标定通道特征。"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).flatten(1)          # [B, C]
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


# ── CBAM: 通道 + 空间双重注意力 ────────────────────────────
class ChannelAttention(nn.Module):
    """CBAM 通道注意力：同时使用 GAP 和 GMP，经共享 MLP 后求和。"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力：沿通道维做均值+最大池化，7×7 卷积生成空间权重。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """串联通道注意力和空间注意力。"""

    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ── 快速 Walsh-Hadamard 变换 ────────────────────────────────
def fwht(x: torch.Tensor) -> torch.Tensor:
    """沿最后一维执行归一化快速 Walsh-Hadamard 变换，输入末维须为 2 的幂。"""
    *_, length = x.shape
    if length & (length - 1) != 0:
        raise ValueError(f"WHT 要求长度为 2 的幂，实际为 {length}。")
    original_shape = x.shape
    x = x.reshape(-1, length).contiguous()
    batch = x.shape[0]
    h = 1
    while h < length:
        # 重排为 [batch, L/(2h), 2, h] 做蝴蝶运算
        x = x.view(batch, length // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=-1).reshape(batch, length)
        h *= 2
    x = x.reshape(original_shape)
    return x / (length ** 0.5)


# ── 通用卷积块 ──────────────────────────────────────────────
class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm → GELU/ReLU 标准块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        padding = kernel_size // 2 if isinstance(kernel_size, int) else 0
        self.conv = SamePadConv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        if activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))
