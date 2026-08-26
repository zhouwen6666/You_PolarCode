"""论文 TDMRNet 的 ICFEM、DCM 与完整分类网络。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SamePadConv2d(nn.Conv2d):
    """为偶数卷积核提供 TensorFlow 风格的 SAME 非对称填充。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先计算 SAME 填充，再执行二维卷积。"""

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


class ICFEM(nn.Module):
    """用 8x2、16x2、32x2 三个尺度提取极化码码间结构特征。"""

    def __init__(self, in_channels: int = 1, channels: int = 16):
        """构造论文图 4 所示的码间特征提取模块。"""

        super().__init__()
        self.local_conv = SamePadConv2d(in_channels, channels, kernel_size=(3, 1), bias=False)
        self.reducers = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False) for _ in range(3)
        )
        self.multiscale_convs = nn.ModuleList(
            SamePadConv2d(channels, channels, kernel_size=(height, 2), bias=False)
            for height in (8, 16, 32)
        )
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """并行提取三种感受野的特征并融合为固定通道数。"""

        x = self.activation(self.local_conv(x))
        branches = [
            self.activation(conv(self.activation(reducer(x))))
            for reducer, conv in zip(self.reducers, self.multiscale_convs)
        ]
        return self.activation(self.norm(self.fuse(torch.cat(branches, dim=1))))


class ResidualBlock(nn.Module):
    """实现 DCM 内的预激活残差块，并按需完成步长为 2 的降维。"""

    def __init__(self, channels: int, stride: int = 1):
        """构造标准残差块或论文图 5(a) 中的降维残差块。"""

        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv1 = SamePadConv2d(channels, channels, kernel_size=(3, 2), stride=stride, bias=False)
        self.conv2 = SamePadConv2d(channels, channels, kernel_size=(3, 2), bias=False)
        self.shortcut = (
            SamePadConv2d(channels, channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 BN-ReLU-卷积两次，并与捷径分支相加。"""

        residual = self.shortcut(x)
        x = self.conv1(F.relu(self.norm1(x), inplace=True))
        x = self.conv2(F.relu(self.norm2(x), inplace=True))
        return x + residual


class DCM(nn.Module):
    """串联一个降维残差块和一个标准残差块以提取抽象特征。"""

    def __init__(self, channels: int = 16):
        """构造论文图 5(c) 中的一组 Data Coupling Module。"""

        super().__init__()
        self.reduction_block = ResidualBlock(channels, stride=2)
        self.standard_block = ResidualBlock(channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先将空间尺寸减半，再用标准残差块细化特征。"""

        return self.standard_block(self.reduction_block(x))


class TDMRNet(nn.Module):
    """组合二维重排后的输入归一化、ICFEM、两个 DCM 和 Softmax 前分类头。"""

    def __init__(self, num_classes: int = 18, channels: int = 16, num_dcms: int = 2):
        """按论文最优配置构造 TDMRNet；默认固定使用两个 DCM。"""

        super().__init__()
        if num_dcms != 2:
            raise ValueError("本复现步骤严格按照论文最优配置使用 2 个 DCM。")
        self.input_norm = nn.BatchNorm2d(1)
        self.icfem = ICFEM(1, channels)
        self.dcms = nn.Sequential(*(DCM(channels) for _ in range(num_dcms)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """从 ``[B,1,256,W]`` LLR 矩阵生成参数类别的未归一化分数。"""

        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != 256 or x.shape[3] < 4:
            raise ValueError(f"模型输入应为 [B,1,256,W] 且 W>=4，实际为 {tuple(x.shape)}。")
        x = self.icfem(self.input_norm(x))
        x = self.dcms(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中需要梯度更新的参数数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
