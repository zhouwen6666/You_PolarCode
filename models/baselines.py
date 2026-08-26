"""论文第 4.4 节用于与 TDMRNet 比较的 DenseNet、Inception 和 ResNet。"""

import torch
from torch import nn


class DenseLayer(nn.Module):
    """实现 DenseNet 中将新特征与历史特征拼接的稠密层。"""

    def __init__(self, in_channels: int, growth_rate: int):
        """构造 BN-ReLU-3x3 卷积，并指定每层新增通道数。"""

        super().__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把本层新特征与输入沿通道维拼接。"""

        return torch.cat((x, self.features(x)), dim=1)


class DenseBlock(nn.Module):
    """串联若干 DenseLayer 形成论文所述 dense block。"""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        """根据输入通道、增长率和层数构造稠密连接。"""

        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次执行稠密层并返回累计特征。"""

        return self.layers(x)


class DenseNetBaseline(nn.Module):
    """按论文说明使用两个 dense block 和一个 transition layer。"""

    def __init__(self, num_classes: int = 18, channels: int = 16, **_: object):
        """构造两块一过渡的 DenseNet baseline。"""

        super().__init__()
        growth_rate = channels
        self.stem = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.block1 = DenseBlock(channels, growth_rate, num_layers=5)
        transition_channels = self.block1.out_channels // 2
        self.transition = nn.Sequential(
            nn.BatchNorm2d(self.block1.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.block1.out_channels, transition_channels, 1, bias=False),
            nn.AvgPool2d(2),
        )
        self.block2 = DenseBlock(transition_channels, growth_rate, num_layers=5)
        self.norm = nn.BatchNorm2d(self.block2.out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.block2.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """经过两个 dense block 和一个 transition 输出分类 logits。"""

        x = self.block1(self.stem(x))
        x = self.block2(self.transition(x))
        x = torch.relu(self.norm(x))
        return self.classifier(self.pool(x).flatten(1))


class InceptionModule(nn.Module):
    """使用 1x1、3x3、5x5 和池化四分支提取多尺度特征。"""

    def __init__(self, in_channels: int, branch_channels: int):
        """构造经典 Inception 四分支模块并保持空间尺寸。"""

        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, branch_channels, 1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """拼接四条不同感受野分支的输出。"""

        outputs = (self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x))
        return torch.relu(torch.cat(outputs, dim=1))


class InceptionBaseline(nn.Module):
    """按论文说明串联六个 Inception module。"""

    def __init__(self, num_classes: int = 18, channels: int = 16, **_: object):
        """构造六模块 Inception baseline 和统一分类头。"""

        super().__init__()
        branch_channels = channels * 2
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        modules = []
        in_channels = channels
        for module_index in range(6):
            modules.append(InceptionModule(in_channels, branch_channels))
            in_channels = branch_channels * 4
            if module_index in (1, 3):
                modules.append(nn.MaxPool2d(2))
        self.inception_stack = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过论文指定的六个 Inception module 输出分类 logits。"""

        return self.classifier(self.pool(self.inception_stack(self.stem(x))).flatten(1))


class BasicResidualBlock(nn.Module):
    """实现 ResNet2D 基线使用的标准两层残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """构造主分支和按需执行通道、尺寸匹配的捷径分支。"""

        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将残差主分支与捷径分支相加并执行激活。"""

        return self.activation(self.main(x) + self.shortcut(x))


class ResNetBaseline(nn.Module):
    """使用常规 3x3 残差块实现论文比较的 ResNet。"""

    def __init__(self, num_classes: int = 18, channels: int = 16, **_: object):
        """构造三个尺度阶段的轻量 ResNet2D。"""

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            BasicResidualBlock(channels, channels),
            BasicResidualBlock(channels, channels * 2, stride=2),
            BasicResidualBlock(channels * 2, channels * 2),
            BasicResidualBlock(channels * 2, channels * 4, stride=2),
            BasicResidualBlock(channels * 4, channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """从统一二维 LLR 输入生成指定类别数量的 logits。"""

        return self.classifier(self.pool(self.blocks(self.stem(x))).flatten(1))
