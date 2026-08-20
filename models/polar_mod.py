"""PolarMod: 调制机制动态通道选择 + TDMRNet 多尺度码间特征。


核心公式:
    context = g(act(dw(f(x))))          # 1x1降维 → DWConv空间 → 1x1升维
    modulated = act(fc1(x)) * context   # MLP扩展 × 上下文调制
    out = fc2(dropout(modulated))       # 投影回原维度

设计思路:
    - ICFEM: 完全保留 TDMRNet 多尺度码字边界检测
    - ModResidualBlock: TDMRNet 两次空间卷积之间插入调制 MLP
      第一次卷积提取空间码字特征 → 调制动态选择通道 → 第二次卷积细化
    - 低容量: channels=16, mlp_ratio=2, context_dim=4
    - 多尺度上下文: DWConv [(3,2),(7,2)] 并行，扩展空间感受野
    - 正则化: dropout=0.3 + weight_decay 抑制过拟合

v1 单尺度上下文 (3,2) 导致 P06(N=64,R=3/4) 完全坍塌至 0.9%，
v2 改为多尺度上下文 [(3,2),(7,2)] + 增大 dropout=0.3，
扩展空间感受野以区分不同码长的 R=3/4 码字。

"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .tdmrnet import SamePadConv2d, ICFEM, ResidualBlock


class ModContext(nn.Module):
    """调制上下文生成器: 1x1降维 → 多尺度DWConv空间 → GELU → 1x1升维。

    输出与调制 MLP 隐层同维度的上下文信号，用于元素乘法调制。
    多个 DWConv 核尺寸并行提取不同尺度的码字边界空间模式，
    求和融合后升维生成调制信号。

    流程:
        x [B, C, H, W]
        → 1x1 Conv (C → d)             降维到低维上下文空间
        → DWConv_k1(d) + DWConv_k2(d)  多尺度空间上下文（并行求和）
        → GELU
        → 1x1 Conv (d → C*mlp_ratio)   升维到调制维度
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        mlp_ratio: int,
        context_sizes: list[tuple[int, int]] | None = None,
    ):
        """构造多尺度调制上下文生成器。"""

        super().__init__()
        if context_sizes is None:
            context_sizes = [(3, 2), (7, 2)]

        self.f = nn.Conv2d(channels, context_dim, kernel_size=1, bias=False)
        # 多尺度分组卷积做空间上下文，SAME 填充保持码字行对齐
        self.dw_branches = nn.ModuleList([
            SamePadConv2d(
                context_dim, context_dim,
                kernel_size=ks, groups=context_dim, bias=False,
            )
            for ks in context_sizes
        ])
        self.act = nn.GELU()
        self.g = nn.Conv2d(context_dim, channels * mlp_ratio, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成 [B, C*mlp_ratio, H, W] 多尺度调制上下文。"""

        c = self.f(x)
        # 多尺度空间上下文并行提取并求和
        out = sum(self.act(branch(c)) for branch in self.dw_branches)
        return self.g(out)


class ModResidualBlock(nn.Module):
    """调制残差块: TDMRNet 两层空间卷积之间插入调制 MLP。

    流程:
        x → norm1 → ReLU → conv1(空间, stride) →      # 第一层空间特征
            → mod_norm → fc1(expand) → GELU → ×context → fc2(project) → dropout → +h  # 调制
            → norm2 → ReLU → conv2(空间) →              # 第二层空间细化
            + shortcut(x)

    调制 MLP 在两次卷积之间工作:
        - 第一次卷积提取空间码字边界特征
        - 调制用 DWConv 上下文动态选择有用通道
        - 第二次卷积在调制后的特征上进一步细化
    """

    def __init__(
        self,
        channels: int,
        stride: int = 1,
        mlp_ratio: int = 2,
        context_dim: int = 4,
        context_sizes: list[tuple[int, int]] | None = None,
        drop_rate: float = 0.3,
    ):
        """构造调制残差块。"""

        super().__init__()
        if context_sizes is None:
            context_sizes = [(3, 2), (7, 2)]
        expanded = channels * mlp_ratio

        # 第一层空间卷积（与 TDMRNet 一致）
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = SamePadConv2d(
            channels, channels, kernel_size=(3, 2), stride=stride, bias=False,
        )

        # 调制 MLP（插入在两次卷积之间）
        self.mod_norm = nn.BatchNorm2d(channels)
        self.mod_context = ModContext(
            channels, context_dim, mlp_ratio, context_sizes,
        )
        self.mod_fc1 = nn.Conv2d(channels, expanded, kernel_size=1, bias=False)
        self.mod_act = nn.GELU()
        self.mod_fc2 = nn.Conv2d(expanded, channels, kernel_size=1, bias=False)
        self.mod_drop = nn.Dropout2d(drop_rate)

        # 第二层空间卷积（与 TDMRNet 一致）
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = SamePadConv2d(
            channels, channels, kernel_size=(3, 2), bias=False,
        )

        # 捷径分支
        self.shortcut = (
            SamePadConv2d(channels, channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """两次空间卷积 + 中间调制。"""

        residual = self.shortcut(x)

        # 第一层空间卷积
        h = self.conv1(F.relu(self.norm1(x), inplace=True))

        # 调制 MLP: norm → fc1 → act → ×context → fc2 → dropout
        mod_input = self.mod_norm(h)
        context = self.mod_context(mod_input)
        mod = self.mod_act(self.mod_fc1(mod_input))
        mod = mod * context
        h = h + self.mod_drop(self.mod_fc2(mod))

        # 第二层空间卷积
        h = self.conv2(F.relu(self.norm2(h), inplace=True))

        return h + residual


class ModDCM(nn.Module):
    """调制数据耦合模块: 降维块(stride=2) + 标准块，均含调制。"""

    def __init__(
        self,
        channels: int = 16,
        mlp_ratio: int = 2,
        context_dim: int = 4,
        context_sizes: list[tuple[int, int]] | None = None,
        drop_rate: float = 0.3,
    ):
        """构造一个调制 DCM。"""

        super().__init__()
        if context_sizes is None:
            context_sizes = [(3, 2), (7, 2)]
        self.reduction_block = ModResidualBlock(
            channels, stride=2, mlp_ratio=mlp_ratio,
            context_dim=context_dim, context_sizes=context_sizes,
            drop_rate=drop_rate,
        )
        self.standard_block = ModResidualBlock(
            channels, stride=1, mlp_ratio=mlp_ratio,
            context_dim=context_dim, context_sizes=context_sizes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先降维再细化。"""

        return self.standard_block(self.reduction_block(x))


class PolarMod(nn.Module):
    """PolarMod: ICFEM 多尺度码间特征 + 调制动态通道选择。

    架构流程:
        [B, 1, 256, 32] → InputNorm → ICFEM → ModDCM×2 → GAP → Classifier

    与 TDMRNet 的区别:
        - ICFEM 完全保留
        - 每个 ResidualBlock 内插入调制 MLP（DWConv 上下文 × 隐层特征）
        - 调制机制提供输入依赖的动态通道选择，增强判别力
    """

    def __init__(
        self,
        num_classes: int = 18,
        channels: int = 16,
        num_dcms: int = 2,
        mlp_ratio: int = 2,
        context_dim: int = 4,
        context_sizes: list[tuple[int, int]] | None = None,
        drop_rate: float = 0.3,
    ):
        """构造 PolarMod 网络。"""

        super().__init__()
        if context_sizes is None:
            context_sizes = [(3, 2), (7, 2)]
        self.input_norm = nn.BatchNorm2d(1)
        self.icfem = ICFEM(1, channels)
        self.dcms = nn.Sequential(
            *(ModDCM(channels, mlp_ratio, context_dim, context_sizes, drop_rate)
              for _ in range(num_dcms))
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """从 [B, 1, 256, W] LLR 矩阵生成参数类别分数。"""

        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != 256 or x.shape[3] < 4:
            raise ValueError(
                f"模型输入应为 [B,1,256,W] 且 W>=4，实际为 {tuple(x.shape)}。"
            )
        x = self.icfem(self.input_norm(x))
        x = self.dcms(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
