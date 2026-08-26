"""针对类别不平衡与难例分类的损失函数。

- FocalLoss: (1-p_t)^gamma * CE，自动降低易分类样本权重、聚焦难例
- load_class_weights: 从 test_metrics.json 读取分类别准确率，构建归一化类别权重
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Focal Loss: 通过 (1-p_t)^gamma 降低易分类样本权重，gamma=0 退化为标准交叉熵。"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def load_class_weights(
    metrics_path: str | Path,
    num_classes: int = 18,
    epsilon: float = 0.01,
    mode: str = "inverse_acc",
) -> torch.Tensor:
    """从 test_metrics.json 读取分类别准确率，构建均值为 1 的归一化类别权重。

    mode 可选 inverse_acc（默认）、sqrt_inverse_acc 或 manual。"""

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    if mode == "manual":
        raw_values = metrics["class_weights"]
        if len(raw_values) != num_classes:
            raise ValueError(
                f"manual weights 数量 {len(raw_values)} 与 num_classes={num_classes} 不匹配。"
            )
    else:
        class_metrics = metrics["class_metrics"]
        accuracies = [row["accuracy"] for row in class_metrics]
        if len(accuracies) != num_classes:
            raise ValueError(
                f"metrics 中的类别数 {len(accuracies)} 与 num_classes={num_classes} 不匹配。"
            )
        raw_values = [1.0 / max(acc, epsilon) for acc in accuracies]
        if mode == "sqrt_inverse_acc":
            raw_values = [v ** 0.5 for v in raw_values]
    raw = torch.tensor(raw_values, dtype=torch.float32)
    # 归一化使均值为 1，保持总体损失尺度不变
    raw *= num_classes / raw.sum()
    return raw
