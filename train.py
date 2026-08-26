"""仅使用 train 目录训练所选模型，不读取或评估 test 数据。"""

from __future__ import annotations

import os

# Windows WDDM 驱动下 CUBLAS 工作空间分配失败，需在 import torch 前设置
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from config import TrainConfig
from data import NpyShardDataset
from losses import FocalLoss, load_class_weights
from models import available_models, build_model, count_trainable_parameters


def set_random_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子以提高复现实验的一致性。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """优先选择 CUDA；不可用时自动回退到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    """返回训练设备名称；CUDA 模式下同时包含显卡型号和 CUDA 运行时版本。"""

    if device.type == "cuda":
        return f"cuda | {torch.cuda.get_device_name(device)} | PyTorch CUDA {torch.version.cuda}"
    return "cpu"


def peak_gpu_memory_mb(device: torch.device) -> float:
    """返回当前轮次 CUDA 峰值显存（MiB）；CPU 训练时返回 0。"""

    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def build_train_loaders(config: TrainConfig, max_samples: int | None = None) -> tuple[DataLoader, DataLoader]:
    """仅从 train 数据构建训练与内部验证 DataLoader。"""

    dataset = NpyShardDataset(
        config.dataset_dir / "train",
        sample_length=config.sample_length,
        matrix_rows=config.matrix_rows,
    )
    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples 至少为 2。")
        sample_count = min(max_samples, len(dataset))
        sample_generator = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(len(dataset), generator=sample_generator)[:sample_count].tolist()
        dataset = Subset(dataset, indices)

    validation_size = max(1, round(len(dataset) * config.validation_ratio))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=generator)

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation_set, shuffle=False, **common)
    return train_loader, validation_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> tuple[float, float]:
    """执行一个训练或验证轮次，并返回样本加权损失与分类准确率。"""

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(model: nn.Module, config: TrainConfig, epoch: int, best_loss: float, path: Path) -> None:
    """保存最佳模型权重、训练轮次、损失和复现实验配置。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_validation_loss": best_loss,
            "model_state_dict": model.state_dict(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        },
        path,
    )


def build_criterion(config: TrainConfig, device: torch.device) -> nn.Module:
    """根据 config.loss_type 构建对应的损失函数。"""

    if config.loss_type == "focal":
        alpha = None
        if config.class_weights_path is not None:
            alpha = load_class_weights(config.class_weights_path, config.num_classes, mode=config.class_weight_mode).to(device)
        criterion = FocalLoss(gamma=config.focal_gamma, alpha=alpha)
        print(f"loss=FocalLoss(gamma={config.focal_gamma}, alpha=mode:{config.class_weight_mode})")
    elif config.loss_type == "weighted_ce":
        if config.class_weights_path is None:
            raise ValueError("weighted_ce 损失需要 --class-weights-from 参数。")
        weights = load_class_weights(config.class_weights_path, config.num_classes, mode=config.class_weight_mode).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"loss=CrossEntropyLoss(weight_mode={config.class_weight_mode})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("loss=CrossEntropyLoss")
    return criterion


def train_model(config: TrainConfig, max_samples: int | None = None) -> Path:
    """只用训练数据拟合所选模型，并以验证损失执行统一早停策略。"""

    set_random_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    train_loader, validation_loader = build_train_loaders(config, max_samples=max_samples)
    model = build_model(config.model_name, config.num_classes, config.channels, config.num_dcms).to(device)

    # 两阶段微调：从已有 checkpoint 恢复权重
    if config.resume_checkpoint is not None:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"已恢复权重: {config.resume_checkpoint} (epoch={checkpoint.get('epoch', '?')})")

    criterion = build_criterion(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    checkpoint_path = config.output_dir / f"{config.model_name}_best.pt"
    history_path = config.output_dir / "training_history.csv"

    print(f"model={config.model_name} device={describe_device(device)} train={len(train_loader.dataset)} validation={len(validation_loader.dataset)}")
    print(f"input=[B,1,{config.matrix_rows},{config.matrix_columns}] parameters={count_trainable_parameters(model):,}")

    use_acc = config.early_stop_metric == "acc"
    best_loss = float("inf")
    best_acc = -1.0
    stale_epochs = 0
    with history_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
                "epoch_seconds",
                "peak_gpu_memory_mb",
            ]
        )
        for epoch in range(1, config.max_epochs + 1):
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, grad_clip=config.grad_clip)
            validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, device)
            epoch_seconds = time.perf_counter() - epoch_start
            learning_rate = optimizer.param_groups[0]["lr"]
            gpu_memory_mb = peak_gpu_memory_mb(device)
            writer.writerow(
                [
                    epoch,
                    learning_rate,
                    train_loss,
                    train_accuracy,
                    validation_loss,
                    validation_accuracy,
                    epoch_seconds,
                    gpu_memory_mb,
                ]
            )
            stream.flush()
            print(
                f"epoch={epoch:03d} lr={learning_rate:.2e} "
                f"train_loss={train_loss:.6f} train_acc={train_accuracy:.4f} "
                f"val_loss={validation_loss:.6f} val_acc={validation_accuracy:.4f} "
                f"time={epoch_seconds:.1f}s gpu_peak={gpu_memory_mb:.1f}MiB"
            )

            if use_acc:
                improved = validation_accuracy > best_acc
            else:
                improved = validation_loss < best_loss
            if improved:
                if use_acc:
                    best_acc = validation_accuracy
                    metric_value = validation_accuracy
                else:
                    best_loss = validation_loss
                    metric_value = validation_loss
                stale_epochs = 0
                save_checkpoint(model, config, epoch, metric_value, checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    metric_name = "验证准确率" if use_acc else "验证损失"
                    print(f"{metric_name}连续 {config.patience} 轮未改善，提前停止。")
                    break

    return checkpoint_path


def parse_args() -> argparse.Namespace:
    """解析数据路径、训练轮数和小规模冒烟测试参数。"""

    parser = argparse.ArgumentParser(description="使用统一流程训练指定模型。")
    parser.add_argument("--model", choices=available_models(), default="tdmrnet", help="选择需要训练的模型。")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="包含 train 和 test 子目录的外部数据集根目录。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖论文默认的最大 600 轮。")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖论文默认 batch size 64。")
    parser.add_argument("--lr", type=float, default=None, help="覆盖论文默认学习率 0.001。")
    parser.add_argument("--grad-clip", type=float, default=None, help="梯度范数裁剪阈值，防止训练发散。")
    parser.add_argument("--weight-decay", type=float, default=None, help="Adam 权重衰减系数（L2 正则化），默认 0。")
    parser.add_argument("--loss-type", choices=["ce", "focal", "weighted_ce"], default=None, help="损失函数类型，默认 ce（标准交叉熵）。")
    parser.add_argument("--early-stop-metric", choices=["loss", "acc"], default=None, help="早停与最优模型选优指标，默认 loss；val_loss 波动大时建议用 acc。")
    parser.add_argument("--patience", type=int, default=None, help="覆盖早停耐心轮数，默认 10。")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Focal Loss 聚焦参数，默认 2.0。")
    parser.add_argument("--resume", type=Path, default=None, help="从已有 checkpoint 恢复权重，用于两阶段微调。")
    parser.add_argument("--class-weights-from", type=Path, default=None, help="类别权重来源（test_metrics.json 路径），用于 weighted_ce 或 focal+alpha。")
    parser.add_argument("--weight-mode", choices=["inverse_acc", "sqrt_inverse_acc", "manual"], default=None, help="类别权重缩放模式，默认 inverse_acc；manual 表示直接从 JSON 的 class_weights 字段读取。")
    parser.add_argument("--output-suffix", type=str, default=None, help="输出目录后缀，如 _focal、_weighted，自动拼接为 outputs/{model}_{suffix}/。")
    parser.add_argument("--max-samples", type=int, default=None, help="仅用于快速验证代码链路；正式训练请勿设置。")
    return parser.parse_args()


def main() -> None:
    """创建配置并启动只依赖 train 目录的指定模型训练。"""

    args = parse_args()
    config = TrainConfig()
    updates = {"model_name": args.model}
    if args.dataset_dir is not None:
        updates["dataset_dir"] = args.dataset_dir.resolve()
    if args.epochs is not None:
        updates["max_epochs"] = args.epochs
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.lr is not None:
        updates["learning_rate"] = args.lr
    if args.grad_clip is not None:
        updates["grad_clip"] = args.grad_clip
    if args.weight_decay is not None:
        updates["weight_decay"] = args.weight_decay
    if args.loss_type is not None:
        updates["loss_type"] = args.loss_type
    if args.focal_gamma is not None:
        updates["focal_gamma"] = args.focal_gamma
    if args.early_stop_metric is not None:
        updates["early_stop_metric"] = args.early_stop_metric
    if args.patience is not None:
        updates["patience"] = args.patience
    if args.resume is not None:
        updates["resume_checkpoint"] = args.resume.resolve()
    if args.class_weights_from is not None:
        updates["class_weights_path"] = args.class_weights_from.resolve()
    if args.weight_mode is not None:
        updates["class_weight_mode"] = args.weight_mode
    # 输出目录命名
    dir_name = args.model
    if args.output_suffix is not None:
        dir_name = f"{args.model}_{args.output_suffix.lstrip('_')}"
    if args.max_samples is not None:
        updates["output_dir"] = config.output_dir.parent / "smoke_outputs" / dir_name
    elif dir_name != "tdmrnet":
        updates["output_dir"] = config.output_dir / dir_name
    config = replace(config, **updates)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoint = train_model(config, max_samples=args.max_samples)
    print(f"最佳模型已保存：{checkpoint}")


if __name__ == "__main__":
    main()
