"""加载指定模型的最佳权重，在 test 数据上输出统一评估指标。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import DEFAULT_DATASET_DIR, PROJECT_DIR
from data import NpyShardDataset
from models import available_models, build_model, count_trainable_parameters


def make_label_names(num_classes: int) -> list[str]:
    """按类别总数生成与数据文件一致的 P01、P02 等类别名称。"""

    width = max(2, len(str(num_classes)))
    return [f"P{index:0{width}d}" for index in range(1, num_classes + 1)]


def safe_divide(numerator: float, denominator: float) -> float:
    """执行安全除法；分母为零时返回 0，避免稀有类别指标产生 NaN。"""

    return numerator / denominator if denominator else 0.0


def load_model(checkpoint_path: Path, device: torch.device, fallback_model_name: str = "tdmrnet") -> tuple[nn.Module, dict]:
    """从可信的本地检查点恢复模型结构、权重和训练元数据。"""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_name = str(config.get("model_name", fallback_model_name))
    model = build_model(
        model_name=model_name,
        num_classes=int(config.get("num_classes", 18)),
        channels=int(config.get("channels", 16)),
        num_dcms=int(config.get("num_dcms", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def evaluate_test_set(
    model: nn.Module,
    dataset: NpyShardDataset,
    batch_size: int,
    device: torch.device,
    num_classes: int,
) -> dict:
    """遍历完整 test 集并计算 loss、混淆矩阵、分 SNR 结果和推理速度。"""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    snr_correct: dict[int, int] = {}
    snr_total: dict[int, int] = {}
    total_loss = 0.0
    total_samples = 0
    inference_seconds = 0.0

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(loader):
            start_index = batch_index * batch_size
            snrs = [dataset.metadata_at(start_index + offset)[1] for offset in range(labels.size(0))]
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started

            total_loss += criterion(logits, labels).item()
            predictions = logits.argmax(dim=1)
            true_values = labels.cpu().numpy()
            predicted_values = predictions.cpu().numpy()
            for true_label, predicted_label, snr in zip(true_values, predicted_values, snrs):
                confusion[true_label, predicted_label] += 1
                snr_total[snr] = snr_total.get(snr, 0) + 1
                snr_correct[snr] = snr_correct.get(snr, 0) + int(true_label == predicted_label)
            total_samples += labels.size(0)

            if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
                print(f"已测试 {total_samples}/{len(dataset)} 条样本")

    class_metrics = []
    for class_index, class_name in enumerate(make_label_names(num_classes)):
        true_positive = int(confusion[class_index, class_index])
        support = int(confusion[class_index].sum())
        predicted = int(confusion[:, class_index].sum())
        precision = safe_divide(true_positive, predicted)
        recall = safe_divide(true_positive, support)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        class_metrics.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": recall,
                "support": support,
            }
        )

    per_snr = [
        {"snr_db": snr, "accuracy": safe_divide(snr_correct[snr], snr_total[snr]), "support": snr_total[snr]}
        for snr in sorted(snr_total)
    ]
    overall_accuracy = safe_divide(int(np.trace(confusion)), total_samples)
    low_mask = [row for row in per_snr if -4 <= row["snr_db"] <= 2]
    high_mask = [row for row in per_snr if 4 <= row["snr_db"] <= 20]
    return {
        "samples": total_samples,
        "loss": total_loss / total_samples,
        "accuracy": overall_accuracy,
        "macro_precision": float(np.mean([row["precision"] for row in class_metrics])),
        "macro_recall": float(np.mean([row["recall"] for row in class_metrics])),
        "macro_f1": float(np.mean([row["f1"] for row in class_metrics])),
        "low_snr_accuracy_-4_to_2_db": float(np.mean([row["accuracy"] for row in low_mask])),
        "high_snr_accuracy_4_to_20_db": float(np.mean([row["accuracy"] for row in high_mask])),
        "inference_ms_per_sample": inference_seconds * 1000.0 / total_samples,
        "throughput_samples_per_second": total_samples / inference_seconds,
        "class_metrics": class_metrics,
        "per_snr": per_snr,
        "confusion_matrix": confusion.tolist(),
    }


def save_results(results: dict, output_dir: Path) -> None:
    """将完整指标保存为 JSON，并分别导出分类别、分 SNR 和混淆矩阵 CSV。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for filename, rows in (("per_class_metrics.csv", results["class_metrics"]), ("per_snr_metrics.csv", results["per_snr"])):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savetxt(output_dir / "confusion_matrix.csv", np.asarray(results["confusion_matrix"]), fmt="%d", delimiter=",")


def print_summary(results: dict, checkpoint: dict, parameter_count: int) -> None:
    """在终端输出最值得关注的总体、分类别和分 SNR 测试指标。"""

    print("\n===== TDMRNet 测试结果 =====")
    print(f"checkpoint_epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"parameters: {parameter_count:,}")
    for key in ("samples", "loss", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                "low_snr_accuracy_-4_to_2_db", "high_snr_accuracy_4_to_20_db",
                "inference_ms_per_sample", "throughput_samples_per_second"):
        value = results[key]
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("\n每类指标（accuracy 等同于该类 recall）：")
    for row in results["class_metrics"]:
        print(f"{row['class']}: acc={row['accuracy']:.4f} precision={row['precision']:.4f} f1={row['f1']:.4f}")
    print("\n每个 SNR 的准确率：")
    print("  ".join(f"{row['snr_db']:+d}dB={row['accuracy']:.4f}" for row in results["per_snr"]))


def parse_args() -> argparse.Namespace:
    """解析模型、数据集、输出目录和测试批量大小。"""

    parser = argparse.ArgumentParser(description="在 test 数据集上评估指定模型。")
    parser.add_argument("--model", choices=available_models(), default="tdmrnet")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    """加载最佳权重、运行完整 test 集评估并落盘关键指标。"""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        model_dir = PROJECT_DIR / "outputs" if args.model == "tdmrnet" else PROJECT_DIR / "outputs" / args.model
        checkpoint_path = model_dir / f"{args.model}_best.pt"
    output_dir = args.output_dir or checkpoint_path.parent / "test_results"
    model, checkpoint = load_model(checkpoint_path, device, args.model)
    checkpoint_config = checkpoint.get("config", {})
    sample_length = int(checkpoint_config.get("sample_length", 8192))
    matrix_rows = int(checkpoint_config.get("matrix_rows", 256))
    num_classes = int(checkpoint_config.get("num_classes", 18))
    dataset = NpyShardDataset(args.dataset_dir / "test", sample_length=sample_length, matrix_rows=matrix_rows)
    print(f"model={args.model} device={device} test_samples={len(dataset)} checkpoint={checkpoint_path}")
    results = evaluate_test_set(model, dataset, args.batch_size, device, num_classes)
    save_results(results, output_dir)
    print_summary(results, checkpoint, count_trainable_parameters(model))
    print(f"\n完整结果已保存：{output_dir}")


if __name__ == "__main__":
    main()
