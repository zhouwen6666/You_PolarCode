"""TDMRNet 复现项目 - 全量可视化脚本。

从 outputs/ 目录下所有已训练模型的 training_history.csv 和 test_results/
中读取数据，自动生成 9 张可视化图表。支持任意数量的模型对比，未训练的
模型会自动跳过。

用法：
    python plot_results.py
依赖：numpy, matplotlib
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 路径配置 ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 全局绘图样式 ──────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

# 模型颜色与显示名
COLORS = {
    "tdmrnet": "#1f77b4",
    "densenet": "#2ca02c",
    "inception": "#ff7f0e",
    "resnet": "#d62728",
    "polar_mod_focal_g3": "#9467bd",
    "polar_mod_weighted_sqrt": "#8c564b",
}
DISPLAY_NAMES = {
    "tdmrnet": "TDMRNet",
    "densenet": "DenseNet",
    "inception": "Inception",
    "resnet": "ResNet",
    "polar_mod_focal_g3": "PolarMod-FocalG3",
    "polar_mod_weighted_sqrt": "PolarMod-WSqrt",
}
DEFAULT_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]


# ── 工具函数 ──────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    """读取 CSV 文件，数值列自动转 float，非数值列保留字符串。"""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def col(rows: list[dict], name: str) -> np.ndarray:
    return np.array([r[name] for r in rows])


def discover_models() -> list[dict]:
    """扫描 outputs/ 目录，发现所有已训练模型。

    TDMRNet 的文件直接在 outputs/ 下，其他模型在 outputs/<model>/ 下。
    返回列表，每项包含 name, hist_path, hist(数据), color, checkpoint_exists。
    """
    models = []

    # TDMRNet：文件直接在 outputs/ 下
    tdmr_hist = OUTPUT_DIR / "training_history.csv"
    if tdmr_hist.exists():
        models.append({
            "name": "tdmrnet",
            "display": "TDMRNet",
            "hist_path": tdmr_hist,
            "hist": load_csv(tdmr_hist),
            "color": COLORS.get("tdmrnet", DEFAULT_COLORS[0]),
        })

    # 其他模型：在 outputs/<model>/ 下
    for sub in sorted(OUTPUT_DIR.iterdir()):
        if not sub.is_dir() or sub.name == "figures":
            continue
        hist = sub / "training_history.csv"
        if hist.exists():
            name = sub.name
            models.append({
                "name": name,
                "display": DISPLAY_NAMES.get(name, name),
                "hist_path": hist,
                "hist": load_csv(hist),
                "color": COLORS.get(name, DEFAULT_COLORS[len(models) % len(DEFAULT_COLORS)]),
            })

    return models


def discover_test_results(model_name: str) -> dict | None:
    """查找指定模型的测试结果目录。"""
    if model_name == "tdmrnet":
        tr = OUTPUT_DIR / "test_results"
    else:
        tr = OUTPUT_DIR / model_name / "test_results"
    if not (tr / "test_metrics.json").exists():
        return None
    return {
        "dir": tr,
        "metrics": json.loads((tr / "test_metrics.json").read_text(encoding="utf-8")),
        "per_class": load_csv(tr / "per_class_metrics.csv"),
        "per_snr": load_csv(tr / "per_snr_metrics.csv"),
        "confusion_matrix": np.loadtxt(tr / "confusion_matrix.csv", delimiter=","),
    }


# ── 发现已训练模型 ─────────────────────────────────────────
models = discover_models()
if not models:
    print("错误：在 outputs/ 下未找到任何 training_history.csv，请先训练模型。")
    raise SystemExit(1)

print(f"发现 {len(models)} 个已训练模型：{', '.join(m['display'] for m in models)}")

# 查找第一个有测试结果的模型（优先 tdmrnet）
primary_model = None
for m in models:
    tr = discover_test_results(m["name"])
    if tr is not None:
        m["test_results"] = tr
        if primary_model is None:
            primary_model = m
    else:
        m["test_results"] = None

if primary_model is None:
    print("警告：未找到任何 test_results，将仅生成训练曲线图。")
else:
    print(f"主测试模型：{primary_model['display']}")


# ============================================================
# 图 1：损失曲线（所有模型对比）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for m in models:
    ep = col(m["hist"], "epoch").astype(int)
    ax.plot(ep, col(m["hist"], "train_loss"), "-", color=m["color"], linewidth=2, label=f"{m['display']} Train Loss")
    ax.plot(ep, col(m["hist"], "validation_loss"), "--", color=m["color"], linewidth=2, label=f"{m['display']} Val Loss")

# 标注第一个模型（TDMRNet）的最佳轮次
if models:
    best_idx = np.argmin(col(models[0]["hist"], "validation_loss"))
    ax.axvline(col(models[0]["hist"], "epoch")[best_idx], color="gray", linestyle=":", alpha=0.6)
    ax.annotate(
        f"Best (Epoch {int(col(models[0]['hist'], 'epoch')[best_idx])})",
        xy=(col(models[0]["hist"], "epoch")[best_idx], col(models[0]["hist"], "validation_loss")[best_idx]),
        xytext=(col(models[0]["hist"], "epoch")[best_idx] + 3, col(models[0]["hist"], "validation_loss")[best_idx] + 0.15),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"),
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Training & Validation Loss: All Models")
ax.legend(loc="upper right")
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_loss_curves.png")
plt.close(fig)
print("Saved: 01_loss_curves.png")


# ============================================================
# 图 2：准确率曲线（所有模型对比）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for m in models:
    ep = col(m["hist"], "epoch").astype(int)
    ax.plot(ep, col(m["hist"], "train_accuracy") * 100, "-", color=m["color"], linewidth=2, label=f"{m['display']} Train Acc")
    ax.plot(ep, col(m["hist"], "validation_accuracy") * 100, "--", color=m["color"], linewidth=2, label=f"{m['display']} Val Acc")

# 标注最佳验证准确率
if models:
    best_acc_idx = np.argmax(col(models[0]["hist"], "validation_accuracy"))
    ax.axvline(col(models[0]["hist"], "epoch")[best_acc_idx], color="gray", linestyle=":", alpha=0.6)
    ax.annotate(
        f"Best Val Acc {col(models[0]['hist'], 'validation_accuracy')[best_acc_idx] * 100:.2f}%",
        xy=(col(models[0]["hist"], "epoch")[best_acc_idx], col(models[0]["hist"], "validation_accuracy")[best_acc_idx] * 100),
        xytext=(col(models[0]["hist"], "epoch")[best_acc_idx] + 2, col(models[0]["hist"], "validation_accuracy")[best_acc_idx] * 100 - 8),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"),
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Training & Validation Accuracy: All Models")
ax.legend(loc="lower right")
ax.set_xlim(left=0)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_accuracy_curves.png")
plt.close(fig)
print("Saved: 02_accuracy_curves.png")


# ============================================================
# 图 3：分 SNR 准确率（主模型）
# ============================================================
if primary_model and primary_model["test_results"]:
    per_snr = primary_model["test_results"]["per_snr"]
    snr_vals = [int(r["snr_db"]) for r in per_snr]
    snr_accs = [r["accuracy"] * 100 for r in per_snr]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors_snr = ["#d62728" if a < 50 else "#ff7f0e" if a < 80 else "#2ca02c" for a in snr_accs]
    bars = ax.bar(range(len(snr_vals)), snr_accs, color=colors_snr, width=0.65, edgecolor="black", linewidth=0.5)

    for i, (bar, acc) in enumerate(zip(bars, snr_accs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axvspan(-0.5, 3.5, alpha=0.08, color="red", label="Low SNR (-4 to 2 dB)")
    ax.axvspan(3.5, len(snr_vals) - 0.5, alpha=0.08, color="green", label="High SNR (4 to 20 dB)")

    ax.set_xticks(range(len(snr_vals)))
    ax.set_xticklabels([f"{v} dB" for v in snr_vals])
    ax.set_xlabel("SNR (Eb/N0)")
    ax.set_ylabel("Recognition Accuracy (%)")
    ax.set_title(f"{primary_model['display']} Per-SNR Recognition Accuracy")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_snr_accuracy.png")
    plt.close(fig)
    print("Saved: 03_per_snr_accuracy.png")


# ============================================================
# 图 4：分类别指标（主模型）
# ============================================================
if primary_model and primary_model["test_results"]:
    per_class = primary_model["test_results"]["per_class"]
    classes = [r["class"] for r in per_class]
    num_cls = len(classes)

    precisions = [r["precision"] * 100 for r in per_class]
    recalls = [r["recall"] * 100 for r in per_class]
    f1s = [r["f1"] * 100 for r in per_class]
    accuracies = [r["accuracy"] * 100 for r in per_class]

    x = np.arange(num_cls)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, num_cls * 0.7), 6))
    ax.bar(x - 1.5 * width, precisions, width, label="Precision", color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5 * width, recalls, width, label="Recall", color="#2ca02c", edgecolor="black", linewidth=0.4)
    bars3 = ax.bar(x + 0.5 * width, f1s, width, label="F1-Score", color="#9467bd", edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5 * width, accuracies, width, label="Accuracy", color="#1f77b4", edgecolor="black", linewidth=0.4)

    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7.5, color="#9467bd", fontweight="bold")

    # 码长分组分隔线（6 种码长，每种 3 个码率）
    code_lengths = [32, 64, 128, 256, 512, 1024]
    if num_cls == 18:
        for i in range(3, 18, 3):
            ax.axvline(i - 0.5, color="gray", linestyle="--", alpha=0.5)
        for i, n in enumerate(code_lengths):
            ax.text(i * 3 + 1, 108, f"N={n}", ha="center", fontsize=10, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Polar Code Parameter Class")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{primary_model['display']} Per-Class Metrics: Precision / Recall / F1 / Accuracy")
    ax.legend(loc="upper right", ncol=4)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_class_metrics.png")
    plt.close(fig)
    print("Saved: 04_per_class_metrics.png")


# ============================================================
# 图 5：混淆矩阵热力图（主模型）
# ============================================================
if primary_model and primary_model["test_results"]:
    cm = primary_model["test_results"]["confusion_matrix"]
    class_names = [f"P{i:02d}" for i in range(1, cm.shape[0] + 1)]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig_size = max(8, cm.shape[0] * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    im = ax.imshow(cm_norm, cmap="Blues", aspect="equal", vmin=0, vmax=100)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{int(cm[i, j])}\n({val:.1f}%)", ha="center", va="center",
                    fontsize=7 if len(class_names) > 12 else 8,
                    color=text_color,
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"{primary_model['display']} Confusion Matrix (Count + Row-Normalized %)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Row-Normalized Accuracy (%)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_confusion_matrix.png")
    plt.close(fig)
    print("Saved: 05_confusion_matrix.png")


# ============================================================
# 图 6：每轮训练时间与峰值显存（所有模型对比）
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 每轮训练时间
for m in models:
    ep = col(m["hist"], "epoch").astype(int)
    ax1.plot(ep, col(m["hist"], "epoch_seconds"), "o-", color=m["color"], markersize=4, label=m["display"])
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Seconds per Epoch")
ax1.set_title("Training Time per Epoch")
ax1.legend()

# 峰值显存
gpu_names = [m["display"] for m in models]
gpu_vals = [col(m["hist"], "peak_gpu_memory_mb")[0] for m in models]
bar_colors = [m["color"] for m in models]
bars = ax2.bar(gpu_names, gpu_vals, color=bar_colors, width=0.5, edgecolor="black")
for bar, val in zip(bars, gpu_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15, f"{val:.1f} MiB",
             ha="center", fontweight="bold")
ax2.set_ylabel("Peak GPU Memory (MiB)")
ax2.set_title("Peak GPU Memory Usage")
ax2.set_ylim(0, max(gpu_vals) * 1.2)

fig.tight_layout()
fig.savefig(FIG_DIR / "06_training_efficiency.png")
plt.close(fig)
print("Saved: 06_training_efficiency.png")


# ============================================================
# 图 7：总体测试指标汇总（主模型）
# ============================================================
if primary_model and primary_model["test_results"]:
    tm = primary_model["test_results"]["metrics"]

    metric_names = ["Overall\nAccuracy", "Macro\nPrecision", "Macro\nRecall", "Macro\nF1",
                    "Low SNR\n(-4~2dB)", "High SNR\n(4~20dB)"]
    metric_vals = [
        tm["accuracy"] * 100,
        tm["macro_precision"] * 100,
        tm["macro_recall"] * 100,
        tm["macro_f1"] * 100,
        tm.get("low_snr_accuracy_-4_to_2_db", 0) * 100,
        tm.get("high_snr_accuracy_4_to_20_db", 0) * 100,
    ]
    colors_m = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(metric_names, metric_vals, color=colors_m, width=0.55, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.text(0.98, 0.97,
            f"Inference: {tm.get('inference_ms_per_sample', 0):.3f} ms/sample\n"
            f"Throughput: {tm.get('throughput_samples_per_second', 0):.0f} samples/s\n"
            f"Test Samples: {tm.get('samples', 0):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax.set_ylabel("Score (%)")
    ax.set_title(f"{primary_model['display']} Overall Test Metrics Summary")
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_overall_metrics.png")
    plt.close(fig)
    print("Saved: 07_overall_metrics.png")


# ============================================================
# 图 8：过拟合分析（所有模型对比）
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 训练-验证准确率差距
for m in models:
    ep = col(m["hist"], "epoch").astype(int)
    gap = (col(m["hist"], "train_accuracy") - col(m["hist"], "validation_accuracy")) * 100
    ax1.plot(ep, gap, "o-", color=m["color"], markersize=4, label=f"{m['display']} Gap")
ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy Gap (%)")
ax1.set_title("Overfitting Gap: Train Acc - Val Acc")
ax1.legend()

# 验证损失稳定性
for m in models:
    ep = col(m["hist"], "epoch").astype(int)
    ax2.plot(ep, col(m["hist"], "validation_loss"), "o-", color=m["color"], markersize=4, label=f"{m['display']} Val Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Loss")
ax2.set_title("Validation Loss Stability Comparison")
ax2.legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "08_overfitting_analysis.png")
plt.close(fig)
print("Saved: 08_overfitting_analysis.png")


# ============================================================
# 图 9：模型复杂度与性能对比表
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

# 论文对比数据（Table 5）
PAPER_DATA = {
    "TDMRNet": {"params": "46.6k", "flops": "70.8", "inference": "0.103", "train_time": "266.6", "acc": "93.80"},
    "DenseNet": {"params": "143.4k", "flops": "382.4", "inference": "0.493", "train_time": "1716.8", "acc": "99.77"},
    "ResNet": {"params": "175.9k", "flops": "240.4", "inference": "0.105", "train_time": "241.2", "acc": "99.88"},
    "Inception": {"params": "1253.6k", "flops": "890.8", "inference": "0.165", "train_time": "443.3", "acc": "88.74"},
}

table_data = [
    ["Model", "Params", "FLOPs\n(MFLOPs)", "Inference\n(ms)", "Train Time\n(s/epoch)", "Test Acc\n(%)"]
]

for m in models:
    name = m["display"]
    paper = PAPER_DATA.get(name, PAPER_DATA.get(m["name"]))
    avg_time = f"{np.mean(col(m['hist'], 'epoch_seconds')):.1f}" if m["hist"] else "N/A"
    if m["test_results"]:
        tm = m["test_results"]["metrics"]
        acc = f"{tm['accuracy'] * 100:.2f}"
        inf = f"{tm.get('inference_ms_per_sample', 0):.3f}"
    else:
        acc = "N/A"
        inf = "N/A"

    table_data.append([
        f"{name}\n(Reproduction)", paper["params"] if paper else "N/A",
        paper["flops"] if paper else "N/A", inf, avg_time, acc
    ])
    if paper:
        table_data.append([
            f"{name}\n(Paper)", paper["params"], paper["flops"],
            paper["inference"], paper["train_time"], paper["acc"]
        ])

# 补充论文中有但本项目未训练的模型
for name, paper in PAPER_DATA.items():
    if not any(m["display"] == name for m in models):
        table_data.append([
            f"{name}\n(Paper)", paper["params"], paper["flops"],
            paper["inference"], paper["train_time"], paper["acc"]
        ])

table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.0)

# 表头样式
for j in range(6):
    table[0, j].set_facecolor("#4472C4")
    table[0, j].set_text_props(color="white", fontweight="bold")

# 高亮复现行
for i in range(1, len(table_data)):
    if "Reproduction" in table_data[i][0]:
        for j in range(6):
            table[i, j].set_facecolor("#D6E4F0")

ax.set_title("Model Complexity & Performance Comparison", fontsize=13, pad=15)
fig.tight_layout()
fig.savefig(FIG_DIR / "09_model_comparison_table.png")
plt.close(fig)
print("Saved: 09_model_comparison_table.png")


# ── 总结 ───────────────────────────────────────────────────
print(f"\n所有图表已保存到：{FIG_DIR}")
print(f"共生成 9 张图表")
