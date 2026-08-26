"""PolarMod 消融实验汇总脚本。

PolarMod 两阶段微调的消融对比：
  A. PolarMod (CE)              — 标准交叉熵基线
  B. PolarMod_FocalG3           — Focal Loss γ=3 微调
  C. PolarMod_WeightedSqrt      — 加权交叉熵 (sqrt_inverse_acc) 微调

比较 A→B: Focal Loss 的贡献
比较 A→C: 加权交叉熵的贡献
比较 B vs C: 两种难类增强策略的对比
"""

import argparse
import json
from pathlib import Path

import torch

from models import build_model, count_trainable_parameters


def get_metrics(ckpt_path):
    """读取 checkpoint 对应的测试结果。"""
    metrics_path = ckpt_path.parent / "test_results" / "test_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def run_ablation():
    """读取已有评估结果并生成消融汇总。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    variants = [
        ("A_polar_mod", "polar_mod",
         Path("outputs/polar_mod/polar_mod_best.pt"),
         "PolarMod 基线（CrossEntropyLoss）"),
        ("B_focal_g3", "polar_mod",
         Path("outputs/polar_mod_focal_g3/polar_mod_best.pt"),
         "PolarMod + Focal Loss γ=3 微调"),
        ("C_weighted_sqrt", "polar_mod",
         Path("outputs/polar_mod_weighted_sqrt/polar_mod_best.pt"),
         "PolarMod + Weighted CE (sqrt_inverse_acc) 微调"),
    ]

    results = {}

    for label, model_key, ckpt_path, desc in variants:
        print(f"\n{'='*60}")
        print(f"变体 {label}: {desc}")
        print(f"{'='*60}")

        if not ckpt_path.exists():
            print(f"  checkpoint 不存在: {ckpt_path}")
            if label == "A_polar_mod":
                print(f"  请先训练基线: python train.py --model polar_mod --early-stop-metric acc")
            continue

        metrics = get_metrics(ckpt_path)
        if metrics is None:
            print(f"  评估结果不存在: {ckpt_path.parent}/test_results/test_metrics.json")
            if label == "A_polar_mod":
                print(f"  请先评估: python evaluate.py --model polar_mod --checkpoint {ckpt_path}")
            continue

        model = build_model(model_key)
        params = count_trainable_parameters(model)
        del model

        results[label] = {
            "model": model_key,
            "params": params,
            "samples": metrics.get("samples", 0),
            "accuracy": metrics.get("accuracy", 0),
            "macro_precision": metrics.get("macro_precision", 0),
            "macro_recall": metrics.get("macro_recall", 0),
            "macro_f1": metrics.get("macro_f1", 0),
            "low_snr_accuracy_-4_to_2_db": metrics.get("low_snr_accuracy_-4_to_2_db", 0),
            "high_snr_accuracy_4_to_20_db": metrics.get("high_snr_accuracy_4_to_20_db", 0),
            "inference_ms_per_sample": metrics.get("inference_ms_per_sample", 0),
        }
        print(f"  参数量={params:,}，准确率={metrics['accuracy']*100:.2f}%，Macro-F1={metrics['macro_f1']*100:.2f}%")

    # 打印汇总
    print(f"\n{'='*80}")
    print("PolarMod 消融实验结果汇总")
    print(f"{'='*80}")
    print(f"{'变体':<25} {'参数量':>10} {'准确率':>8} {'Macro-F1':>8} {'低SNR':>8} {'高SNR':>8}")
    print("-" * 80)
    for label in ["A_polar_mod", "B_focal_g3", "C_weighted_sqrt"]:
        if label in results:
            r = results[label]
            print(f"{label:<25} {r['params']:>10,} {r['accuracy']*100:>7.2f}% {r['macro_f1']*100:>7.2f}% "
                  f"{r['low_snr_accuracy_-4_to_2_db']*100:>7.2f}% {r['high_snr_accuracy_4_to_20_db']*100:>7.2f}%")

    # 贡献分解
    if all(k in results for k in ["A_polar_mod", "B_focal_g3", "C_weighted_sqrt"]):
        a = results["A_polar_mod"]["accuracy"]
        b = results["B_focal_g3"]["accuracy"]
        c = results["C_weighted_sqrt"]["accuracy"]
        print(f"\n{'='*80}")
        print("贡献分解")
        print(f"{'='*80}")
        print(f"基线 (A) PolarMod CE:           {a*100:.2f}%")
        print(f"+ Focal γ=3 (B):                 {b*100:.2f}%  (Δ = {(b-a)*100:+.2f}%)")
        print(f"+ Weighted CE sqrt (C):          {c*100:.2f}%  (Δ = {(c-a)*100:+.2f}%)")
        print(f"\nFocal γ=3 贡献:       {(b-a)*100:+.2f}%")
        print(f"Weighted CE 贡献:     {(c-a)*100:+.2f}%")
        print(f"Focal vs Weighted:    {(b-c)*100:+.2f}%")
    elif "B_focal_g3" in results and "C_weighted_sqrt" in results:
        b = results["B_focal_g3"]["accuracy"]
        c = results["C_weighted_sqrt"]["accuracy"]
        print(f"\n{'='*80}")
        print("对比（基线 A 缺失，仅 B vs C）")
        print(f"{'='*80}")
        print(f"Focal γ=3 (B):           {b*100:.2f}%")
        print(f"Weighted CE sqrt (C):    {c*100:.2f}%")
        print(f"Focal vs Weighted:       {(b-c)*100:+.2f}%")

    # 保存
    output_path = Path("outputs/ablation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolarMod 消融实验汇总")
    args = parser.parse_args()
    run_ablation()
