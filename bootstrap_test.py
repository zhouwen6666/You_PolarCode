"""Bootstrap 显著性检验：比较两个模型在测试集上的准确率差异是否显著。

方法：500 次重采样 bootstrap，计算 95% 置信区间和 p-value。
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from config import DEFAULT_DATASET_DIR
from data import NpyShardDataset
from evaluate import load_model


def collect_predictions(checkpoint_path, dataset_dir, device, batch_size=64):
    """加载模型，在测试集上收集每个样本的预测结果和真实标签。"""
    model, checkpoint = load_model(Path(checkpoint_path), device)
    config = checkpoint.get("config", {})
    sample_length = int(config.get("sample_length", 8192))
    matrix_rows = int(config.get("matrix_rows", 256))
    num_classes = int(config.get("num_classes", 18))

    dataset = NpyShardDataset(Path(dataset_dir) / "test", sample_length=sample_length, matrix_rows=matrix_rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                         pin_memory=device.type == "cuda")

    all_preds, all_labels, all_snrs = [], [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            start_idx = batch_idx * batch_size
            snrs = [dataset.metadata_at(start_idx + i)[1] for i in range(labels.size(0))]
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_snrs.extend(snrs)
            if (batch_idx + 1) % 50 == 0:
                print(f"  推理进度: {(batch_idx+1)*batch_size}/{len(dataset)}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.array(all_preds), np.array(all_labels), np.array(all_snrs), num_classes


def bootstrap_accuracy(preds, labels, n_bootstrap=500, seed=42):
    """对样本级预测结果做 bootstrap 重采样。"""
    rng = np.random.RandomState(seed)
    n = len(labels)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        accs.append((preds[idx] == labels[idx]).mean())
    accs = np.array(accs)
    lower, upper = np.percentile(accs, [2.5, 97.5])
    return accs.mean(), accs.std(), lower, upper


def bootstrap_paired_test(preds_a, preds_b, labels, n_bootstrap=500, seed=42):
    """配对 bootstrap 检验。"""
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        acc_a = (preds_a[idx] == labels[idx]).mean()
        acc_b = (preds_b[idx] == labels[idx]).mean()
        diffs.append(acc_a - acc_b)
    diffs = np.array(diffs)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), diffs.std(), lower, upper, p_value


def per_snr_bootstrap(preds, labels, snrs, n_bootstrap=500, seed=42):
    """分 SNR 级别做 bootstrap。"""
    rng = np.random.RandomState(seed)
    results = {}
    for snr in sorted(set(snrs.tolist())):
        mask = snrs == snr
        if mask.sum() < 10:
            continue
        p, l = preds[mask], labels[mask]
        n = len(l)
        accs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            accs.append((p[idx] == l[idx]).mean())
        accs = np.array(accs)
        lower, upper = np.percentile(accs, [2.5, 97.5])
        results[snr] = {"mean": accs.mean(), "std": accs.std(),
                        "ci_lower": lower, "ci_upper": upper, "n": n}
    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 显著性检验")
    parser.add_argument("--model-a", required=True, help="模型 A 名称")
    parser.add_argument("--ckpt-a", required=True, help="模型 A checkpoint 路径")
    parser.add_argument("--model-b", required=True, help="模型 B 名称")
    parser.add_argument("--ckpt-b", required=True, help="模型 B checkpoint 路径")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/bootstrap_results.json")
    parser.add_argument("--batch-size", type=int, default=64, help="推理批大小，显存不足时调小")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print(f"\n加载 {args.model_a} ...")
    preds_a, labels, snrs, nc = collect_predictions(args.ckpt_a, args.dataset_dir, device, args.batch_size)
    acc_a = (preds_a == labels).mean()
    print(f"  {args.model_a} 测试准确率: {acc_a*100:.2f}% ({len(labels)} 样本)")

    print(f"\n加载 {args.model_b} ...")
    preds_b, _, _, _ = collect_predictions(args.ckpt_b, args.dataset_dir, device, args.batch_size)
    acc_b = (preds_b == labels).mean()
    print(f"  {args.model_b} 测试准确率: {acc_b*100:.2f}%")

    print(f"\n{'='*60}")
    print(f"Bootstrap {args.n_bootstrap} 次重采样 (seed={args.seed})")
    print(f"{'='*60}")

    m_a, s_a, lo_a, hi_a = bootstrap_accuracy(preds_a, labels, args.n_bootstrap, args.seed)
    print(f"\n{args.model_a}: {acc_a*100:.2f}% | CI: [{lo_a*100:.2f}%, {hi_a*100:.2f}%] | std={s_a*100:.3f}%")

    m_b, s_b, lo_b, hi_b = bootstrap_accuracy(preds_b, labels, args.n_bootstrap, args.seed)
    print(f"{args.model_b}: {acc_b*100:.2f}% | CI: [{lo_b*100:.2f}%, {hi_b*100:.2f}%] | std={s_b*100:.3f}%")

    diff_mean, diff_std, diff_lo, diff_hi, p_value = bootstrap_paired_test(
        preds_a, preds_b, labels, args.n_bootstrap, args.seed)
    print(f"\n{'='*60}")
    print(f"配对 Bootstrap 检验: {args.model_a} - {args.model_b}")
    print(f"{'='*60}")
    print(f"准确率差值: {diff_mean*100:.2f}% ± {diff_std*100:.3f}%")
    print(f"95% CI: [{diff_lo*100:.2f}%, {diff_hi*100:.2f}%]")
    print(f"p-value: {p_value:.4f}")
    if diff_lo > 0:
        print(f"结论: {args.model_a} 显著优于 {args.model_b} (CI 下界 > 0, p < 0.05)")
    elif diff_hi < 0:
        print(f"结论: {args.model_b} 显著优于 {args.model_a} (CI 上界 < 0, p < 0.05)")
    else:
        print(f"结论: 差异不显著 (CI 包含 0)")

    print(f"\n{'='*60}")
    print(f"分 SNR 准确率对比 (bootstrap 95% CI)")
    print(f"{'='*60}")
    snr_results_a = per_snr_bootstrap(preds_a, labels, snrs, args.n_bootstrap, args.seed)
    snr_results_b = per_snr_bootstrap(preds_b, labels, snrs, args.n_bootstrap, args.seed)
    print(f"{'SNR':>6}  {args.model_a:>20}  {args.model_b:>20}  {'差异':>8}")
    print("-" * 60)
    for snr in sorted(snr_results_a.keys()):
        ra = snr_results_a[snr]
        rb = snr_results_b.get(snr, {"mean": 0, "ci_lower": 0, "ci_upper": 0})
        diff = ra["mean"] - rb["mean"]
        print(f"{snr:>+4d}dB  {ra['mean']*100:>6.2f}% [{ra['ci_lower']*100:.2f},{ra['ci_upper']*100:.2f}]  "
              f"{rb['mean']*100:>6.2f}% [{rb['ci_lower']*100:.2f},{rb['ci_upper']*100:.2f}]  {diff*100:>+6.2f}%")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model_a": {"name": args.model_a, "accuracy": acc_a, "ci": [lo_a, hi_a], "std": s_a},
        "model_b": {"name": args.model_b, "accuracy": acc_b, "ci": [lo_b, hi_b], "std": s_b},
        "paired_test": {"mean_diff": diff_mean, "std_diff": diff_std,
                        "ci": [diff_lo, diff_hi], "p_value": p_value},
        "per_snr_a": {str(k): v for k, v in snr_results_a.items()},
        "per_snr_b": {str(k): v for k, v in snr_results_b.items()},
        "n_bootstrap": args.n_bootstrap,
        "n_samples": len(labels),
    }
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
