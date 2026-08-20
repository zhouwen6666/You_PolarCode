"""生成约 0.7 GB 的纯 Polar 码 LLR 长序列训练集和测试集。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLE_LENGTH = 8192
SNRS_DB = tuple(range(-4, 22, 2))
CODE_LENGTHS = (32, 64, 128, 256, 512, 1024)
RATES = (0.25, 0.5, 0.75)
SAMPLES_PER_CONDITION = 90
TRAIN_SAMPLES = 72
TEST_SAMPLES = 18
SEED = 20260817
PW_BETA = 2.0**0.25


def polarization_weight_information_set(n: int, k: int) -> np.ndarray:
    """使用 Polarization Weight 方法选择最可靠的 K 个信息位位置，返回升序索引。"""

    if n <= 0 or n & (n - 1):
        raise ValueError("Polar 码长必须是2的幂。")
    if not 0 < k <= n:
        raise ValueError("信息位数量必须满足 0 < K <= N。")
    stages = int(math.log2(n))
    indices = np.arange(n, dtype=np.int64)
    weights = np.zeros(n, dtype=np.float64)
    for stage in range(stages):
        weights += ((indices >> stage) & 1) * (PW_BETA**stage)
    return np.sort(np.argsort(weights, kind="stable")[-k:]).astype(np.int64)


def polar_transform_rows(source: np.ndarray) -> np.ndarray:
    """对一批源向量执行 Arıkan GF(2) 蝶形编码变换，返回编码后的码字矩阵。"""

    codewords = np.array(source, dtype=np.uint8, copy=True)
    n = codewords.shape[1]
    step = 1
    while step < n:
        blocks = codewords.reshape(-1, 2 * step)
        blocks[:, :step] ^= blocks[:, step : 2 * step]
        step *= 2
    return codewords


def generate_clean_bit_streams(
    n: int,
    k: int,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """独立生成信息位、完成 Polar 编码并拼接成 [sample_count, 8192] 连续码流。"""

    if SAMPLE_LENGTH % n != 0:
        raise ValueError(f"样本长度 {SAMPLE_LENGTH} 必须能被码长 {n} 整除。")
    codewords_per_sample = SAMPLE_LENGTH // n
    total_codewords = sample_count * codewords_per_sample
    information_set = polarization_weight_information_set(n, k)
    source = np.zeros((total_codewords, n), dtype=np.uint8)
    source[:, information_set] = rng.integers(
        0, 2, size=(total_codewords, k), dtype=np.uint8
    )
    return polar_transform_rows(source).reshape(sample_count, SAMPLE_LENGTH)


def bits_to_llr(
    bit_streams: np.ndarray,
    rate: float,
    snr_db: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """执行 BPSK 调制、AWGN 信道和软解调，返回同形状 float32 LLR 序列。"""

    eb_n0 = 10.0 ** (float(snr_db) / 10.0)
    noise_variance = 1.0 / (2.0 * rate * eb_n0)
    symbols = 1.0 - 2.0 * bit_streams.astype(np.float32)
    noise = rng.normal(
        0.0, math.sqrt(noise_variance), size=bit_streams.shape
    ).astype(np.float32)
    received = symbols + noise
    return np.asarray(2.0 * received / noise_variance, dtype=np.float32)


def make_class_definitions() -> list[dict]:
    """建立“修改以及实现思路.docx”第四章规定的18个Polar类别。"""

    classes = []
    label_index = 0
    for n in CODE_LENGTHS:
        for rate in RATES:
            k = int(n * rate)
            classes.append(
                {
                    "label": f"P{label_index + 1:02d}",
                    "label_index": label_index,
                    "N": n,
                    "K": k,
                    "rate": rate,
                }
            )
            label_index += 1
    return classes


def ensure_output_is_safe(output_dir: Path, overwrite: bool) -> None:
    """防止在未明确指定时覆盖已经存在的训练集或测试集。"""

    existing = [output_dir / "train", output_dir / "test"]
    if not overwrite and any(path.exists() and any(path.iterdir()) for path in existing):
        raise FileExistsError(
            "输出目录已经包含数据。若确认重新生成并覆盖同名文件，请添加 --overwrite。"
        )


def generate_dataset(output_dir: Path, overwrite: bool = False) -> None:
    """生成全部类别和SNR条件，按 80/20 划分保存训练集、测试集及元数据。"""

    output_dir = output_dir.resolve()
    ensure_output_is_safe(output_dir, overwrite)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    classes = make_class_definitions()
    records = []
    total_conditions = len(classes) * len(SNRS_DB)
    completed = 0

    for class_definition in classes:
        n = class_definition["N"]
        k = class_definition["K"]
        rate = class_definition["rate"]
        label = class_definition["label"]
        for snr_db in SNRS_DB:
            condition_seed = (
                SEED
                + class_definition["label_index"] * 1_000_000
                + (snr_db + 4) * 10_000
            )
            rng = np.random.default_rng(condition_seed)
            bit_streams = generate_clean_bit_streams(
                n, k, SAMPLES_PER_CONDITION, rng
            )
            llr = bits_to_llr(bit_streams, rate, snr_db, rng)
            permutation = rng.permutation(SAMPLES_PER_CONDITION)
            train = np.ascontiguousarray(llr[permutation[:TRAIN_SAMPLES]])
            test = np.ascontiguousarray(llr[permutation[TRAIN_SAMPLES:]])
            filename = f"{label}_N{n}_K{k}_snr{snr_db:+03d}.npy"
            train_path = train_dir / filename
            test_path = test_dir / filename
            np.save(train_path, train, allow_pickle=False)
            np.save(test_path, test, allow_pickle=False)
            records.append(
                {
                    **class_definition,
                    "snr_db": snr_db,
                    "seed": condition_seed,
                    "train_file": str(train_path.relative_to(output_dir)),
                    "test_file": str(test_path.relative_to(output_dir)),
                    "train_samples": TRAIN_SAMPLES,
                    "test_samples": TEST_SAMPLES,
                }
            )
            completed += 1
            print(f"[{completed:03d}/{total_conditions}] {filename}", flush=True)

    metadata = {
        "dataset": "Polar LLR long-sequence dataset for parameter recognition",
        "source_requirements": "修改以及实现思路.docx, Chapter 4",
        "seed": SEED,
        "sample_length": SAMPLE_LENGTH,
        "samples_per_class_snr": SAMPLES_PER_CONDITION,
        "split": {
            "train": TRAIN_SAMPLES,
            "test": TEST_SAMPLES,
            "ratio": "80/20",
        },
        "classes": classes,
        "snrs_db": list(SNRS_DB),
        "modulation": {"type": "BPSK", "mapping": {"0": 1, "1": -1}},
        "channel": "AWGN",
        "snr_definition": "Eb/N0 in dB",
        "noise_variance": "1/(2*(K/N)*10**(EbN0_dB/10))",
        "llr_formula": "2*y/noise_variance",
        "dtype": "float32",
        "polar_construction": (
            "polarization weight, beta=2**0.25; "
            "largest K weights are information positions"
        ),
        "polar_transform": (
            "Arikan kernel transform over GF(2), no CRC, no rate matching"
        ),
        "records": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """解析输出目录和显式覆盖开关。"""

    parser = argparse.ArgumentParser(description="生成约0.7 GB的Polar LLR数据集。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="数据集根目录，默认是脚本所在的polars目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中已有的同名数据文件。",
    )
    return parser.parse_args()


def main() -> None:
    """根据固定复现实验参数生成数据集。"""

    args = parse_args()
    generate_dataset(args.output_dir, overwrite=args.overwrite)
    print(f"数据集生成完成：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
