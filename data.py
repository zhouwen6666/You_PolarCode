"""从 train 目录读取 LLR，并转换为 TDMRNet 的二维输入。"""

from __future__ import annotations

import bisect
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_PATTERN = re.compile(
    r"^P(?P<label>\d{1,2})_N(?P<n>\d+)_K(?P<k>\d+)_snr(?P<snr>[+-]\d+)\.npy$"
)


def llr_to_tdmr_matrix(llr: np.ndarray, matrix_rows: int = 256) -> np.ndarray:
    """把一维 LLR 按行优先重排为 [1, matrix_rows, cols] 单通道二维矩阵。"""

    # NPY 内存映射是只读的；显式复制可确保交给 PyTorch 的张量拥有可写内存。
    llr = np.array(llr, dtype=np.float32, copy=True)
    if llr.ndim != 1:
        raise ValueError(f"LLR 必须是一维序列，实际形状为 {llr.shape}。")
    if llr.size % matrix_rows != 0:
        raise ValueError(f"LLR 长度 {llr.size} 不能被矩阵行数 {matrix_rows} 整除。")
    return np.ascontiguousarray(llr.reshape(matrix_rows, -1)[None, ...])


def discover_train_shards(train_dir: Path) -> list[Path]:
    """发现并校验 train 目录中的全部类别-SNR NPY 分片。"""

    files = sorted(train_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"train 目录中没有找到 NPY 文件：{train_dir}")
    invalid = [path.name for path in files if LABEL_PATTERN.match(path.name) is None]
    if invalid:
        raise ValueError(f"以下训练文件名无法解析标签：{invalid[:5]}")
    return files


class NpyShardDataset(Dataset):
    """以内存映射方式惰性读取所有训练分片，避免一次载入完整数据集。"""

    def __init__(self, train_dir: Path, sample_length: int = 8192, matrix_rows: int = 256):
        """扫描训练分片并建立全局样本索引。"""

        self.files = discover_train_shards(Path(train_dir))
        self.sample_length = sample_length
        self.matrix_rows = matrix_rows
        self._arrays: dict[int, np.ndarray] = {}
        self._ends: list[int] = []
        self._labels: list[int] = []
        self._snrs: list[int] = []

        total = 0
        for path in self.files:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or array.shape[1] != sample_length:
                raise ValueError(f"{path.name} 形状应为 [样本数, {sample_length}]，实际为 {array.shape}。")
            match = LABEL_PATTERN.match(path.name)
            assert match is not None
            label = int(match.group("label"))
            if label < 1:
                raise ValueError(f"{path.name} 的类别编号必须从 P01/P1 开始。")
            self._labels.append(label - 1)
            self._snrs.append(int(match.group("snr")))
            total += array.shape[0]
            self._ends.append(total)

    def __len__(self) -> int:
        """返回 train 目录中的 LLR 样本总数。"""

        return self._ends[-1]

    def _get_array(self, shard_index: int) -> np.ndarray:
        """按需打开并缓存一个只读 NPY 内存映射。"""

        if shard_index not in self._arrays:
            self._arrays[shard_index] = np.load(self.files[shard_index], mmap_mode="r")
        return self._arrays[shard_index]

    def metadata_at(self, index: int) -> tuple[int, int]:
        """返回全局样本索引对应的零起始类别和 SNR，用于分组评估。"""

        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        return self._labels[shard_index], self._snrs[shard_index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读取一条 LLR、重排为二维矩阵，并返回零起始类别标签。"""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        llr = self._get_array(shard_index)[index - shard_start]
        matrix = llr_to_tdmr_matrix(llr, self.matrix_rows)
        return torch.from_numpy(matrix), torch.tensor(self._labels[shard_index], dtype=torch.long)
