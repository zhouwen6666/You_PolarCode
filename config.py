"""TDMRNet 训练阶段的集中配置。"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "my_polar_llr"


@dataclass(frozen=True)
class TrainConfig:
    """保存论文复现实验所需的模型、数据和训练超参数。"""

    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: Path = PROJECT_DIR / "outputs"
    model_name: str = "tdmrnet"
    matrix_rows: int = 256
    sample_length: int = 8192
    num_classes: int = 18
    channels: int = 16
    num_dcms: int = 2
    batch_size: int = 64
    learning_rate: float = 1e-3
    grad_clip: float | None = None
    weight_decay: float = 0.0
    max_epochs: int = 600
    patience: int = 10
    validation_ratio: float = 0.1
    seed: int = 20260816
    num_workers: int = 0
    # 损失函数与两阶段微调
    loss_type: str = "ce"  # ce | focal | weighted_ce
    focal_gamma: float = 2.0
    # 早停与最优模型选优指标：loss 或 acc
    early_stop_metric: str = "loss"  # loss | acc
    resume_checkpoint: Path | None = None
    class_weights_path: Path | None = None
    class_weight_mode: str = "inverse_acc"  # inverse_acc | sqrt_inverse_acc

    @property
    def matrix_columns(self) -> int:
        """根据样本长度和论文选定的 256 行计算二维输入列数。"""

        if self.sample_length % self.matrix_rows != 0:
            raise ValueError("sample_length 必须能被 matrix_rows 整除。")
        return self.sample_length // self.matrix_rows
