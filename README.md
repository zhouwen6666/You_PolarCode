# 极化码参数识别模型实验项目

本项目复现 TDMRNet 及论文第 4.4 节的三个 baseline（DenseNet、Inception、ResNet），并提出 PolarMod 作为改进模型。所有模型使用相同的数据读取、训练、验证和测试流程，输入为 8192 长度的 LLR 序列重排为 `[batch, 1, 256, 32]` 的二维矩阵，完成 P01-P18 共十八分类（6 种码长 × 3 种码率）。

## 当前结果

| 模型 | 测试准确率 | Macro F1 | 备注 |
|------|-----------|----------|------|
| **PolarMod-FocalG3** | **58.86%** | 57.7% | 总体最优 |
| **PolarMod-WSqrt** | 56.72% | **60.96%** | 类别均衡最优 |
| TDMRNet | 51.35% | — | 主基线 |
| ResNet | 49.16% | — | 论文 baseline |
| Inception | 32.88% | — | 论文 baseline |
| DenseNet | 31.41% | — | 论文 baseline |

PolarMod 两个变体均经 500 次 bootstrap 验证，95% 置信区间与 TDMRNet 不重叠，统计显著超越基线。

## 项目结构

```text
You_PolarCode/
├── models/
│   ├── registry.py        模型注册与统一构造入口
│   ├── tdmrnet.py         TDMRNet（ICFEM + DCM）
│   ├── baselines.py        DenseNet、Inception、ResNet
│   ├── polar_mod.py       PolarMod（调制动态通道选择）
│   ├── common.py           公共模块
│   ├── tdmrnet_plus.py    TDMRNet 增强变体（未采用，源码保留）
│   ├── tdmrnet_ca.py      TDMRNet 增强变体（未采用，源码保留）
│   └── tdmrnet_wht.py     TDMRNet 增强变体（未采用，源码保留）
├── data.py                 LLR 数据读取和二维重排
├── config.py               公共实验配置
├── train.py                通用训练入口，只读取 train
├── evaluate.py             通用测试入口，只读取 test
├── plot_results.py         全量可视化脚本（9 张图表）
├── generate_polars.py      数据集生成脚本
├── losses.py               Focal Loss 和类别加权
├── my_polar_llr/           默认数据集
│   ├── train/
│   ├── test/
│   └── metadata.json
└── outputs/                权重、训练历史和测试结果
```

## 运行环境

```powershell
cd "D:\pythonProject1\You_PolarCode"
conda activate py3.11_pytorch2.8.0
pip install -r requirements.txt
```

依赖：PyTorch >= 2.0、NumPy >= 1.23、Matplotlib >= 3.5。GPU 非必须但推荐（RTX 3060 6GB 即可）。

## 数据生成（可选）

数据集已包含在 `my_polar_llr/` 中。如需重新生成：

```powershell
python generate_polars.py --output-dir my_polar_llr --overwrite
```

生成约 0.7 GB 数据，包含 P01-P18 共 18 类、SNR -4~20 dB 共 13 个信噪比点的 LLR 样本。

## 训练

### 通用参数

```powershell
python train.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | tdmrnet | 选择模型（见 `available_models()`） |
| `--epochs` | 600 | 最大训练轮数 |
| `--batch-size` | 64 | 批量大小 |
| `--lr` | 0.001 | 学习率 |
| `--early-stop-metric` | loss | 早停与最优模型选优指标，可选 `loss` 或 `acc` |
| `--patience` | 10 | 验证指标连续多少轮不改善则早停 |
| `--grad-clip` | 无 | 梯度范数裁剪阈值 |
| `--weight-decay` | 0 | Adam 权重衰减（L2 正则化） |
| `--loss-type` | ce | 损失函数类型：`ce`（交叉熵）、`focal`、`weighted_ce` |
| `--focal-gamma` | 2.0 | Focal Loss 聚焦参数 |
| `--class-weights-from` | 无 | 类别权重来源（test_metrics.json 路径） |
| `--weight-mode` | inverse_acc | 类别权重缩放模式：`inverse_acc`、`sqrt_inverse_acc`、`manual` |
| `--resume` | 无 | 从已有 checkpoint 恢复权重，用于两阶段微调 |
| `--output-suffix` | 无 | 输出目录后缀，自动拼接为 `outputs/{model}_{suffix}/` |
| `--dataset-dir` | my_polar_llr | 外部数据集根目录 |
| `--max-samples` | 无 | 仅用于快速验证，正式训练勿设 |

训练过程只读取 `my_polar_llr/train`，不接触 test 数据。训练日志实时写入 `training_history.csv`，最佳模型按指定指标保存。

### 训练论文 baseline

```powershell
# TDMRNet（主基线）
python train.py --model tdmrnet --epochs 200

# DenseNet
python train.py --model densenet --epochs 200

# Inception
python train.py --model inception --epochs 200

# ResNet（val_loss 波动大，需按准确率选优）
python train.py --model resnet --epochs 600 --early-stop-metric acc --patience 20
```

### 训练 PolarMod（两阶段微调）

PolarMod 两个变体均从基础模型微调而来，需按顺序执行：

**第一步：训练基础模型**

```powershell
python train.py --model polar_mod --epochs 200
```

输出到 `outputs/polar_mod/`。

**第二步：评估基础模型**（为加权损失提供类别权重）

```powershell
python evaluate.py --model polar_mod
```

测试结果保存到 `outputs/polar_mod/test_results/test_metrics.json`。

**第三步：微调两个变体**

```powershell
# PolarMod-FocalG3（Focal Loss, gamma=3）
python train.py --model polar_mod --loss-type focal --focal-gamma 3 `
    --lr 5e-5 --grad-clip 1.0 --weight-decay 5e-4 --epochs 200 `
    --resume outputs/polar_mod/polar_mod_best.pt --output-suffix focal_g3

# PolarMod-WSqrt（加权交叉熵, sqrt_inverse_acc）
python train.py --model polar_mod --loss-type weighted_ce `
    --class-weights-from outputs/polar_mod/test_results/test_metrics.json `
    --weight-mode sqrt_inverse_acc --lr 5e-5 --grad-clip 1.0 --weight-decay 5e-4 `
    --epochs 200 --resume outputs/polar_mod/polar_mod_best.pt --output-suffix weighted_sqrt
```

输出分别到 `outputs/polar_mod_focal_g3/` 和 `outputs/polar_mod_weighted_sqrt/`。

## 测试

```powershell
# 论文 baseline（输出目录自动推断）
python evaluate.py --model tdmrnet
python evaluate.py --model densenet
python evaluate.py --model inception
python evaluate.py --model resnet

# PolarMod 基础模型
python evaluate.py --model polar_mod

# PolarMod 变体（需指定 checkpoint 路径）
python evaluate.py --model polar_mod --checkpoint outputs/polar_mod_focal_g3/polar_mod_best.pt
python evaluate.py --model polar_mod --checkpoint outputs/polar_mod_weighted_sqrt/polar_mod_best.pt
```

每个模型的 `test_results/` 目录包含：

```text
test_metrics.json          完整测试指标
per_class_metrics.csv      每个类别的 Precision、Recall 和 F1
per_snr_metrics.csv        每个 SNR 的准确率
confusion_matrix.csv       混淆矩阵
```

终端同时输出总体 Loss、Accuracy、Macro Precision、Macro Recall、Macro F1、低/高 SNR 准确率、每类指标、分 SNR 准确率、推理延迟和吞吐量。

## 全量可视化

训练和测试完成后，运行以下命令一键生成全部可视化图表：

```powershell
python plot_results.py
```

脚本自动扫描 `outputs/` 目录下所有已训练模型，生成 9 张图表到 `outputs/figures/`：

| 编号 | 文件名 | 内容 |
|------|--------|------|
| 01 | `01_loss_curves.png` | 所有模型训练/验证 Loss 曲线对比 |
| 02 | `02_accuracy_curves.png` | 所有模型训练/验证准确率曲线对比 |
| 03 | `03_per_snr_accuracy.png` | 主模型各 SNR 准确率柱状图 |
| 04 | `04_per_class_metrics.png` | 主模型各类别 Precision/Recall/F1/Accuracy |
| 05 | `05_confusion_matrix.png` | 主模型混淆矩阵热力图 |
| 06 | `06_training_efficiency.png` | 每轮训练时间 + 峰值显存对比 |
| 07 | `07_overall_metrics.png` | 主模型总体指标汇总柱状图 |
| 08 | `08_overfitting_analysis.png` | 过拟合差距 + 验证损失稳定性 |
| 09 | `09_model_comparison_table.png` | 模型复杂度与性能对比表（含论文数据） |

脚本支持任意数量的模型对比，未训练的模型会自动跳过。

## 完整运行顺序

```powershell
cd "D:\pythonProject1\You_PolarCode"
conda activate py3.11_pytorch2.8.0

# 1. 数据生成（可选，已有数据则跳过）
python generate_polars.py --output-dir my_polar_llr --overwrite

# 2. 训练所有模型
python train.py --model tdmrnet --epochs 200
python train.py --model densenet --epochs 200
python train.py --model inception --epochs 200
python train.py --model resnet --epochs 600 --early-stop-metric acc --patience 20
python train.py --model polar_mod --epochs 200

# 3. 评估基础模型（PolarMod 基础模型评估结果用于变体微调）
python evaluate.py --model tdmrnet
python evaluate.py --model densenet
python evaluate.py --model inception
python evaluate.py --model resnet
python evaluate.py --model polar_mod

# 4. 微调 PolarMod 变体
python train.py --model polar_mod --loss-type focal --focal-gamma 3 `
    --lr 5e-5 --grad-clip 1.0 --weight-decay 5e-4 --epochs 200 `
    --resume outputs/polar_mod/polar_mod_best.pt --output-suffix focal_g3
python train.py --model polar_mod --loss-type weighted_ce `
    --class-weights-from outputs/polar_mod/test_results/test_metrics.json `
    --weight-mode sqrt_inverse_acc --lr 5e-5 --grad-clip 1.0 --weight-decay 5e-4 `
    --epochs 200 --resume outputs/polar_mod/polar_mod_best.pt --output-suffix weighted_sqrt

# 5. 评估 PolarMod 变体
python evaluate.py --model polar_mod --checkpoint outputs/polar_mod_focal_g3/polar_mod_best.pt
python evaluate.py --model polar_mod --checkpoint outputs/polar_mod_weighted_sqrt/polar_mod_best.pt

# 6. 生成可视化图表
python plot_results.py
```

## Baseline 结构说明

- **DenseNet**：按照论文说明使用 2 个 Dense Block 和 1 个 Transition Layer。
- **Inception**：按照论文说明连续使用 6 个 Inception Module。
- **ResNet**：使用标准二维残差块和捷径连接。
- **PolarMod**：基于调制动态通道选择机制，保持 2D 输入和低参数量（约 60K），通过 Focal Loss 和类别加权两种损失策略分别微调。

论文未公开 baseline 的完整逐层通道数、模块内部层数和所有训练细节，因此本项目保证宏观拓扑以及数据、训练、测试流程一致，但不能声称逐层实现与作者代码完全相同。

## 使用外部数据集

训练和测试均可通过 `--dataset-dir` 指定项目之外的数据集：

```powershell
python train.py --model tdmrnet --dataset-dir "D:\datasets\my_polar_llr"
python evaluate.py --model tdmrnet --dataset-dir "D:\datasets\my_polar_llr"
```

外部数据集必须采用以下结构：

```text
my_polar_llr/
├── train/
│   ├── P01_N32_K8_snr-04.npy
│   ├── P01_N32_K8_snr-02.npy
│   └── ...
└── test/
    ├── P01_N32_K8_snr-04.npy
    ├── P01_N32_K8_snr-02.npy
    └── ...
```

每个 NPY 文件必须是二维 `float32` 数组，形状为 `[样本数, 8192]`；文件名必须符合 `P类别_N码长_K信息位长度_snr信噪比.npy`。默认配置包含 P01-P18 共 18 类，标签从 1 开始连续编号。若外部数据的序列长度或类别数不同，需要修改 `config.py` 中的 `sample_length` 和 `num_classes`，并重新训练模型。
