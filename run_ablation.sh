#!/bin/bash
# ============================================================
# You_PolarCode PolarMod 消融实验 + 显著性检验 一键脚本
#
# 消融变体（PolarMod 损失函数）：
#   A. PolarMod (CE)           — 标准交叉熵基线（需重训）
#   B. PolarMod_FocalG3        — Focal Loss γ=3 微调（已有）
#   C. PolarMod_WeightedSqrt   — 加权交叉熵 (sqrt_inverse_acc) 微调（已有）
#
# 显著性检验：PolarMod vs TDMRNet / DenseNet / Inception / ResNet
#
# 用法:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh                     # 全流程
#   ./run_ablation.sh --skip-ablation     # 跳过消融训练，只做显著性检验
#   ./run_ablation.sh --skip-bootstrap    # 跳过显著性检验
# ============================================================

set -u

PYTHON=${PYTHON:-python}
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 参数解析
SKIP_ABLATION=false
SKIP_BOOTSTRAP=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ablation) SKIP_ABLATION=true; shift ;;
        --skip-bootstrap) SKIP_BOOTSTRAP=true; shift ;;
        --python) PYTHON="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

log()  { echo -e "\033[32m[$(date '+%m-%d %H:%M:%S')]\033[0m $*"; }
warn() { echo -e "\033[33m[$(date '+%m-%d %H:%M:%S')]\033[0m $*"; }
err()  { echo -e "\033[31m[$(date '+%m-%d %H:%M:%S')] ERROR:\033[0m $*"; }

# ── GPU 检查 ──────────────────────────────────────────
log "============================================"
log "You_PolarCode PolarMod 消融实验 + 显著性检验"
log "Python: $($PYTHON --version 2>&1)"
if $PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    log "GPU: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null)"
else
    warn "PyTorch 未安装或无 CUDA"
fi
log "============================================"

# ── 工具函数：确保模型有评估结果 ──────────────────────
ensure_eval() {
    local model="$1"
    local ckpt="$2"
    local results_dir="$3"
    local metrics="$results_dir/test_metrics.json"

    if [ -f "$metrics" ]; then
        log "-- $model 已有评估结果，跳过"
        return 0
    fi

    if [ ! -f "$ckpt" ]; then
        warn "缺少 checkpoint: $ckpt"
        return 1
    fi

    log ">> 评估 $model ..."
    if $PYTHON evaluate.py --model "$model" --checkpoint "$ckpt" \
            --output-dir "$results_dir"; then
        log "<< $model 评估完成"
    else
        err "$model 评估失败"
        return 1
    fi
}

# ============================================================
# 阶段 1: PolarMod 消融实验
# ============================================================
if [ "$SKIP_ABLATION" = false ]; then
    log "========== 阶段 1: PolarMod 消融实验 =========="

    # A. PolarMod CE 基线（checkpoint 丢失，需重训）
    POLAR_CKPT="outputs/polar_mod/polar_mod_best.pt"
    POLAR_METRICS="outputs/polar_mod/test_results/test_metrics.json"

    if [ -f "$POLAR_METRICS" ]; then
        log "-- PolarMod CE 基线已有评估结果，跳过"
    elif [ -f "$POLAR_CKPT" ]; then
        log "-- PolarMod CE 基线已有 checkpoint，跳过训练"
    else
        log ">> 训练 PolarMod CE 基线 ..."
        $PYTHON train.py --model polar_mod --early-stop-metric acc
        log "<< PolarMod CE 基线训练完成"
    fi

    # 评估 CE 基线
    ensure_eval "polar_mod" "$POLAR_CKPT" "outputs/polar_mod/test_results"

    # B. PolarMod FocalG3（已有 checkpoint，确保有评估结果）
    log "========== 变体 B: PolarMod FocalG3 =========="
    FOCAL_CKPT="outputs/polar_mod_focal_g3/polar_mod_best.pt"
    ensure_eval "polar_mod" "$FOCAL_CKPT" "outputs/polar_mod_focal_g3/test_results"

    # C. PolarMod WeightedSqrt（已有 checkpoint，确保有评估结果）
    log "========== 变体 C: PolarMod WeightedSqrt =========="
    WEIGHTED_CKPT="outputs/polar_mod_weighted_sqrt/polar_mod_best.pt"
    ensure_eval "polar_mod" "$WEIGHTED_CKPT" "outputs/polar_mod_weighted_sqrt/test_results"

    # 生成消融汇总
    log ">> 生成消融汇总 ..."
    $PYTHON ablation_study.py || warn "消融汇总失败"
else
    log "-- 跳过消融实验（--skip-ablation）"
fi

# ============================================================
# 阶段 2: 显著性检验
# ============================================================
if [ "$SKIP_BOOTSTRAP" = false ]; then
    log "========== 阶段 2: 显著性检验 =========="

    POLAR_CKPT="outputs/polar_mod/polar_mod_best.pt"
    TDMR_CKPT="outputs/tdmrnet_best.pt"
    DENSE_CKPT="outputs/densenet/densenet_best.pt"
    INCEP_CKPT="outputs/inception/inception_best.pt"
    RES_CKPT="outputs/resnet/resnet_best.pt"

    run_bootstrap() {
        local name_a="$1" ckpt_a="$2" name_b="$3" ckpt_b="$4" output="$5"
        if [ ! -f "$ckpt_a" ] || [ ! -f "$ckpt_b" ]; then
            warn "缺少 checkpoint，跳过 $name_a vs $name_b"
            return 0
        fi
        if [ -f "$output" ]; then
            log "-- $name_a vs $name_b 已有结果，跳过"
            return 0
        fi
        log ">> Bootstrap: $name_a vs $name_b ..."
        $PYTHON bootstrap_test.py \
            --model-a "$name_a" --ckpt-a "$ckpt_a" \
            --model-b "$name_b" --ckpt-b "$ckpt_b" \
            --output "$output"
    }

    # PolarMod vs 各基线
    run_bootstrap "polar_mod" "$POLAR_CKPT" "tdmrnet" "$TDMR_CKPT" \
        "outputs/bootstrap_polarmod_vs_tdmrnet.json"
    run_bootstrap "polar_mod" "$POLAR_CKPT" "resnet" "$RES_CKPT" \
        "outputs/bootstrap_polarmod_vs_resnet.json"
    run_bootstrap "polar_mod" "$POLAR_CKPT" "densenet" "$DENSE_CKPT" \
        "outputs/bootstrap_polarmod_vs_densenet.json"
    run_bootstrap "polar_mod" "$POLAR_CKPT" "inception" "$INCEP_CKPT" \
        "outputs/bootstrap_polarmod_vs_inception.json"
else
    log "-- 跳过显著性检验（--skip-bootstrap）"
fi

# ============================================================
# 最终输出
# ============================================================
log "============================================"
log "全部流程完成！"
log "============================================"
log "输出目录:"
log "  消融汇总:     outputs/ablation_results.json"
log "  显著性检验:   outputs/bootstrap_*.json"
log "  模型权重:     outputs/<model>/<model>_best.pt"
log "============================================"
