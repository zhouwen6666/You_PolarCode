"""集中注册模型，使训练和测试代码无需依赖具体网络实现。"""

from collections.abc import Callable

from torch import nn

from .baselines import DenseNetBaseline, InceptionBaseline, ResNetBaseline
from .tdmrnet import TDMRNet
from .tdmrnet_plus import TDMRNetPlus
from .tdmrnet_ca import TDMRNetCA
from .tdmrnet_wht import TDMRNetWHT
from .polar_mod import PolarMod


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    # 原始 TDMRNet 与论文 baseline
    "tdmrnet": TDMRNet,
    "densenet": DenseNetBaseline,
    "inception": InceptionBaseline,
    "resnet": ResNetBaseline,
    # TDMRNet 增强变体
    "tdmrnet_plus": TDMRNetPlus,        # 容量扩展 + SE + WHT
    "tdmrnet_ca": TDMRNetCA,            # 容量扩展 + SE（无 WHT）
    "tdmrnet_wht": TDMRNetWHT,          # 原容量 + WHT
    "polar_mod": PolarMod,              # 调制动态通道选择 (EfficientMod)
}


def available_models() -> tuple[str, ...]:
    """返回所有可通过命令行选择的模型名称。"""

    return tuple(MODEL_REGISTRY)


def build_model(model_name: str, num_classes: int = 18, channels: int = 16, num_dcms: int = 2) -> nn.Module:
    """根据注册名称构造模型，并传入统一的公共超参数。"""

    try:
        constructor = MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(f"未知模型 {model_name!r}，可选模型：{', '.join(available_models())}") from error
    return constructor(num_classes=num_classes, channels=channels, num_dcms=num_dcms)


def count_trainable_parameters(model: nn.Module) -> int:
    """统计任意注册模型中需要梯度更新的参数数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
