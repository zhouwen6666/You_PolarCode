"""旧版导入兼容层；新代码应从 models 包导入。"""

from models import DCM, ICFEM, TDMRNet, available_models, build_model, count_trainable_parameters

__all__ = ["DCM", "ICFEM", "TDMRNet", "available_models", "build_model", "count_trainable_parameters"]
