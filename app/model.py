"""模型加载与运行信息（用 lru_cache 保证只读一次盘）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from app.config import get_settings


@lru_cache(maxsize=1)
def get_model():
    """加载模型，文件缺失时立即抛错（启动即失败，好过带病服务返回垃圾预测）。"""
    model_path = get_settings().model_path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行 ml/train.py 导出模型")
    return joblib.load(model_path)


def get_model_info() -> dict:
    """返回当前服务的模型版本与运行环境（用于 /model-info 与响应字段）。

    统一从 app.config.get_settings() 读取，避免「这里一个默认值、那里一个默认值」。
    """
    settings = get_settings()
    return {
        "model_version": settings.model_version,
        "app_env": settings.app_env,
        "model_path": settings.model_path,
    }
