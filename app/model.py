"""模型加载与运行信息（用 lru_cache 保证只读一次盘）。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "linear_regression.pkl"


@lru_cache(maxsize=1)
def get_model():
    """加载模型，文件缺失时立即抛错（启动即失败，好过带病服务返回垃圾预测）。"""
    model_path = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}，请先运行 ml/train.py 导出模型"
        )
    return joblib.load(model_path)


def get_model_info() -> dict:
    """返回当前服务的模型版本与运行环境（用于 /model-info 与响应字段）。"""
    return {
        "model_version": os.getenv("MODEL_VERSION", "local-dev"),
        "app_env": os.getenv("APP_ENV", "development"),
    }