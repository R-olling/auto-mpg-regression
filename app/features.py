from __future__ import annotations

import pandas as pd

# 必须与 ml/data_pipeline.py 的 V2_FEATURE_COLUMNS 完全一致
FEATURE_COLUMNS = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin_1",
    "origin_2",
    "origin_3",
]


def payload_to_features(payload: dict) -> list[float]:
    """把 PredictionRequest.model_dump() 转成按 FEATURE_COLUMNS 顺序的特征向量。

    两个映射：
    1. model_year：训练数据是两位数（70~82），API 用四位年份（1970~1982），取模 100 映射回去。
    2. origin：训练数据是 one-hot 三列，API 入参是 1/2/3 整数，这里展开成 0/1。
    """
    origin = payload["origin"]
    mapped_year = payload["model_year"] % 100
    return [
        float(payload["cylinders"]),
        float(payload["displacement"]),
        float(payload["horsepower"]),
        float(payload["weight"]),
        float(payload["acceleration"]),
        float(mapped_year),
        1.0 if origin == 1 else 0.0,
        1.0 if origin == 2 else 0.0,
        1.0 if origin == 3 else 0.0,
    ]


def to_frame(features: list[float]) -> pd.DataFrame:
    """包成带列名的 DataFrame（与训练时输入形态一致，消除 sklearn 的 feature names 警告）。"""
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)