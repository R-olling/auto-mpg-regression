"""模型单元测试：模型可加载、预测值稳定、行为单调、特征契约一致。"""
from __future__ import annotations

import pytest

from app.features import FEATURE_COLUMNS, payload_to_features, to_frame
from app.model import get_model
from ml.data_pipeline import V2_FEATURE_COLUMNS

# 1982 年福特 Ranger，该车真实油耗 28 mpg
SAMPLE = {
    "cylinders": 4,
    "displacement": 120.0,
    "horsepower": 79.0,
    "weight": 2625.0,
    "acceleration": 18.6,
    "model_year": 1982,
    "origin": 1,
}
EXPECTED_MPG = 29.16  # 实测值；容差 ±1.5 mpg
TOLERANCE = 1.5


def test_model_file_loads():
    """边界 1：生产模型文件存在且可加载（缺文件时应直接失败而不是返回垃圾预测）。"""
    model = get_model()
    assert hasattr(model, "predict")


def test_fixed_sample_prediction_within_tolerance():
    """边界 2：对固定样本的预测值应稳定在预期附近（防止模型文件被误替换）。"""
    predicted = float(get_model().predict(to_frame(payload_to_features(SAMPLE)))[0])
    assert predicted == pytest.approx(EXPECTED_MPG, abs=TOLERANCE)


def test_heavier_car_predicted_lower_mpg():
    """边界 3：单调性——其他条件相同，车越重预测油耗越高（mpg 数值越低）。"""
    light = get_model().predict(to_frame(payload_to_features(SAMPLE)))[0]
    heavy_payload = {**SAMPLE, "weight": SAMPLE["weight"] + 800}
    heavy = get_model().predict(to_frame(payload_to_features(heavy_payload)))[0]
    assert heavy < light


def test_newer_car_predicted_higher_mpg():
    """边界 4：单调性——年代越新预测油耗越高（1977 vs 1982，都在训练域内）。"""
    old_payload = {**SAMPLE, "model_year": 1977}
    older = get_model().predict(to_frame(payload_to_features(old_payload)))[0]
    newer = get_model().predict(to_frame(payload_to_features(SAMPLE)))[0]  # 1982
    assert newer > older


def test_feature_contract_matches_pipeline():
    """边界 5：服务端特征顺序必须与训练管道产出的列顺序一致（防契约漂移）。"""
    assert FEATURE_COLUMNS == V2_FEATURE_COLUMNS


def test_origin_one_hot_expansion():
    """边界 6：origin 整数应正确展开为 one-hot 三列。"""
    assert payload_to_features({**SAMPLE, "origin": 2})[-3:] == [0.0, 1.0, 0.0]
    assert payload_to_features({**SAMPLE, "origin": 3})[-3:] == [0.0, 0.0, 1.0]