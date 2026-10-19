"""边界补充测试（团队分工 B）：在既有 API 用例之外补齐契约边界。"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "field, value",
    [
        ("origin", 0),  # 产地代码下界
        ("origin", 4),  # 产地代码上界
        ("weight", 6001),  # 车重超上限
        ("horsepower", 601),  # 马力超上限
    ],
)
def test_rejects_out_of_contract_values(client, valid_payload, field, value):
    """边界：超出 schemas.py 声明的范围 -> 422。"""
    resp = client.post("/predict", json={**valid_payload, field: value})
    assert resp.status_code == 422, field


@pytest.mark.parametrize("year", [1970, 1982])
def test_accepts_domain_boundary_years(client, valid_payload, year):
    """边界：训练域的两个端点年份应当放行（与 1969/1983 的拒绝形成对照）。"""
    assert client.post("/predict", json={**valid_payload, "model_year": year}).status_code == 200


def test_rejects_missing_multiple_fields(client, valid_payload):
    """边界：一次缺 3 个必填字段 -> 422，且错误里列出多处缺失。"""
    drop = {"cylinders", "weight", "origin"}
    payload = {k: v for k, v in valid_payload.items() if k not in drop}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert len(resp.json()["detail"]) >= 3


def test_model_info_exposes_model_path(client):
    """契约：/model-info 应暴露模型文件路径（部署排查用）。"""
    body = client.get("/model-info").json()
    assert "model_path" in body
    assert body["model_path"].endswith("linear_regression.pkl")
