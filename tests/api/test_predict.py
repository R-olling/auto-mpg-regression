"""API 测试：/predict 的正常路径与各类非法输入（每个非法输入都是一个边界情况）。"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ({"weight": -1}, "负车重"),
        ({"weight": 0}, "车重为零"),
        ({"cylinders": 2}, "气缸数低于数据集范围"),
        ({"cylinders": 20}, "气缸数高于数据集范围"),
        ({"horsepower": -50}, "负马力"),
        ({"acceleration": -3}, "负加速时间"),
        ({"origin": 4}, "产地代码越界"),
        ({"model_year": 1969}, "年份低于训练域下限"),
        ({"model_year": 1983}, "年份高于训练域上限"),
        ({"model_year": 2050}, "跨世纪年份（%100 会错误映射）"),
        ({"model_year": 2101}, "年份超出校验上限"),
    ],
)
def test_predict_rejects_out_of_range(client, valid_payload, mutation, reason):
    """边界：数值超出合理范围/训练域 -> 422，且错误信息指出了问题字段。"""
    resp = client.post("/predict", json={**valid_payload, **mutation})
    assert resp.status_code == 422, reason
    assert "detail" in resp.json()


def test_predict_rejects_missing_field(client, valid_payload):
    """边界：缺少必填字段 -> 422。"""
    payload = {k: v for k, v in valid_payload.items() if k != "horsepower"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_wrong_type(client, valid_payload):
    """边界：字段类型错误（数字传成字符串）-> 422。"""
    payload = {**valid_payload, "weight": "very heavy"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_extra_field(client, valid_payload):
    """边界：提交未定义字段 -> 422（extra=forbid 契约）。"""
    payload = {**valid_payload, "price": 9999}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_valid_returns_structure(client, valid_payload):
    """正常路径：合法输入 -> 200，且响应结构完整、数值合理。"""
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["unit"] == "mpg"
    assert isinstance(body["mpg"], float)
    assert 5.0 < body["mpg"] < 60.0  # 数据集中 mpg 范围是 9~46.6
    assert body["model_version"]
    assert body["app_env"]


def test_health_ok(client):
    """正常路径：健康检查 -> 200 且模型就绪。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_year_mapping_stays_in_domain(client, valid_payload):
    """跨世纪边界：1982 正常返回 200；2082（与 1982 撞车）与 2000（%100=0）一律 422。"""
    assert client.post("/predict", json={**valid_payload, "model_year": 1982}).status_code == 200
    assert client.post("/predict", json={**valid_payload, "model_year": 2082}).status_code == 422
    assert client.post("/predict", json={**valid_payload, "model_year": 2000}).status_code == 422
