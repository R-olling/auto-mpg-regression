"""集成测试：TestClient 走完整链路（HTTP -> 校验 -> 特征工程 -> 模型 -> 响应）。"""

from __future__ import annotations

from app.features import FEATURE_COLUMNS


def test_full_prediction_flow(client, valid_payload):
    """正常路径：一次完整预测请求，从进 HTTP 到出结果。"""
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["unit"] == "mpg"
    assert isinstance(body["mpg"], float)


def test_health_and_model_info(client):
    """正常路径：健康检查 + 模型信息接口，特征契约有 9 维。"""
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["features"] == FEATURE_COLUMNS
    assert len(info.json()["features"]) == 9


def test_invalid_input_through_full_stack(client, valid_payload):
    """边界：非法输入穿过完整技术栈后仍被校验层拦下（422 而非 500）。"""
    resp = client.post("/predict", json={**valid_payload, "horsepower": -10})
    assert resp.status_code == 422


def test_index_page_served(client):
    """正常路径：静态首页可访问（200 且返回 HTML）。"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
