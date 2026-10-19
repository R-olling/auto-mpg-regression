"""公共测试夹具：整个测试会话共用一个 TestClient（模型只加载一次）。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    """一个合法的预测请求（1982 年福特 Ranger 的参数）。"""
    return {
        "cylinders": 4,
        "displacement": 120.0,
        "horsepower": 79.0,
        "weight": 2625.0,
        "acceleration": 18.6,
        "model_year": 1982,
        "origin": 1,
    }