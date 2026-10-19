"""公共测试夹具：整个测试会话共用一个 TestClient（模型只加载一次）。"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# 应用启动时会校验必需环境变量（fail fast，见 app/config.py）。
# 测试环境在此统一提供一套默认值，保证测试不依赖开发者本机的 .env。
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("MODEL_VERSION", "local-dev")

from app.main import app  # noqa: E402  （环境变量要先于 app 导入就绪）


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
