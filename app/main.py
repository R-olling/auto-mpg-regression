"""FastAPI 预测服务入口：启动时加载模型，提供 /predict、/health、/model-info 与静态页面。"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.features import FEATURE_COLUMNS, payload_to_features, to_frame
from app.model import get_model, get_model_info
from app.schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时预热模型：文件缺失或损坏时服务直接起不来（快速失败，避免半死状态）
    get_model()
    yield


app = FastAPI(
    title="Auto MPG Regression Service",
    description="输入车辆参数，用线性回归模型预测城市循环油耗（mpg）",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """健康检查：模型是否就绪。"""
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info() -> dict:
    """返回生产模型信息与特征契约（前端与调试用）。"""
    return {**get_model_info(), "features": FEATURE_COLUMNS}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """输入车辆特征，返回预测油耗（mpg）。输入非法由 Pydantic 自动返回 422。"""
    features = payload_to_features(request.model_dump())
    mpg = float(get_model().predict(to_frame(features))[0])
    info = get_model_info()
    return PredictionResponse(
        mpg=round(mpg, 2),
        unit="mpg",
        model_version=info["model_version"],
        app_env=info["app_env"],
    )


@app.get("/")
def index() -> FileResponse:
    """极简静态页：一个表单 + fetch 调用 /predict。"""
    return FileResponse("app/static/index.html")