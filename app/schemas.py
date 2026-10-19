"""请求/响应模型与校验规则：本文件只定义契约，不做任何模型或数据库访问。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """一次油耗预测请求的输入。字段边界取自 Auto MPG 数据集的取值范围（略放宽）。"""

    model_config = ConfigDict(extra="forbid")  # 提交任何未定义字段都判 422

    cylinders: int = Field(ge=3, le=12, description="气缸数")
    displacement: float = Field(gt=0, le=1000, description="排量（立方英寸）")
    horsepower: float = Field(gt=0, le=600, description="马力")
    weight: float = Field(gt=0, le=6000, description="车重（磅）")
    acceleration: float = Field(gt=0, le=40, description="0-60 mph 加速时间（秒）")
    model_year: int = Field(ge=1970, le=1982, description="四位年份，限定训练域 1970~1982")
    origin: int = Field(ge=1, le=3, description="产地：1=美国 2=欧洲 3=日本")


class PredictionResponse(BaseModel):
    """预测响应：油耗（mpg）+ 模型与环境信息。"""

    mpg: float = Field(description="预测的每加仑英里数")
    unit: str = Field(default="mpg")
    model_version: str = Field(description="生产模型版本标识")
    app_env: str = Field(description="运行环境（由 APP_ENV 环境变量决定）")