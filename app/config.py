"""应用配置：多环境参数集中管理 + 启动时校验必需变量（fail fast）。

对应课堂 12-Factor 原则：
  - 配置全部来自环境变量，代码里不含任何环境差异——同一份代码跑在本地 /
    staging / production，差异全在环境变量里；
  - 本地开发从 .env.example 复制出 .env（已被 .gitignore 排除）后由 load_dotenv
    读入；容器与 CI 里没有 .env，变量由 Docker ENV / --build-arg / workflow
    的 GitHub Secrets 注入，load_dotenv 此时是空操作；
  - 启动时发现必需变量缺失立即报错退出（fail fast），避免"带病运行"。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 本地开发把 .env 读入环境变量；容器/CI 中无 .env 时不生效

# 启动时必须存在的环境变量（缺失即阻止启动）。
# 这两个是"环境身份"标识：不知道自己在哪个环境、是哪个模型版本，就不该对外提供服务。
# MODEL_PATH 有合理默认值，不属于必需变量。
REQUIRED_ENV_VARS = ("APP_ENV", "MODEL_VERSION")

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "model" / "linear_regression.pkl")


class ConfigError(RuntimeError):
    """必需环境变量缺失——服务应当拒绝启动（fail fast）。"""


@dataclass(frozen=True)
class Settings:
    app_env: str
    model_version: str
    model_path: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def get_settings() -> Settings:
    """读取并校验环境配置。必需变量缺失时抛出 ConfigError。"""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise ConfigError(
            f"缺少必需的环境变量: {', '.join(missing)}。"
            "本地开发：从 .env.example 复制创建 .env 并填写；"
            "CI/CD：确认 GitHub Secrets 已注入（见 ci.yml 与 DEPLOYMENT.md）。"
        )
    return Settings(
        app_env=os.environ["APP_ENV"],
        model_version=os.environ["MODEL_VERSION"],
        model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
    )
