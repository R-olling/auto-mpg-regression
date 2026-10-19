# auto-mpg-regression

用线性回归预测汽车城市循环油耗（mpg）的 FastAPI 服务，包含完整的 MLOps / DevOps 流程：
DVC 数据版本（v1/v2）、MLflow 实验跟踪与模型注册、pytest 测试、ruff 代码检查、Docker 容器化、
GitHub Actions CI/CD（PR 全检查 + staging/production 两阶段镜像构建）。

数据集：UCI Auto MPG（398 行 × 9 列，目标列 `mpg`）。

## 应用

| 端点 | 方法 | 说明 |
|---|---|---|
| `/predict` | POST | 输入 7 个车辆参数，返回预测油耗（mpg） |
| `/health` | GET | 健康检查（模型是否就绪） |
| `/model-info` | GET | 模型版本、运行环境、9 维特征契约 |
| `/` | GET | 静态预测表单页 |
| `/docs` | GET | FastAPI 自动生成的交互式 API 文档 |

输入示例：

```json
{"cylinders": 4, "displacement": 120, "horsepower": 79,
 "weight": 2625, "acceleration": 18.6, "model_year": 1982, "origin": 1}
```

返回：

```json
{"mpg": 29.16, "unit": "mpg", "model_version": "local-dev", "app_env": "development"}
```

本地运行：

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Docker 运行：

```bash
docker build -t mpg-app:local .
docker run -d --name mpg-app -p 127.0.0.1:8000:8000 mpg-app:local
```

## 分支方案与合并路径

```
feature/*  ──PR──>  dev  ──PR──>  staging  ──PR──>  main
                     ↑                ↑                 ↑
                 集成分支          预发分支           生产分支
```

- `main`：生产分支，永远保持可交付状态；push 触发 production 镜像构建
- `staging`：预发分支；push 触发 staging 镜像构建
- `dev`：集成分支，日常开发汇聚点
- `feature/*`：功能分支，从 `dev` 切出，如 `feature/dvc-data`、`feature/fastapi-app`

三个长命分支均为**受保护分支**：必须走 PR、必须 `ci` 状态检查通过、
禁止绕过（enforce admins）、禁止强推与删除。

## 流水线触发矩阵

| 事件 | 触发的 job | 内容 |
|---|---|---|
| PR → `dev` / `staging` / `main` | `ci` | black --check + ruff check + pytest + docker build |
| push → `staging` | `ci` + `deploy-staging` | 上述检查 + staging secrets 生成 `.env` + 构建 `:staging` 镜像 |
| push → `main` | `ci` + `deploy-production` | 上述检查 + production secrets 生成 `.env` + 构建 `:production` 镜像 |
| push → `feature/*` | 不触发 | 功能分支只在开 PR 时运行检查 |

## 代码流转路径

```
commit(PR) → [ci: lint + test + docker build] → merge dev
           → PR staging → [ci] → merge → [deploy-staging: .env(secrets) + build :staging]
           → PR main    → [ci] → merge → [deploy-production: .env(secrets) + build :production]
```

## 数据与模型

**数据版本（DVC，远程 Azure `myazure`）**

| 版本 | tag | 说明 | 行数 | `.dvc` md5 |
|---|---|---|---|---|
| v1 | `data-v1` | 原始数据转规范 CSV（`"?"` 原样保留） | 398 | `4f2321b47a64d9849f9325d64ceb5361` |
| v2 | `data-v2` | 中位数填补 + origin one-hot + IQR 剔除异常 | 397 | `3e4175cf1366fa913048bf2678166b78` |

复现：`git checkout data-v1 && dvc pull`。详见 `ml/README_data.md`。

**实验（MLflow，后端 `sqlite:///mlflow.db`）**

| run | 配置 | CV RMSE | 测试 RMSE | MAE | R² |
|---|---|---|---|---|---|
| `baseline-v1` | 线性回归，无缩放（313 行） | — | 3.2727 | 2.4198 | 0.7902 |
| `improved-v2-linear` | 填补 + 标准化 + 线性（319 行） | 3.3672 | 3.2691 | 2.4890 | 0.7906 |
| `improved-v2-ridge-a01` | Ridge α=0.1 | 3.3670 | 3.2699 | 2.4889 | 0.7905 |
| `improved-v2-ridge-a1` | Ridge α=1.0 | 3.3664 | 3.2772 | 2.4878 | 0.7896 |
| `improved-v2-ridge-a10` | Ridge α=10 | 3.4013 | 3.3294 | 2.5078 | 0.7828 |

评估口径：按记录身份固定划分（6 条马力缺失记录一律入训练折），
baseline 与 improved 共用同一批 **79 条**测试记录；中位数与缩放都在 Pipeline 内按训练折拟合（95.0），
测试集不参与；linear 与 Ridge 的选型只看训练折 5 折 CV。

**生产模型**：`AutoMPG_LinearRegression` v6（run `improved-v2-linear-export`），
导出为 `app/model/linear_regression.pkl`（2105 字节），随 `app/` 进入镜像。
详见 `ml/README_experiments.md` 与 `ml/registry/current_model.md`。

## 目录结构

```
auto-mpg-regression/
├── app/                        FastAPI 服务
│   ├── config.py               环境配置集中读取 + 必需变量校验（fail fast）
│   ├── features.py             特征契约（9 维，训练/服务共用）
│   ├── schemas.py              Pydantic 请求/响应与边界校验
│   ├── model.py                模型加载（lru_cache，缺文件快速失败）
│   ├── main.py                 /predict /health /model-info /
│   ├── static/index.html       预测表单页
│   └── model/linear_regression.pkl   生产模型
├── ml/
│   ├── data_pipeline.py        v1/v2 数据生成
│   ├── train.py                训练 + MLflow 五要素 + 注册 + 导出
│   ├── README_data.md          数据来源 / v1→v2 变化 / commit 对应
│   ├── README_experiments.md   哪个 run 可上线 + 优化指标理由
│   ├── PROVENANCE.md           数据-代码-模型溯源链
│   └── registry/current_model.md  生产模型登记卡
├── data/                       DVC 指针（CSV 本体不进 git）
├── tests/                      unit / api / integration（35 个用例）
├── .github/workflows/ci.yml    CI + 两阶段 CD
├── .pre-commit-config.yaml     本地预检（black + ruff）
├── Dockerfile / .dockerignore
├── requirements.txt / requirements-dev.txt / pyproject.toml / pytest.ini
├── README.md / DEPLOYMENT.md / .env.example
```

## 环境变量

见 `.env.example`。`APP_ENV` 与 `MODEL_VERSION` 是**启动必需变量**，由 `app/config.py`
在启动时校验，缺失即抛 `ConfigError` 拒绝启动（fail fast，对应 12-Factor 要求）；
`MODEL_PATH` 有默认值，不属于必需变量。

| 变量 | 默认 | 说明 |
|---|---|---|
| `APP_ENV` | 无（必需） | 运行环境标识，构建期由 `--build-arg` 注入 |
| `MODEL_VERSION` | 无（必需） | 模型版本标识，同上 |
| `MODEL_PATH` | `app/model/linear_regression.pkl` | 运行期可覆盖的模型路径 |

本地开发：`cp .env.example .env` 后填写（`.env` 已被 `.gitignore` 排除）；
容器 / CI：无 `.env`，值由 Docker ENV、`--build-arg` 或 GitHub Secrets 注入。

## 代码质量与测试

工具分工：**Black 管排版**（formatter）、**Ruff 管检查**（linter）、**pytest 管功能**（测试）。
三者行宽统一为 100，避免两个工具互相打架。

```bash
.\.venv\Scripts\python.exe -m black --check .    # All done! 14 files would be left unchanged
.\.venv\Scripts\python.exe -m ruff check .       # All checks passed
.\.venv\Scripts\python.exe -m pytest -q          # 35 passed
```

本地预检（推荐，每个新克隆装一次）：

```bash
.\.venv\Scripts\python.exe -m pre_commit install
.\.venv\Scripts\python.exe -m pre_commit run --all-files
```

CI 里 `ci` job 的步骤顺序是：checkout → setup-python → install → **black** → **ruff** → **pytest** → docker build，
任何一步失败都会阻止合并（分支保护要求 `ci` 通过）。
