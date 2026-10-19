# 部署说明（DEPLOYMENT）

本项目的「交付」到**构建镜像为止**（课程项目不要求真正发布/部署到服务器）。
两个环境（staging / production）的区别只在于**触发分支**、**`.env` 取值来源**与**镜像标签**。

## staging 环境

- **触发**：PR 合并到 `staging` 分支（即 push 到 `staging`）
- **流程**：
  1. `ci` job 先跑（ruff + pytest + docker build），全绿才进入下一步（`needs: ci`）
  2. `deploy-staging` job 用 GitHub Secret `STAGING_MODEL_VERSION` 现场生成 `.env`：
     ```
     APP_ENV=staging
     MODEL_VERSION=<secret 值，当前为 v6-staging>
     ```
  3. 执行：
     ```
     docker build --build-arg APP_ENV=staging --build-arg MODEL_VERSION=v6-staging \
                  -t <repo>:staging .
     ```
- **所需 secrets**：`STAGING_MODEL_VERSION`
- **产物标签**：`:staging`

## production 环境

- **触发**：PR 合并到 `main` 分支（即 push 到 `main`）
- **流程**：
  1. `ci` job 先跑，全绿才继续
  2. `deploy-production` job 用 GitHub Secret `PRODUCTION_MODEL_VERSION` 生成 `.env`：
     ```
     APP_ENV=production
     MODEL_VERSION=<secret 值，当前为 v6>
     ```
  3. 执行：
     ```
     docker build --build-arg APP_ENV=production --build-arg MODEL_VERSION=v6 \
                  -t <repo>:production .
     ```
- **所需 secrets**：`PRODUCTION_MODEL_VERSION`
- **产物标签**：`:production`

## 两阶段差异

| 项 | staging | production |
|---|---|---|
| 源分支 | `staging` | `main` |
| 触发事件 | push（PR 合并后） | push（PR 合并后） |
| `APP_ENV` | `staging` | `production` |
| 版本 secret | `STAGING_MODEL_VERSION` | `PRODUCTION_MODEL_VERSION` |
| 当前值 | `v6-staging` | `v6` |
| 镜像标签 | `:staging` | `:production` |
| 前置条件 | `ci` 全绿 | `ci` 全绿 |

## 为什么 `.env` 必须在流水线里现生成

`.env` 被 `.dockerignore` 排除，且被 `.gitignore` 忽略，仓库里根本不存在这个文件；
它包含的值来自 GitHub Secrets，**只能由流水线在运行时写入**。

`.env` 只是「取值来源」：真正进镜像靠两个 `--build-arg`。少了任何一个，
容器里的 `APP_ENV` / `MODEL_VERSION` 就会停留在默认值（`production` / `local-dev`），
`/model-info` 返回的标识与实际环境对不上。

## 本地验证镜像

```bash
docker build --build-arg APP_ENV=development --build-arg MODEL_VERSION=local-dev -t mpg-app:local .
docker run -d --name mpg-app -p 127.0.0.1:8000:8000 mpg-app:local
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/model-info
```

预期 `/model-info` 返回：

```json
{"model_version":"local-dev","app_env":"development","features":[...9 维...]}
```

镜像约 1.26 GB（`python:3.12-slim` + 全量训练依赖）。

## 本地开发（不用容器）

```bash
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 安全说明

- 所有凭证只存 **GitHub Secrets**，工作流文件里没有任何明文；
  `${{ secrets.* }}` 在日志中会被自动打码
- `.env` 在流水线中现生成、不入库（`.gitignore` + `.dockerignore` 双重排除）
- DVC 的 SAS 令牌只存 `.dvc/config.local`，DVC 默认将其排除出 Git
- 原始数据只进 DVC / Azure，不进 Git、不进镜像（`.dockerignore` 排除 `data/`）
- 镜像以非 root 用户（uid 10001）运行

## 回滚

镜像按环境打标签。回滚到上一个可用版本：

```bash
git checkout <上一条合并提交或 tag>
docker build --build-arg APP_ENV=production --build-arg MODEL_VERSION=<旧版本标识> -t <repo>:production .
```

模型侧的回滚以 MLflow Registry 的版本号为准（版本不可变，`current_model.md` 记录当前生产版本）。
