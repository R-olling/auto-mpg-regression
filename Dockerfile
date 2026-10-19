# 构建参数：默认 Docker Hub 基础镜像；国内可用 --build-arg 切课程 Azure 代理
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# 先只拷贝依赖清单——利用层缓存：依赖不变则本层可复用，改代码不重装依赖
COPY requirements.txt ./

# 可选 pip 镜像源：国内网络下构建容器内访问 PyPI 可能受限
ARG PIP_INDEX_URL=
RUN if [ -n "$PIP_INDEX_URL" ]; then \
        python -m pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt; \
    else \
        python -m pip install --no-cache-dir -r requirements.txt; \
    fi

# 再拷贝应用代码（app/model/ 里的生产模型随 app/ 一起进镜像）
COPY app/ ./app/

# 以非 root 数字用户运行（安全加固）
RUN useradd --create-home --uid 10001 appuser
USER 10001

# 构建期写进镜像的环境标识，由 CI 的 --build-arg 传入
ARG APP_ENV=production
ARG MODEL_VERSION=local-dev
ENV APP_ENV=${APP_ENV}
ENV MODEL_VERSION=${MODEL_VERSION}

EXPOSE 8000

# exec 形式让 Python 成为 PID 1；--host 0.0.0.0 才能被端口映射访问
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]