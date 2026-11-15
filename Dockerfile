# syntax=docker/dockerfile:1
# RAG_QA 后端镜像（FastAPI + uvicorn）。
# 依赖来自 requirements.txt（由 uv.lock 导出，版本全部锁定）。
# 国内构建可换 PyPI 镜像:
#   docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ -t rag-qa .

FROM python:3.11-slim AS builder

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先只装依赖，利用层缓存：代码改动不会触发重新安装
COPY requirements.txt ./
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --index-url "$PIP_INDEX_URL" -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 1000 app

COPY --from=builder /app/.venv /app/.venv
COPY app ./app
# 后端的评估报告接口会导入 evaluation.reporting
COPY evaluation ./evaluation
COPY main.py ./

# 结构化数据都在 PostgreSQL；这里只放仍落盘的两类文件，必须挂卷持久化：
#   data/docs   上传的原始文件（列表/删除时会回读）
#   data/repair 文档删除失败的修复日志
RUN mkdir -p /app/data /app/evaluation/results \
    && chown -R app:app /app/data /app/evaluation/results
VOLUME ["/app/data"]

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
