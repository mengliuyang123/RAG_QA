# RAG_QA

一个基于 FastAPI、LangGraph、Chroma 和 React 的检索增强生成（RAG）知识库问答系统。

## 功能

- 文档上传、解析、分块、向量检索和 BM25 关键词检索
- 多轮对话、流式回答、引用和证据展示
- 知识库管理、用户认证、反馈与运行追踪
- React + Vite 前端，支持演示数据和真实后端两种模式
- Python 测试与评估工具

## 项目结构

- `app/`：后端应用、领域模型和基础设施
- `frontend/`：React 前端
- `evaluation/`：评估运行时和指标
- `tests/`：后端测试
- `scripts/`：开发辅助脚本

项目运行时产生的数据库、向量索引和日志会写入本地 `data/` 或 `evaluation/` 目录，不应提交到 Git。

## 环境要求

- Python 3.11+
- Node.js 20+
- npm
- 推荐使用 uv 管理 Python 环境

## 后端安装与启动

```bash
uv sync
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

如果不使用 uv，也可以使用 Python 虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 配置

应用从进程环境变量和项目根目录的本地 `.env` 文件读取配置。请在本地创建 `.env`，不要把它提交到 Git。

最小配置示例：

```dotenv
OPENAI_API_KEY=replace-with-your-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

也可以使用兼容 OpenAI API 的服务：

```dotenv
LLM_API_KEY=replace-with-your-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model
EMBEDDING_API_KEY=replace-with-your-key
EMBEDDING_BASE_URL=https://your-provider.example/v1
EMBEDDING_MODEL=your-embedding-model
```

密钥只应通过本地环境变量或密钥管理服务注入。不要把真实密钥、数据库文件或运行日志提交到仓库。

## 前端安装与启动

```bash
cd frontend
npm install
npm run dev
```

默认前端使用演示数据。连接本地后端时：

```bash
cd frontend
VITE_API_MODE=live VITE_BACKEND_URL=http://127.0.0.1:8000 npm run dev -- --mode live
```

前端默认地址为 `http://127.0.0.1:5173`。

## 测试

后端：

```bash
uv run pytest
```

前端：

```bash
cd frontend
npm test
npm run build
```

## 安全提示

- 不要提交 `.env`、API 密钥、私有证书、数据库和本地运行产物。
- 公开部署前，请配置可信主机、CORS 来源、HTTPS 和安全 Cookie。
- 生产环境应使用独立数据库和密钥管理方案。

## License

当前仓库未附带许可证文件。如需允许他人复制、修改或分发代码，请先添加合适的开源许可证。
