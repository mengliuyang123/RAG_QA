# RAG_QA

一个基于 FastAPI、LangGraph、PostgreSQL、pgvector 和 React 的检索增强生成（RAG）知识库问答系统。

## 功能

- 文档上传、解析、分块、向量检索和 PostgreSQL 中文全文检索（Jieba + GIN）
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

业务数据、向量、LangGraph 检查点和评估运行记录统一保存在 PostgreSQL。原始文档、修复日志与评估报告仍写入本地 `data/` 或 `evaluation/`，不应提交到 Git。

## 环境要求

- Python 3.11+
- Node.js 20+
- npm
- PostgreSQL（本机已验证 18.7）及已启用的 pgvector 扩展（已验证 0.8.7）
- 推荐使用 uv 管理 Python 环境

## 后端安装与启动

```bash
uv sync
uv run python -m app.interfaces.cli.database
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

最小配置示例（`DATABASE_URL` 必须指向专用应用数据库，先由数据库管理员启用 `vector` 扩展）：

```dotenv
DATABASE_URL=postgresql://rag_user:replace-with-password@127.0.0.1:5432/rag_qa
DATABASE_SCHEMA=rag_qa
EVALUATION_DATABASE_SCHEMA=rag_qa_evaluation
OPENAI_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

也可以使用兼容 OpenAI API 的服务：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model
EMBEDDING_API_KEY=your-api-key
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

前端默认地址为 `http://127.0.0.1:15173`（避开 Windows 常见保留端口段 5078–5177）。
若该端口仍报 `EACCES`，用 `netsh interface ipv4 show excludedportrange protocol=tcp`
检查保留端口，再通过 `--port <端口> --strictPort` 换一个未被保留的端口。

全新数据库没有默认账号。在项目根目录的交互终端创建管理员：

```bash
uv run python -m app.interfaces.cli.users create --username mengl --name 管理员 --role admin
```

密码通过终端隐藏输入。旧 SQLite/Chroma 数据保留在原目录，不自动导入或回退使用。

## 测试

后端使用独立测试数据库，并在其中为每个测试建立随机 schema。请在 `.env` 设置指向另一数据库的 `TEST_DATABASE_URL`，预先启用 pgvector；禁止把它配置为应用数据库。

```bash
uv sync --group eval
uv run --group eval pytest
```

`python -m app.interfaces.cli.database` 是显式、可重复执行的应用/评估 schema 迁移命令。运行进程不自动变更表结构。当前 2560 维向量使用完整精度的 cosine 精确检索；业务查询与评估在不同 schema 隔离。

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
