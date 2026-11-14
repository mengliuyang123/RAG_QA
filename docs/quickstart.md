# 快速开始

本文介绍如何在本地运行 RAG_QA，并使用真实模型服务完成一次问答。

## 1. 获取代码

```bash
git clone https://github.com/mengliuyang123/RAG_QA.git
cd RAG_QA
```

## 2. 准备 Python 环境

推荐使用 uv：

```bash
uv sync
```

或者：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 创建本地配置

在项目根目录创建 `.env`。准备专用 PostgreSQL 数据库，并由管理员在该数据库执行 `CREATE EXTENSION IF NOT EXISTS vector;`。下面的值只是占位符：

```dotenv
DATABASE_URL=postgresql://rag_user:replace-with-password@127.0.0.1:5432/rag_qa
DATABASE_SCHEMA=rag_qa
EVALUATION_DATABASE_SCHEMA=rag_qa_evaluation
OPENAI_API_KEY=replace-with-your-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

如果使用其他 OpenAI 兼容服务，设置 `LLM_BASE_URL`、`LLM_API_KEY` 和对应模型名称。Embedding 服务可以通过 `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY` 和 `EMBEDDING_MODEL` 单独配置。

## 4. 初始化与启动后端

首次运行或代码包含新迁移时，先在项目根目录执行：

```bash
uv run python -m app.interfaces.cli.database
```

命令初始化应用和评估 schema，并运行官方 LangGraph PostgreSQL 检查点迁移。重复执行不会清空数据。


```bash
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

启动后可访问：

- 健康检查：`http://127.0.0.1:8000/api/health`
- OpenAPI：`http://127.0.0.1:8000/docs`

## 5. 启动前端

新开一个终端：

```bash
cd frontend
npm install
VITE_API_MODE=live VITE_BACKEND_URL=http://127.0.0.1:8000 npm run dev -- --mode live
```

浏览器打开 `http://127.0.0.1:5173`。

Windows PowerShell 可直接使用：

```powershell
cd frontend
npm.cmd run dev -- --mode live --host 127.0.0.1 --port 15173 --strictPort
```

此时访问 `http://127.0.0.1:15173`。如果默认端口 5173 报 `EACCES`，可用
`netsh interface ipv4 show excludedportrange protocol=tcp` 检查 Windows 保留端口，
再选择未被保留的端口。开发服务器会将 `/api` 请求代理到本机 8000 端口。

首次运行需要在项目根目录的新终端中创建账号；已有账号可直接登录，不必重新创建：

```bash
uv run python -m app.interfaces.cli.users create --username admin --name 管理员 --role admin
```

命令会在交互终端中隐藏输入密码。若 `.env` 配置了本地模型地址，需先启动对应服务；
例如 Ollama 使用 `ollama serve`，并确保生成、理解和 Embedding 模型均已安装。
配置 `hybrid_rerank` 时，还必须启动 `RERANKER_BASE_URL` 指向的重排序服务。

## 6. 导入知识

登录后，在知识库页面创建知识库并上传文本、Markdown 或 PDF 文件。系统会执行解析、分块、Embedding 和索引构建，然后即可在聊天页面提问。

## 7. 使用演示模式

不配置后端时，前端默认使用内置演示数据：

```bash
cd frontend
npm run dev
```

演示模式适合查看界面和交互，不会调用真实模型服务。

## 常见问题

### 启动时报 API 密钥错误

确认 `.env` 位于项目根目录，并检查变量名和模型服务地址。不要把密钥写进源代码或提交到 Git。

### 前端无法访问后端

确认后端监听在 `127.0.0.1:8000`，并设置：

```bash
VITE_API_MODE=live
VITE_BACKEND_URL=http://127.0.0.1:8000
```

同时检查后端的 CORS 配置是否包含前端地址。

### 本地数据在哪里

用户、知识库、会话、引用、反馈、分享、入库队列、追踪、向量和 LangGraph 检查点均保存到 PostgreSQL。评估向量使用独立 schema，评估运行记录按 run_id 隔离。

上传原文件、修复日志和导出的评估报告仍保存在 `data/` 或 `evaluation/`。旧 SQLite 和 Chroma 文件保留，不再作为后端使用；本次切换使用全新空库。

### 运行后端测试

在另一个专用数据库启用 pgvector，并在 `.env` 配置 `TEST_DATABASE_URL`。测试 fixture 仅使用该数据库的随机 schema，不使用应用数据库。

```bash
uv sync --group eval
uv run --group eval pytest
```

### 数据库与模型隔离

`DATABASE_SCHEMA` 与 `EVALUATION_DATABASE_SCHEMA` 必须不同。切换 Embedding 模型或维度应使用新集合并重新上传文档，不混用已有向量。当前 2560 维向量使用 cosine 精确检索，知识库范围和未发布文档过滤在取 Top-K 之前执行。
