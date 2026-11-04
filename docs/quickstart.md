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

在项目根目录创建 `.env`。下面的值只是占位符：

```dotenv
OPENAI_API_KEY=replace-with-your-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

如果使用其他 OpenAI 兼容服务，设置 `LLM_BASE_URL`、`LLM_API_KEY` 和对应模型名称。Embedding 服务可以通过 `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY` 和 `EMBEDDING_MODEL` 单独配置。

## 4. 启动后端

```bash
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

启动后可访问：

- 健康检查：`http://127.0.0.1:8000/health`
- OpenAPI：`http://127.0.0.1:8000/docs`

## 5. 启动前端

新开一个终端：

```bash
cd frontend
npm install
VITE_API_MODE=live VITE_BACKEND_URL=http://127.0.0.1:8000 npm run dev -- --mode live
```

浏览器打开 `http://127.0.0.1:5173`。

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

认证、知识库、会话、追踪和向量数据默认写入项目的 `data/` 子目录。这些运行时数据已被 Git 忽略，请勿上传。
