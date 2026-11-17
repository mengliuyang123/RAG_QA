# RAG_QA · 团队知识库问答

基于 **FastAPI + LangGraph + PostgreSQL/pgvector + React** 的检索增强生成（RAG）知识库问答系统。
把团队的 PDF、Word、PPT、Markdown 文档放进知识库，员工用自然语言提问，系统检索相关段落、生成回答，并给每个结论标注出处。
管理员可以看到哪些问题答不上来、哪些回答被点踩、每天花了多少钱，再通过补文档、写标准答案、标记过时文档持续改进知识库。

![问答页面](docs/images/02-chat.png)

> 截图来自前端演示模式（`npm run dev`，内置模拟数据，不需要后端），界面和真实后端一致。

---

## 目录

- [功能一览](#功能一览)
- [页面演示](#页面演示)
- [系统架构](#系统架构)
- [问答流程](#问答流程)
- [文档入库流程](#文档入库流程)
- [角色与知识库权限](#角色与知识库权限)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [Docker 部署](#docker-部署)
- [测试与评估](#测试与评估)
- [项目结构](#项目结构)
- [安全提示](#安全提示)

---

## 功能一览

### 面向提问的用户

| 功能 | 说明 |
| --- | --- |
| 多轮问答 | 流式输出，自动理解追问里的指代（“它”“上面那个”）；会话可搜索、重命名、删除 |
| 引用出处 | 回答中的 `[1]` `[2]` 对应具体文档段落，点击可在侧栏查看原文和上下文 |
| 检索过程可见 | 展开“参考了 N 份资料”可以看到在哪个知识库搜了什么、找到几段、是否扩大了范围 |
| 自动选库 / 指定库 | 默认按知识库描述自动路由；也可以只在某几个库里提问 |
| 推荐问题 | 首页按知识库展示推荐问题，帮助新用户知道能问什么 |
| 反馈 | 点赞 / 点踩，点踩可以选原因（答错、不完整、答非所问、内容过时、引用错误） |
| 分享回答 | 一键生成 30 天有效的链接，同事登录后可查看问题、回答和引用 |
| 浏览知识库 | 按库查看文档列表、搜索文件名、预览文本块 |
| 修改口令 | 自助修改，修改后其他浏览器上的登录会被退出 |

### 面向管理员和知识库负责人

| 功能 | 说明 |
| --- | --- |
| 知识库管理 | 新建知识库、编辑描述（用于自动路由）、上传 / 删除文档，查看每个库的提问量、拒答率、扩大范围比例 |
| 文档格式 | PDF、Word（`.docx`）、PowerPoint（`.pptx`）、Markdown、TXT，后台异步入库并显示进度 |
| 文档时效 | 给文档设置复审日期、标记“已过时”；已过时的文档立即停止参与检索 |
| 推荐问题 | 每个库最多 20 个，每行一个，保存即发布到聊天首页 |
| 标准答案（FAQ） | 为高频问题写确定答案，合成一篇“常见问题（标准答案）.md”发布到库里，问答时优先被检索和引用 |
| 访问权限 | 把知识库设为“仅成员可见”，成员分“可查看”和“可管理”；可管理的成员只能维护自己负责的库 |
| 问题洞察 | 拒答问题按相似度聚类；点踩反馈分为检索问题、生成问题、缺内容、不是问题 |
| 知识缺口 | 把拒答聚类和点踩汇总成待办，补文档或写标准答案后一键复测样例问题，全部答上即为“已验证” |
| 用量与费用 | 按日期 / 用户 / 知识库统计提问数、模型调用、Token 和估算费用，列出最耗费的请求 |
| 用户管理 | 新建用户、设为管理员或成员、禁用 / 启用、重置口令 |
| 模型配置 | 在页面上切换生成模型、问题理解模型、向量模型、重排模型，保存后立即生效 |

---

## 页面演示

### 首页与问答

<table>
<tr>
<td width="50%"><img src="docs/images/01-home.png" alt="首页"/><br/><b>首页</b>：选择“自动”或指定知识库，下方是各库的推荐问题</td>
<td width="50%"><img src="docs/images/02-chat.png" alt="问答"/><br/><b>问答</b>：检索过程、带编号的引用、有据可查标记、反馈与分享按钮</td>
</tr>
<tr>
<td><img src="docs/images/03-citation.png" alt="引用原文"/><br/><b>引用原文</b>：点击引用，在右侧查看命中段落（高亮）及其前后文</td>
<td><img src="docs/images/04-share.png" alt="分享页"/><br/><b>分享页</b>：分享链接打开后显示问题、回答、引用和分享人</td>
</tr>
</table>

### 知识库

<table>
<tr>
<td width="50%"><img src="docs/images/05-kb-browse.png" alt="浏览知识库"/><br/><b>浏览知识库</b>：所有用户可见，查看文档与文本块</td>
<td width="50%"><img src="docs/images/06-admin-kb.png" alt="知识库管理"/><br/><b>知识库管理 · 文档</b>：库统计、描述、上传，“时效”列显示已过时 / 待复审</td>
</tr>
<tr>
<td><img src="docs/images/07-faq.png" alt="标准答案"/><br/><b>标准答案</b>：新增、编辑、删除，每次修改自动重新发布</td>
<td><img src="docs/images/08-access.png" alt="访问权限"/><br/><b>访问权限</b>：仅成员可见，成员设为“可查看”或“可管理”</td>
</tr>
</table>

### 运营分析

<table>
<tr>
<td width="50%"><img src="docs/images/09-feedback.png" alt="点踩反馈"/><br/><b>点踩反馈分类</b>：用户反馈“内容过时”时，可直接把被引用的文档标记过时</td>
<td width="50%"><img src="docs/images/11-gaps.png" alt="知识缺口"/><br/><b>知识缺口</b>：样例问题、补充到哪个库、写标准答案、复测</td>
</tr>
<tr>
<td><img src="docs/images/10-usage.png" alt="用量与费用"/><br/><b>用量与费用</b>：近 7 / 30 / 90 天，按日期、用户或知识库分组</td>
<td><img src="docs/images/12-users.png" alt="用户管理"/><br/><b>用户管理</b>：角色、启用状态、重置口令</td>
</tr>
<tr>
<td><img src="docs/images/13-settings.png" alt="模型配置"/><br/><b>模型配置</b>：各类模型的地址、模型名和密钥（密钥默认掩码显示）</td>
<td></td>
</tr>
</table>

---

## 系统架构

后端采用分层架构，依赖方向只能由外向内：接口层不直接依赖基础设施，应用层只依赖端口（接口），具体实现由组合根（`app/bootstrap/container.py`）注入。`tests/` 中有架构测试强制检查这些依赖方向。

```mermaid
flowchart TB
    subgraph Browser["浏览器"]
        UI["React 19 + Vite<br/>assistant-ui · Tailwind"]
    end

    subgraph Backend["FastAPI 后端"]
        direction TB
        HTTP["interfaces/http<br/>路由 · 鉴权 · CSRF · 限流 · SSE 流式"]
        CLI["interfaces/cli<br/>数据库迁移 · 用户管理"]
        APP["application<br/>问答编排 · 检索 · 入库 · 权限 · 洞察 · 追踪"]
        DOMAIN["domain<br/>Pydantic 领域模型与规则"]
        PORTS["application/ports<br/>仓储与模型端口"]
        INFRA["infrastructure<br/>PostgreSQL 仓储 · 解析器 · 分块<br/>Embedding / LLM / Rerank 客户端 · LangGraph Agent"]
        BOOT["bootstrap<br/>组合根 · 配置存储"]
        HTTP --> APP
        CLI --> APP
        APP --> DOMAIN
        APP --> PORTS
        INFRA -. 实现 .-> PORTS
        BOOT --> INFRA
        BOOT --> HTTP
    end

    subgraph Data["PostgreSQL 18"]
        PG[("业务表 · 会话 · 追踪<br/>pgvector 向量<br/>中文全文索引 GIN<br/>LangGraph 检查点")]
    end

    subgraph Models["OpenAI 兼容模型服务"]
        LLM["生成模型"]
        UND["问题理解模型（可选）"]
        EMB["Embedding 模型"]
        RR["Rerank 模型（可选）"]
    end

    UI -- "HTTPS / Cookie 会话" --> HTTP
    INFRA --> PG
    INFRA --> LLM
    INFRA --> UND
    INFRA --> EMB
    INFRA --> RR
```

| 层 | 目录 | 职责 |
| --- | --- | --- |
| 领域层 | `app/domain` | 会话、文档、知识库、权限、反馈、追踪等模型和纯规则 |
| 应用层 | `app/application` | 用例编排：问答、检索融合、入库任务、权限判断、洞察聚类、费用统计 |
| 端口 | `app/application/ports` | 仓储、模型、解析器等抽象接口 |
| 基础设施 | `app/infrastructure` | PostgreSQL 仓储与迁移、文档解析、分块、模型客户端、LangGraph Agent |
| 接口层 | `app/interfaces` | HTTP 路由（`/api/...`）和命令行工具 |
| 组合根 | `app/bootstrap` | 读取配置、按需创建并连接各组件 |

所有业务数据——用户、会话、知识库、文档元数据、向量、全文索引、反馈、追踪、LangGraph 检查点、评估记录——都保存在 PostgreSQL。原始上传文件和删除修复日志保存在本地 `data/`。

---

## 问答流程

```mermaid
flowchart TD
    Q["用户提问"] --> P{"知识库权限过滤<br/>只保留当前用户可见的库"}
    P --> U["问题理解<br/>补全指代 · 识别意图和复杂度"]
    U --> R{"确定性路由"}
    R -- 闲聊 / 系统问题 --> D["直接回答<br/>不检索"]
    R -- 问题含糊 --> C["反问澄清"]
    R -- 单跳问题 --> F
    R -- 多跳 / 比较 / 汇总 --> A

    subgraph Fast["快速通道"]
        F["选择知识库<br/>自动按描述路由或用户指定"] --> S["混合检索<br/>pgvector 向量 + 中文全文检索"]
        S --> M["RRF 融合 → Rerank 精排"]
        M --> J{"证据是否足够？"}
        J -- 不足 --> X["扩大到全部可见知识库再查一次"]
        X --> J2{"仍不足？"}
        J -- 足够 --> G["生成回答<br/>校验每条引用"]
        J2 -- 足够 --> G
    end

    J2 -- 仍不足 --> A
    subgraph Agent["Agent 通道（LangGraph）"]
        A["规划子问题"] --> AS["多次检索（最多 3 次）"]
        AS --> AG["生成 + 答案审核"]
    end

    G --> OUT["流式输出回答与引用"]
    AG --> OUT
    AG -- 无可靠证据 --> REF["拒答<br/>记入问题洞察和知识缺口"]
    OUT --> T["记录追踪<br/>耗时 · Token · 估算费用"]
```

要点：

- **只回答知识库里有的内容**。证据不足时明确拒答，而不是编造；拒答会被聚类，出现在“问题洞察”和“知识缺口”里。
- **检索策略**由 `KNOWLEDGE_RETRIEVAL_PROFILE` 选择：`vector`、`hybrid`、`hybrid_rerank`，默认 `auto`——配置了 `RERANKER_BASE_URL` 时用 `hybrid_rerank`，否则用 `hybrid`。混合检索用倒数排名融合（RRF），不依赖两路分数的尺度。
- **已过时文档**和**未发布文档**会在检索阶段被排除。
- **无权访问的知识库**在理解、路由和检索的每一步都会被过滤掉；直接指定它提问，得到的提示与“知识库不存在”相同，不暴露它的存在。
- 每轮请求都有总时限和检索次数上限，从快速通道升级到 Agent 时不会重置预算。

---

## 文档入库流程

```mermaid
flowchart LR
    UP["上传文件<br/>PDF · DOCX · PPTX · MD · TXT"] --> Q["入库任务队列<br/>后台 worker 处理"]
    Q --> PA["解析<br/>PDF 记录页码 · PPT 记录幻灯片序号"]
    PA --> CH["递归分块<br/>默认 500 字，重叠 50"]
    CH --> EM["批量向量化"]
    CH --> TK["Jieba 分词"]
    EM --> PGV[("pgvector")]
    TK --> PGT[("tsvector + GIN")]
    PGV --> PUB["发布文档"]
    PGT --> PUB
```

- 前端上传后可以看到每个文件的排队、解析、向量化进度；失败时显示原因。
- 上传同名文件时可以选择替换旧版本：新版本入库成功后才替换，之后清理旧版本的源文件。
- 标准答案（FAQ）也走同一条入库流程，作为一篇 Markdown 文档发布到所属知识库。

---

## 角色与知识库权限

| 能力 | 管理员 | 库成员 · 可管理 | 库成员 · 可查看 | 普通成员 |
| --- | :---: | :---: | :---: | :---: |
| 在公开知识库中提问 / 浏览 | ✅ | ✅ | ✅ | ✅ |
| 在“仅成员可见”的知识库中提问 / 浏览 | ✅ | ✅ | ✅ | — |
| 维护该库的文档、描述、推荐问题、标准答案、时效 | ✅ | ✅ | — | — |
| 新建知识库、设置访问权限 | ✅ | — | — | — |
| 问题洞察、知识缺口、用量与费用、用户管理、模型配置 | ✅ | — | — | — |

- 知识库默认公开，所有登录用户都能看到并提问。
- 角色和成员关系每次请求都从数据库读取，修改后对方的下一次请求即生效。
- 已禁用的账号什么都看不到；禁用或重置口令会让该账号立即退出所有已登录的浏览器。
- “可管理”的成员在侧栏会看到“我负责的知识库”入口。

---

## 快速开始

### 环境要求

- Python 3.11+（推荐用 [uv](https://docs.astral.sh/uv/) 管理）
- Node.js 20+ 和 npm
- PostgreSQL（已验证 18.7），并启用 pgvector 扩展（已验证 0.8.7）
- 一个 OpenAI 兼容的模型服务（生成模型 + Embedding 模型；Rerank 可选）

### 只想看看界面？

前端自带演示数据，不需要后端和数据库：

```bash
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:15173`，会以演示管理员身份自动登录。演示模式下修改口令时，当前口令是 `demo-password-123`。

### 完整运行

**1. 准备数据库**

创建专用的应用数据库，并由数据库管理员启用扩展：

```sql
CREATE DATABASE rag_qa;
\c rag_qa
CREATE EXTENSION IF NOT EXISTS vector;
```

**2. 配置 `.env`**

```bash
cp .env.example .env
```

按[配置说明](#配置说明)填写数据库地址和模型密钥。

**3. 安装依赖、迁移数据库、创建管理员**

```bash
uv sync
uv run python -m app.interfaces.cli.database
uv run python -m app.interfaces.cli.users create --username admin --name 管理员 --role admin
```

迁移命令可以重复执行，每次升级代码后都需要运行一次（运行中的服务不会自动改表结构）。创建用户时口令在终端隐藏输入，长度 15–128 个字符。

**4. 启动后端**

```bash
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：`http://127.0.0.1:8000/api/health`。

**5. 启动前端（连接真实后端）**

```bash
cd frontend
npm install
VITE_API_MODE=live VITE_BACKEND_URL=http://127.0.0.1:8000 npm run dev -- --mode live
```

用第 3 步创建的管理员登录，先在“知识库管理”里新建一个库、上传文档，入库完成后就可以提问。

> **Windows 端口提示**：前端默认端口 15173 避开了 Windows 常见的保留端口段（5078–5177）。如果仍报 `EACCES`，用 `netsh interface ipv4 show excludedportrange protocol=tcp` 查看保留端口，再用 `--port <端口> --strictPort` 换一个。

<details>
<summary>不使用 uv 时</summary>

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.interfaces.cli.database
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

</details>

---

## 配置说明

配置从进程环境变量和项目根目录的 `.env` 读取。`.env` 不要提交到 Git。模型相关配置也可以在“模型配置”页面修改，保存后立即生效并写回 `.env`。

### 最小配置

```dotenv
DATABASE_URL=postgresql://rag_user:replace-with-password@127.0.0.1:5432/rag_qa
DATABASE_SCHEMA=rag_qa
EVALUATION_DATABASE_SCHEMA=rag_qa_evaluation

OPENAI_API_KEY=<your-api-key>
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 使用其他 OpenAI 兼容服务

```dotenv
LLM_API_KEY=<your-api-key>
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model

EMBEDDING_API_KEY=<your-api-key>
EMBEDDING_BASE_URL=https://your-provider.example/v1
EMBEDDING_MODEL=your-embedding-model

# 可选：重排模型（未配置 RERANKER_BASE_URL 时默认检索策略为 hybrid）
RERANKER_API_KEY=<your-api-key>
RERANKER_BASE_URL=https://your-provider.example/v1
RERANKER_MODEL=qwen3-rerank

# 可选：单独为“问题理解”配置更快更便宜的模型，留空时沿用生成模型
UNDERSTANDING_MODEL=your-small-model
```

### 常用配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `KNOWLEDGE_RETRIEVAL_PROFILE` | `auto` | 检索策略：`auto` / `vector` / `hybrid` / `hybrid_rerank` |
| `TOP_K` | `5` | 每次检索最终交给模型的证据条数（按父块计，1–20） |
| `TOP_K_RERANK` | `20` | 进入重排的候选子块数 |
| `CHUNK_SIZE` | `500` | 子块（检索单位）字数上限；同一父块内的子块不重叠 |
| `PARENT_CHUNK_SIZE` | `1500` | 父块（交给模型的上下文单位）字数上限，不跨章节、不跨页；设成与 `CHUNK_SIZE` 相同即关闭父块扩展 |
| `CHUNK_OVERLAP` | `50` | 同一章节内相邻父块之间的重叠字数 |
| `MAX_UPLOAD_SIZE` | `20971520` | 单个上传文件上限（字节，默认 20 MB） |
| `AGENT_MAX_RETRIEVAL_CALLS` | `3` | Agent 通道单轮最多检索次数 |
| `AGENT_TURN_TIMEOUT_SECONDS` | `180` | 单轮问答总时限 |
| `QUESTION_RATE_LIMIT_PER_MINUTE` | `10` | 每个用户每分钟最多提问数 |
| `QUESTION_MAX_ACTIVE` | `3` | 每个用户同时进行中的提问数 |
| `AUTH_SESSION_SECONDS` | `43200` | 登录会话最长有效期（12 小时） |
| `AUTH_IDLE_SECONDS` | `1800` | 空闲多久后需要重新登录 |
| `AUTH_SECURE_COOKIE` | `true` | 纯 HTTP 访问时必须设为 `false`，否则浏览器不保存登录 Cookie |
| `HTTP_ALLOWED_HOSTS` | `["localhost","127.0.0.1","[::1]"]` | 允许的 Host 头，公网部署时加上域名 |
| `TRACE_RETENTION_DAYS` | `30` | 追踪记录保留天数，用量统计只覆盖保留期内的数据 |
| `TRACE_MODEL_PRICES` | 空 | 模型费率，用于“用量与费用”估算（见下） |

### 配置模型费率

“用量与费用”页面的金额按这里的费率估算。`origin` 是模型服务地址去掉路径后的部分，币种不能混用：

```dotenv
TRACE_MODEL_PRICES='[{"origin":"https://api.openai.com","model":"gpt-4o-mini","version":"2026-11","currency":"USD","input_per_million":0.15,"output_per_million":0.6}]'
```

请以服务商的最新价格为准。未配置费率的模型只统计 Token，不计金额。

### 升级到父子分块

文档现在按章节切成父块（交给模型的上下文）和子块（检索单位），子块带“文件名 › 标题路径”前缀建索引。升级后新上传的文档自动使用新规则；之前入库的文档照常可查，但没有标题和父块上下文，需要用命令行重建一次：

```bash
# 1. 试运行：列出待重建的文档，不读源文件、不调用模型、不写入
uv run python -m app.interfaces.cli.reindex --dry-run

# 2. 按知识库分批执行；--limit 限制本次处理的篇数
uv run python -m app.interfaces.cli.reindex --kb default --limit 20
```

- 默认只处理还是旧格式的文档，重复执行会接着处理剩下的；加 `--all` 会把已是新格式的文档也重建一遍（例如调整了 `CHUNK_SIZE` / `PARENT_CHUNK_SIZE` 之后）。
- 重建用的是 `data/docs` 里保存的源文件，会重新调用 Embedding 服务并产生相应费用。每篇文档先在内存里完成解析和向量化，再一次性替换文本块，替换之前旧版本照常可查；源文件缺失、向量化失败或期间文档被删除时，这篇文档保持原样并在输出中注明。
- 重建不经过入库队列，管理页上看不到进度，进度和结果以命令行输出为准。

---

## Docker 部署

`docker-compose.yml` 包含四个服务：

| 服务 | 作用 |
| --- | --- |
| `db` | `pgvector/pgvector:pg18`，只监听本机 5433 端口 |
| `migrate` | 一次性执行数据库迁移，完成后退出 |
| `app` | FastAPI 后端，只监听本机 8000 端口 |
| `web` | nginx 托管前端并反向代理 `/api`，对外暴露 80 端口 |

在 `.env` 中额外设置：

```dotenv
POSTGRES_PASSWORD=replace-with-strong-password
PUBLIC_HOST=your-domain.example       # 浏览器访问用的域名或 IP，本机试跑可不填
AUTH_SECURE_COOKIE=false              # 仅当用纯 HTTP 访问公网 IP 时
```

启动并创建管理员：

```bash
docker compose up -d --build
docker compose exec app python -m app.interfaces.cli.users create --username admin --name 管理员 --role admin
```

`.env` 以可写方式挂载进 `app` 容器，管理后台修改的模型配置会写回宿主机，重建容器后仍然有效。上传的原始文件保存在 `app_data` 卷中。

---

## 测试与评估

### 后端测试

测试使用独立的测试数据库，每个测试在其中新建随机 schema，结束后清理。在 `.env` 中设置 `TEST_DATABASE_URL`，指向另一个已启用 pgvector 的数据库（**不要**指向应用数据库）：

```bash
uv sync --group eval
uv run --group eval pytest
```

### 前端测试

```bash
cd frontend
npm test            # vitest
npm run typecheck   # tsc --noEmit
npm run build
```

### 密钥检查

提交前检查是否误提交了密钥：

```bash
uv run python scripts/check_secrets.py
```

### RAG 效果评估

`evaluation/` 提供可恢复的离线评估，业务数据与评估数据存放在不同 schema：

```bash
# 固定语料 + QA 集，对比三种检索策略，并跑 RGB 中文鲁棒性用例
uv run --group eval python -m evaluation.run_eval --full

# 分通道（快速 / Agent）小样本评估；先用 --dry-run 预检，不计费
uv run --group eval python -m evaluation.lanes --dry-run
uv run --group eval python -m evaluation.lanes --full
```

评估报告也可以在后端的评估报告接口中查看。评估会调用模型并产生费用，正式运行前请先预检。

---

## 项目结构

```text
RAG_QA/
├── app/
│   ├── domain/              # 领域模型与规则
│   ├── application/         # 用例：query / retrieval / ingestion / knowledge / insights / tracing ...
│   │   └── ports/           # 仓储与模型端口
│   ├── infrastructure/      # PostgreSQL、解析、分块、Embedding/LLM/Rerank、LangGraph Agent
│   │   └── persistence/     # 仓储实现与 schema*.sql 迁移
│   ├── interfaces/
│   │   ├── http/routes/     # REST + SSE 接口
│   │   └── cli/             # database（迁移）、users（用户管理）
│   ├── bootstrap/           # 组合根与配置存储
│   └── config.py            # 全部配置项
├── frontend/
│   └── src/
│       ├── api/             # live（真实后端）与 mock（演示数据）两套实现
│       ├── pages/           # 各页面
│       └── components/      # 聊天、知识库面板、布局等组件
├── evaluation/              # 离线评估：数据集、指标、裁判模型、报告
├── tests/                   # 后端测试（含架构依赖测试）
├── scripts/check_secrets.py # 密钥扫描
├── docker/ · Dockerfile · docker-compose.yml
└── docs/images/             # README 截图
```

---

## 安全提示

- 不要提交 `.env`、API 密钥、私有证书、数据库文件和本地运行产物；提交前运行 `scripts/check_secrets.py`。
- 公开部署前配置 `HTTP_ALLOWED_HOSTS`、CORS 来源、HTTPS，并保持 `AUTH_SECURE_COOKIE=true`。
- 全新数据库没有默认账号，管理员只能通过命令行创建。
- 登录与修改口令都有尝试次数限制；口令只保存哈希。
- 生产环境使用独立数据库账号和密钥管理服务，测试库与应用库分开。

## License

当前仓库未附带许可证文件。如需允许他人复制、修改或分发代码，请先添加合适的开源许可证。
