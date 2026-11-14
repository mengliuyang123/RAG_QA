# PostgreSQL 全量替换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在用户配置的全新 PostgreSQL 数据库中运行所有业务、向量、检查点和评估存储。

**Architecture:** 保持领域端口与 HTTP 契约，以 PostgreSQL 仓储替换 SQLite/Chroma。共享连接池由容器管理，建表由显式版本迁移管理；评估及测试严格隔离。

**Tech Stack:** psycopg 3、psycopg_pool、pgvector、langgraph-checkpoint-postgres、pytest。

**Spec:** `docs/postgresql-design.md`。

## Global Constraints

- 全新空库，不迁移、不删除旧数据；不实现长期双后端或双写。
- 保留现有用户未提交修改，不代用户提交整个工作区。
- 2560 维完整向量、cosine 精确检索；知识库和发布状态过滤必须先于 Top-K。
- 凭据仅经环境变量/.env 读取，不打印 DSN；不将超级用户权限作为正常运行的必要条件。
- 评估 JSON/JSONL、原始上传文件及日志保留为文件；BM25 从 PostgreSQL 文本块构建。
- 所有默认生产及评估装配切换；最终移除 SQLite/Chroma 依赖和直接实例化测试。

## Review Focus

1. 测试 DSN 误指开发库：初始化前拒绝，不能 DROP 用户数据。
2. 队列满时并发入队：容量不超卖；删除与入队/发布使用相同知识库锁协议。
3. 空知识库范围、未发布文档和错误向量维度：不越范围召回，不污染集合。
4. 请求取消或服务关闭：事务回滚、连接归还，旧 worker 不得提交。
5. 测试或评估并行运行：checkpoint、向量、运行记录互不串读。

## Task 1：数据库基础、迁移与测试隔离

**Files:** 新增 `app/infrastructure/persistence/database.py`、`app/infrastructure/persistence/migrations.py`、`app/interfaces/cli/database.py`、`tests/test_postgres_database.py`、`tests/postgres_support.py`；修改 `app/config.py`、`tests/conftest.py`、`pytest.ini`、`pyproject.toml`、`uv.lock`、`requirements.txt`。

**Interfaces:** `Database(dsn: str, schema: str)` 持有连接池；`connection(*, read_only: bool = False)` 提供事务上下文；`close() -> None` 幂等关闭。`migrate_database(dsn: str, schema: str) -> None` 在串行迁移锁内执行带版本记录的迁移。SQL 标识符使用 psycopg.sql.Identifier。

- [x] 写失败测试：重复迁移不重复建表、异常写事务回滚、关闭两次不报错、测试 DSN 与 DATABASE_URL 相同时拒绝初始化。
- [x] 运行 `uv run pytest tests/test_postgres_database.py -q`，确认因缺少新实现失败。
- [x] 引入依赖，实现配置校验、连接池、迁移命令、测试数据库安全检查与 worker schema 隔离。缺少 TEST_DATABASE_URL 时提供明确配置错误，不能静默访问应用库。
- [x] 独立 `database` marker 只放行测试连接目标；保留外部模型网络禁止规则。数据库测试不能借用全面放行网络的 integration marker。
- [x] 重跑该模块；在测试库验证迁移与清理，检查输出不包含凭据。

## Task 2：认证与知识库基础仓储

**Files:** 新增 `app/infrastructure/auth/postgres_repository.py`；将 `app/infrastructure/persistence/knowledge_bases.py`、`knowledge_registry.py`、`knowledge_deletion.py` 的 SQL 实现改为 PostgreSQL；修改对应认证/知识目录/删除/建议问题仓储测试及 Task 1 迁移定义。

**Interfaces:** `PostgresAuthRepository(database: Database)`、`PostgresKnowledgeBaseRepository(database: Database)`、`PostgresKnowledgeReadRegistry(database: Database)`、`PostgresKnowledgeDeletionRepository(database: Database)`；公开方法保持对应既有端口签名。

- [x] 将认证持久化、并发登录限流、空目录与默认库、删除计划幂等、建议问题覆盖替换测试参数改为隔离 Database，先观察失败。
- [x] 使用 PostgreSQL 参数化查询与唯一约束实现；限流桶的检查和更新必须在同一锁定事务中完成，不能只转换占位符。
- [x] 定义共享知识库行锁顺序，冻结、恢复、入队均遵循该协议；不在仓储构造器内执行任意 DDL。
- [x] 运行上述仓储及认证服务测试，确认角色、密码哈希和权限契约保留。

## Task 3：向量、文本内容及元数据

**Files:** 新增 `app/infrastructure/vectorstores/postgres.py`；修改 `app/infrastructure/vectorstores/knowledge_reader.py` 或新增 PostgreSQL 内容读取实现；修改 `tests/test_vectorstore.py`、`tests/test_scoped_retrieval.py`、知识目录/浏览/文档块测试与迁移定义。

**Interfaces:** `PostgresVectorStore(database: Database, collection_name: str, collection_metadata: Mapping[str, str | int] | None = None)` 覆盖现有 VectorStoreAdapter 的所有能力，包括知识内容读取；既有 DocumentChunk 与 VectorSearchResult 不变。

- [x] 写/迁移失败测试：真实 2560 维向量 cosine 排序；空范围为空；范围外高分不能挤占范围内结果；未发布/排除文档不入选；维度不匹配拒绝；重复块与事务失败不留下半批数据。
- [x] 实现 collection 身份和维度校验、向量适配注册、批量写入和精确 SQL 检索，所有过滤条件先于 ORDER BY/LIMIT。
- [x] 覆盖文档列表、文本块、版本时间、计数、元数据合并及精确集合删除；BM25 原算法复用。
- [x] 运行向量、知识内容和引用元数据测试；不得用 Chroma 测试通过替代 PostgreSQL 验证。

## Task 4：持久任务与跨进程执行保护

**Files:** 修改 `app/infrastructure/persistence/ingestion_jobs.py`、`ingestion_execution.py`、`knowledge_deletion.py`；修改 `tests/test_ingestion_job_repository.py`、`tests/test_ingestion_execution_guard.py`、`tests/test_ingestion_publication.py`、`tests/test_knowledge_base_deletion_api.py` 及迁移定义。

**Interfaces:** `PostgresIngestionJobRepository(database: Database)` 保留任务端口；`PostgresIngestionExecutionGuard(database: Database, collection_name: str)` 保留执行保护端口；`has_pending_ingestion_jobs(database: Database) -> bool`。

- [x] 写失败并发测试：容量为一时两客户端提交仅一成功；两 worker 不领取同一任务；旧租约不得发布；知识库冻结后不能入队或发布。
- [x] 采用事务容量控制锁、统一知识库行锁、FOR UPDATE SKIP LOCKED 和租约栅栏；明确锁顺序并测试失败事务释放锁。
- [x] 保留文件写入补偿、任务状态错误映射和取消语义，不将跨文件系统操作伪装为单库事务。
- [x] 运行该组测试，增加两个独立连接/进程的锁竞争验证。

## Task 5：会话与 LangGraph 检查点

**Files:** 新增 `app/infrastructure/conversations/postgres_repository.py`；修改 `app/infrastructure/conversations/checkpoint.py`、`pagination.py`；修改会话、反馈、取消和检查点相关测试；扩展迁移定义。

**Interfaces:** `PostgresConversationRepository(database: Database)` 保留 ConversationRepository 及反馈接口；`PostgresCheckpointStore(database: Database)` 暴露 `checkpointer`、`delete_thread(UUID)`、`close()`。

- [x] 迁移失败测试：分页无重复遗漏、其他用户不可读写、删除状态不可提交、两轮保存后重开仓储可读、删除线程同时清除 pending writes。
- [x] PostgreSQL 会话和引用事务实现保留正式保存/取消边界；键集分页按既有排序规则实现。
- [x] 接入官方 PostgresSaver，按官方连接要求配置 autocommit/row_factory/prepare_threshold，初始化迁移中调用 setup；测试 schema/search_path 必须与业务隔离一致。
- [x] 运行会话/认证 API、检查点和取消测试，验证连接归还和关闭幂等。

## Task 6：追踪、分享及管理读模型

**Files:** 修改 `app/infrastructure/persistence/traces.py`、`trace_queries.py`、`document_citations.py`、`feedback_admin.py`、`insights.py`、`shares.py`；修改相应追踪、成本、引用统计、反馈、洞察和分享测试，扩展迁移定义。

**Interfaces:** 各 `SQLite*` 仓储替换为对应 `Postgres*` 类，均接收 Database；跨库 ATTACH 查询改为同一 schema 内的关系查询，保持端口返回类型。

- [x] 将既有数据校验、保留期、模型用量、commit 关联、引用去重与用户可见性测试迁入真实 PostgreSQL，观察失败。
- [x] 将 JSON 查询、聚合、参数、时间和 upsert 改为 PostgreSQL 语义；追踪失败不得破坏已经提交的业务结果。
- [x] 核查 read_only 事务和快照一致性；分享 token 仍保存摘要，管理读权限仍由后端强制执行。
- [x] 运行上述管理与业务 API 测试，检查错误响应不泄漏 SQL/凭据。

## Task 7：评估存储与默认装配切换

**Files:** 修改 `evaluation/runtime.py`、`evaluation/lanes/runtime.py`、`evaluation/lanes/checkpoints.py`、`evaluation/lanes/cli.py`、`app/bootstrap/container.py`、`app/bootstrap/factories.py`、`app/bootstrap/trace_configuration.py`、`app/config.py` 及所有使用存储路径覆盖的评估/容器测试。

**Interfaces:** RuntimeOptions 用数据库 schema/collection 隔离取代路径隔离；LaneRunStore 接收 Database 与运行身份，保留 save/load/save_score/load_score/prune_expired 的行为。默认 factories 绑定容器拥有的 Database，测试通过显式 factories 替换，无隐式全局连接。

- [x] 写失败测试：两个评估运行相同 case ID 不串数据；manifest 不匹配不复用；评估不能误删应用集合；两个容器关闭互不影响。
- [x] 替换评估向量、lane 记录和恢复存储，保留 JSON/JSONL 报告产物；更新索引身份与配置指纹。
- [x] 切换所有默认工厂与容器资源所有权；清除所有 SQLite 路径/Chroma 路径相关装配，移除旧适配器及依赖，更新锁文件。
- [x] 搜索 `sqlite3|chromadb|checkpoint.sqlite|SQLite|Chroma`，逐一处理运行与测试引用；历史文档提及与旧数据说明可保留。
- [x] 运行容器、配置、评估恢复及架构契约测试。

## Task 8：全链路交付

**Files:** 更新 `README.md`、`docs/quickstart.md`、`docs/postgresql-design.md`；测试验证产物置于被忽略的 `data/repair/`，不写入凭据。

- [x] 执行 `uv run pytest -q`，不得通过跳过原持久化测试掩盖遗漏；报告任何缺失的测试环境条件。
- [x] 执行 `npm.cmd test`、`npm.cmd run build -- --mode live`（frontend 工作目录）。
- [x] 在新应用库执行显式迁移、创建临时测试账号和知识库；验证登录、文档上传、2560 维向量入库、重排序、带引用 SSE、非流式追问、会话保存。
- [x] 重启服务后复核持久化、删除恢复及评估记录；确认原 SQLite/Chroma 文件未被新运行修改。
- [x] 精确清理本次临时数据，保留用户原始文件；审查 diff，仅保留任务所需变更。
- [x] 最终独立代码审查，修复阻塞问题并重跑受影响验证；记录实际测试结果和服务地址。

## 执行建议

建议在当前会话按任务顺序直接实施，因各仓储共享 schema、锁协议和组合根接口；最终由独立审查者统一复核。实施前请用户复核本计划；不自动提交用户已有工作区改动。

实施完成：全量回归、真实模型链路及重启持久化均通过。详细结果见 docs/postgresql-verification.md。
