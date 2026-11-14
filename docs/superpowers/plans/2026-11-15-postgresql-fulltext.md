# PostgreSQL Full Text Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 用 PostgreSQL 全文检索取代默认运行时内存 BM25，支持中文、旧数据回填、多进程一致性。
**Architecture:** 固定版本 Jieba 搜索模式生成 search_text；数据库生成 search_vector 并维护 GIN；直接查询已提交记录，无进程级索引副本。
**Tech Stack:** Jieba 0.42.1、psycopg 3、PostgreSQL tsvector/tsquery。
**Spec:** docs/postgresql-fulltext-design.md（用户已批准）。

## Global Constraints
只改后端及文档、测试；保留用户修改；不提交、不额外启动常驻项目服务；迁移时重启既有后端以恢复原运行状态；迁移不重算向量。
## Review Focus
空/标点查询不能成为 tsquery 操作符；中文短语不降级为单字；限定集合/知识库在 Top-K 前生效；迁移失败原子回滚；独立进程提交/删除无需重启索引。

### Task 1: 分词、数据库迁移及原子写入
Files: app/infrastructure/indexes/tokenization.py、app/infrastructure/persistence/schema_v4.sql、migrations.py、app/infrastructure/vectorstores/postgres.py、pyproject.toml、uv.lock、requirements.txt。
- [x] 写 tests/test_postgres_fulltext.py，观察缺少全文列/中文检索的失败。
- [x] 加 Jieba 固定版本；共用 tokenize_search(text)；文档写入 search_text；迁移 v4 分批回填（同事务、表锁、失败回滚），生成列和 GIN。
- [x] 验证真实数据库旧记录回填、失败回滚、索引结构和新记录事务原子性。

### Task 2: 查询和默认装配
Files: app/infrastructure/indexes/postgres.py、app/bootstrap/factories.py、端口文档、后端测试。
- [x] 先验证中文词语、OR 查询、英文标识符/数字、特殊字符、集合/KB/未发布范围及排序失败。
- [x] PostgresKeywordIndex(database,collection_name) 使用绑定参数、simple 配置及 ts_rank_cd；rebuild 不再持有或复制快照。
- [x] 工厂改用 PostgreSQL 实现，兼容既有 KeywordIndex 端口；更新默认装配测试。
- [x] 真实子进程新增/删除后，已创建检索器立即可见；回滚记录不可见。

### Task 3: 回归、部署和交接
- [x] 运行关键词/向量/入库/容器/混合检索/评估测试，然后后端全量回归。
- [x] 一次独立审查；修复有依据的问题并验证。
- [x] 执行现有应用和评估 schema 的显式迁移并检查回填/索引状态，不启动服务。
- [x] README/quickstart/设计记录更新，记录分数变化、Jieba 版本、直接 SQL 写入边界及验证证据。

完成记录：docs/postgresql-fulltext-verification.md。
