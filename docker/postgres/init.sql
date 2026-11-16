-- 仅在数据卷首次初始化时执行（postgres 镜像的 docker-entrypoint-initdb.d 约定）。
-- 应用库启用 pgvector；schema 和表由应用启动时的迁移创建。
CREATE EXTENSION IF NOT EXISTS vector;

-- 独立测试库（TEST_DATABASE_URL），同样预先启用 pgvector。
CREATE DATABASE rag_qa_postgres_test;
\connect rag_qa_postgres_test
CREATE EXTENSION IF NOT EXISTS vector;
