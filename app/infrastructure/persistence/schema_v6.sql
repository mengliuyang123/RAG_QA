-- 文档登记：内容摘要查重、同名版本替换与按 doc_id 删除的权威来源。
-- 旧文档不回填；部分唯一索引只约束未删除的文档。
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE RESTRICT,
    filename TEXT NOT NULL CHECK (length(filename) BETWEEN 1 AND 255),
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'failed')),
    replaces_doc_id TEXT,
    uploaded_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX documents_live_kb_sha256 ON documents(kb_id, sha256) WHERE deleted_at IS NULL;
CREATE INDEX documents_live_kb_filename ON documents(kb_id, filename, uploaded_at) WHERE deleted_at IS NULL;
CREATE INDEX vector_chunks_doc_id ON vector_chunks(doc_id);
