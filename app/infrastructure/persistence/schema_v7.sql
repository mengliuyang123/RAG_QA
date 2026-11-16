-- 知识缺口工作台：只保存管理员处理状态，缺口来源仍是拒答聚类与点踩反馈。
CREATE TABLE knowledge_gaps (
    gap_key TEXT PRIMARY KEY CHECK (gap_key ~ '^(cluster|feedback):[A-Za-z0-9_-]{1,128}$'),
    source TEXT NOT NULL CHECK (source IN ('refusal_cluster', 'feedback')),
    title TEXT NOT NULL CHECK (length(title) BETWEEN 1 AND 8000),
    questions TEXT NOT NULL,
    kb_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('open', 'document_added', 'verified')),
    doc_id TEXT,
    last_check TEXT,
    updated_at TIMESTAMPTZ NOT NULL,
    updated_by TEXT NOT NULL
);
CREATE INDEX knowledge_gaps_updated ON knowledge_gaps(updated_at DESC);
