-- 知识库访问控制：默认公开（与旧行为一致）；受限库只有成员和管理员可见。
CREATE TABLE knowledge_base_access (
    kb_id TEXT PRIMARY KEY REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    restricted BOOLEAN NOT NULL DEFAULT FALSE
);
-- viewer 可查看和提问；editor 还可以管理该库的文档、描述、推荐问题和标准答案。
CREATE TABLE knowledge_base_members (
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('viewer', 'editor')),
    PRIMARY KEY (kb_id, user_id)
);
CREATE INDEX knowledge_base_members_user ON knowledge_base_members(user_id);

-- 文档时效：复审日期与"已过时"标记。已过时的文档不再参与检索。
CREATE TABLE document_reviews (
    doc_id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    review_due_on DATE,
    outdated BOOLEAN NOT NULL DEFAULT FALSE,
    note TEXT CHECK (note IS NULL OR length(note) <= 500),
    updated_at TIMESTAMPTZ NOT NULL,
    updated_by TEXT NOT NULL
);
CREATE INDEX document_reviews_kb ON document_reviews(kb_id);

-- 标准答案：发布为知识库中的「常见问题」文档，由正常检索与引用链路使用。
CREATE TABLE faq_entries (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    question TEXT NOT NULL CHECK (length(question) BETWEEN 1 AND 500),
    answer TEXT NOT NULL CHECK (length(answer) BETWEEN 1 AND 5000),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    updated_by TEXT NOT NULL
);
CREATE INDEX faq_entries_kb ON faq_entries(kb_id, created_at);
