CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('member', 'admin')),
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    csrf_hash TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    last_seen_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS login_buckets (
    bucket_key TEXT PRIMARY KEY,
    window_start INTEGER NOT NULL,
    attempts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'deleting')),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    turns_count INTEGER NOT NULL DEFAULT 0 CHECK(turns_count >= 0)
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    sequence_id BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    answer_mode TEXT NOT NULL CHECK(answer_mode IN ('direct', 'retrieval')),
    retrieval_calls INTEGER NOT NULL CHECK(retrieval_calls BETWEEN 0 AND 3),
    latency_ms DOUBLE PRECISION NOT NULL CHECK(latency_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS turn_citations (
    sequence_id BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    turn_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK(position >= 1),
    doc_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    page INTEGER,
    chunk_text TEXT NOT NULL,
    PRIMARY KEY(turn_id, position),
    FOREIGN KEY(turn_id) REFERENCES conversation_turns(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS knowledge_bases (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        icon TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    );

CREATE TABLE IF NOT EXISTS ingestion_jobs (
                        job_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL UNIQUE,
                        kb_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                        uploaded_by TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN (
                            'queued', 'running', 'succeeded', 'failed'
                        )),
                        stage TEXT NOT NULL CHECK(stage IN (
                            'parsing', 'chunking', 'embedding', 'indexing'
                        )),
                        progress DOUBLE PRECISION NOT NULL CHECK(
                            progress >= 0 AND progress <= 1
                        ),
                        completed_batches INTEGER NOT NULL DEFAULT 0,
                        total_batches INTEGER NOT NULL DEFAULT 0,
                        chunks_count INTEGER,
                        error_code TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        lease_token TEXT,
                        lease_expires_at TIMESTAMPTZ,
                        FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id)
                            ON DELETE RESTRICT
                    );

CREATE TABLE IF NOT EXISTS document_deletions (
                    doc_id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, recorded_at TIMESTAMPTZ NOT NULL
                );

CREATE TABLE IF NOT EXISTS curated_questions (
                    id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, text TEXT NOT NULL,
                    sort_order INTEGER NOT NULL CHECK (sort_order >= 0),
                    UNIQUE(kb_id, text), UNIQUE(kb_id, sort_order),
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
                );

CREATE TABLE IF NOT EXISTS knowledge_deletion_plans (
                kb_id TEXT PRIMARY KEY, plan_json TEXT NOT NULL,
                FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id) ON DELETE RESTRICT
            );

CREATE TABLE IF NOT EXISTS traces (
                        trace_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        expires_at TIMESTAMPTZ NOT NULL,
                        snapshot_json TEXT NOT NULL
                    );

CREATE TABLE IF NOT EXISTS admin_access_audit (
                        id TEXT PRIMARY KEY,
                        expires_at TIMESTAMPTZ NOT NULL,
                        event_json TEXT NOT NULL
                    );

CREATE TABLE IF NOT EXISTS turn_shares (
                token_hash TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                shared_by TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                FOREIGN KEY(turn_id) REFERENCES conversation_turns(id)
                    ON DELETE CASCADE
            );

ALTER TABLE conversations ADD COLUMN IF NOT EXISTS kb_scope_json TEXT;

ALTER TABLE conversations ADD COLUMN IF NOT EXISTS owner_id TEXT;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS metadata_json TEXT;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS feedback_json TEXT;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS trace_id TEXT;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS feedback_revision INTEGER NOT NULL DEFAULT 0;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS feedback_updated_at TIMESTAMPTZ;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS feedback_triage TEXT;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS feedback_triaged_by TEXT;

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS feedback_triaged_at TIMESTAMPTZ;

ALTER TABLE turn_citations ADD COLUMN IF NOT EXISTS chunk_id TEXT;

ALTER TABLE turn_citations ADD COLUMN IF NOT EXISTS kb_id TEXT DEFAULT 'default';

ALTER TABLE turn_citations ADD COLUMN IF NOT EXISTS doc_updated_at TEXT;

ALTER TABLE turn_citations ADD COLUMN IF NOT EXISTS citation_order INTEGER;

ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS owner_name TEXT;

ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS lifecycle TEXT NOT NULL DEFAULT 'active' CHECK(lifecycle IN ('active','deleting','deleted'));

CREATE INDEX IF NOT EXISTS auth_sessions_expiry ON auth_sessions(expires_at);

CREATE INDEX IF NOT EXISTS login_buckets_window ON login_buckets(window_start);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation
    ON conversation_turns(conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_turns_trace ON conversation_turns(trace_id);

CREATE INDEX IF NOT EXISTS idx_citations_document_chunk ON turn_citations(doc_id, chunk_id, turn_id);

CREATE INDEX IF NOT EXISTS idx_turns_feedback_time ON conversation_turns(COALESCE(feedback_updated_at, created_at), id) WHERE feedback_json IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id, status, updated_at);

CREATE INDEX IF NOT EXISTS ingestion_jobs_pending
                    ON ingestion_jobs(status, lease_expires_at, created_at);

CREATE INDEX IF NOT EXISTS traces_created
                        ON traces(created_at, trace_id);

CREATE INDEX IF NOT EXISTS traces_expiry
                        ON traces(expires_at);

CREATE INDEX IF NOT EXISTS admin_access_audit_expiry
                        ON admin_access_audit(expires_at);

CREATE INDEX IF NOT EXISTS idx_shares_turn
                ON turn_shares(turn_id);

INSERT INTO knowledge_bases (id,name,description,created_at,updated_at) VALUES ('default','默认知识库','默认上传文档',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING;

CREATE TABLE vector_collections (
    name TEXT PRIMARY KEY, dimensions INTEGER, metadata JSONB NOT NULL DEFAULT '{}'
);
CREATE TABLE vector_chunks (
    collection TEXT NOT NULL REFERENCES vector_collections(name) ON DELETE CASCADE,
    id TEXT NOT NULL, doc_id TEXT NOT NULL, kb_id TEXT NOT NULL,
    metadata JSONB NOT NULL, text TEXT NOT NULL, embedding vector NOT NULL,
    PRIMARY KEY(collection,id)
);
CREATE INDEX vector_chunks_document ON vector_chunks(collection, doc_id);
CREATE INDEX vector_chunks_scope ON vector_chunks(collection, kb_id);
