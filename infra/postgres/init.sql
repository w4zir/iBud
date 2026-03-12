-- infra/postgres/init.sql — pgvector + core tables (Phase 0/1)
-- EMBEDDING_DIM: 768 for nomic-embed-text (Ollama), 1536 for OpenAI text-embedding-3-small

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    parent_id   UUID REFERENCES documents(id),
    embedding   VECTOR(768),
    company_id  VARCHAR(100),
    source      VARCHAR(50),
    doc_tier    INTEGER DEFAULT 1,
    category    VARCHAR(100),
    source_id   VARCHAR(200),
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON documents (doc_tier);
CREATE INDEX ON documents (source);
CREATE INDEX ON documents (company_id);

CREATE TABLE sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     VARCHAR(100),
    company_id  VARCHAR(100),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    role        VARCHAR(20) NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_number    VARCHAR(50) UNIQUE NOT NULL,
    user_id         VARCHAR(100),
    company_id      VARCHAR(100),
    status          VARCHAR(50),
    items           JSONB,
    total_amount    DECIMAL(10,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    estimated_delivery TIMESTAMPTZ
);

CREATE TABLE tickets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    issue_type  VARCHAR(100),
    summary     TEXT,
    status      VARCHAR(50) DEFAULT 'open',
    priority    VARCHAR(20) DEFAULT 'normal',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
