-- Observability event warehouse tables (spec sections 5-6).
-- Adds agent_spans, outcomes, evaluation_scores and enriches sessions.

-- Enrich sessions with observability columns
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS channel   VARCHAR(50);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS intent    VARCHAR(50);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS escalated BOOLEAN;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS csat      DOUBLE PRECISION;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS end_time  TIMESTAMPTZ;

-- Agent spans: one row per agent action within a conversation trace
CREATE TABLE IF NOT EXISTS agent_spans (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id   VARCHAR(100) NOT NULL,
    span_name  VARCHAR(100) NOT NULL,
    attributes JSONB,
    latency_ms DOUBLE PRECISION,
    timestamp  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_agent_spans_trace_id ON agent_spans (trace_id);

-- Task outcomes per session
CREATE TABLE IF NOT EXISTS outcomes (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(100) NOT NULL,
    task       VARCHAR(200) NOT NULL,
    completed  BOOLEAN NOT NULL DEFAULT FALSE,
    verified   BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS ix_outcomes_session_id ON outcomes (session_id);

-- LLM-evaluated quality scores
CREATE TABLE IF NOT EXISTS evaluation_scores (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id        VARCHAR(100) NOT NULL,
    groundedness      DOUBLE PRECISION,
    hallucination     BOOLEAN,
    helpfulness       DOUBLE PRECISION,
    policy_compliance DOUBLE PRECISION,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_evaluation_scores_session_id ON evaluation_scores (session_id);
