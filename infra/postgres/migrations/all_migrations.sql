-- =============================================================================
-- Combined Postgres migrations (001 through 006)
-- =============================================================================
-- Apply to a database that already has the base schema from infra/postgres/init.sql
-- (or an older snapshot). Safe to re-run where statements use IF NOT EXISTS /
-- idempotent patterns; migration 002 may reset embedding column values when
-- resizing dimensions.
--
-- Individual scripts are kept alongside this file for granular use:
--   001_add_documents_parent_id.sql
--   002_documents_embedding_768.sql
--   003_observability_warehouse.sql
--   004_analytics_views.sql
--   005_intent_eval.sql
--   006_add_company_id.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 001_add_documents_parent_id.sql
-- ---------------------------------------------------------------------------

-- Add any missing columns to documents (for DBs created before full schema)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'parent_id') THEN
    ALTER TABLE documents ADD COLUMN parent_id UUID REFERENCES documents(id) ON DELETE SET NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'source') THEN
    ALTER TABLE documents ADD COLUMN source VARCHAR(50);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'doc_tier') THEN
    ALTER TABLE documents ADD COLUMN doc_tier INTEGER DEFAULT 1;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'category') THEN
    ALTER TABLE documents ADD COLUMN category VARCHAR(100);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'source_id') THEN
    ALTER TABLE documents ADD COLUMN source_id VARCHAR(200);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'metadata') THEN
    ALTER TABLE documents ADD COLUMN metadata JSONB;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'created_at') THEN
    ALTER TABLE documents ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();
  END IF;
END $$;

-- Ensure useful indexes exist (idempotent: CREATE INDEX IF NOT EXISTS in PG 9.5+)
CREATE INDEX IF NOT EXISTS idx_documents_doc_tier ON documents (doc_tier);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);

-- ---------------------------------------------------------------------------
-- 002_documents_embedding_768.sql
-- ---------------------------------------------------------------------------

-- Align embedding column with 768-dim (nomic-embed-text).
-- Drop vector index, resize column (existing values set to NULL), recreate index.
DO $$
DECLARE
  rec RECORD;
BEGIN
  DROP INDEX IF EXISTS documents_embedding_idx;
  DROP INDEX IF EXISTS documents_embedding_vector_cosine_ops_idx;
  FOR rec IN (SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'documents' AND indexdef LIKE '%embedding%')
  LOOP
    EXECUTE format('DROP INDEX IF EXISTS %I', rec.indexname);
  END LOOP;
END $$;
ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768) USING NULL;
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- 003_observability_warehouse.sql
-- ---------------------------------------------------------------------------

-- Phase 5: observability warehouse schema.
-- Note: for high-volume deployments, partition agent_spans by month on "timestamp".

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'intent'
  ) THEN
    ALTER TABLE sessions ADD COLUMN intent VARCHAR(50);
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'escalated'
  ) THEN
    ALTER TABLE sessions ADD COLUMN escalated BOOLEAN DEFAULT FALSE;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'resolved_at'
  ) THEN
    ALTER TABLE sessions ADD COLUMN resolved_at TIMESTAMPTZ;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS agent_spans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
  trace_id VARCHAR(100),
  span_name VARCHAR(100) NOT NULL,
  attributes JSONB,
  latency_ms NUMERIC(12, 3),
  "timestamp" TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS outcomes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  task VARCHAR(100) NOT NULL,
  completed BOOLEAN NOT NULL,
  escalated BOOLEAN NOT NULL DEFAULT FALSE,
  verified BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_scores (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  groundedness DOUBLE PRECISION,
  hallucination BOOLEAN,
  helpfulness DOUBLE PRECISION,
  metadata JSONB,
  evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_spans_session_id ON agent_spans (session_id);
CREATE INDEX IF NOT EXISTS idx_agent_spans_timestamp ON agent_spans ("timestamp");
CREATE INDEX IF NOT EXISTS idx_outcomes_session_id ON outcomes (session_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_created_at ON outcomes (created_at);
CREATE INDEX IF NOT EXISTS idx_eval_scores_session_id ON evaluation_scores (session_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_evaluated_at ON evaluation_scores (evaluated_at);

-- ---------------------------------------------------------------------------
-- 004_analytics_views.sql
-- ---------------------------------------------------------------------------

-- Phase 6: business/product metric query layer.
-- CSAT/NPS placeholders until external survey source is integrated.

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'csat_score'
  ) THEN
    ALTER TABLE sessions ADD COLUMN csat_score SMALLINT;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'nps_score'
  ) THEN
    ALTER TABLE sessions ADD COLUMN nps_score SMALLINT;
  END IF;
END $$;

CREATE OR REPLACE VIEW v_automation_rate AS
SELECT
  COALESCE(
    COUNT(*) FILTER (WHERE completed = TRUE AND escalated = FALSE)::DOUBLE PRECISION
    / NULLIF(COUNT(*), 0),
    0.0
  ) AS automation_rate
FROM outcomes;

CREATE OR REPLACE VIEW v_escalation_rate AS
SELECT
  COALESCE(
    COUNT(*) FILTER (WHERE escalated = TRUE)::DOUBLE PRECISION
    / NULLIF(COUNT(*), 0),
    0.0
  ) AS escalation_rate
FROM outcomes;

CREATE OR REPLACE VIEW v_tool_success_rate AS
SELECT
  COALESCE(
    AVG(
      CASE
        WHEN span_name = 'execute_tool' THEN
          CASE
            WHEN COALESCE((attributes ->> 'success')::BOOLEAN, FALSE) THEN 1.0
            ELSE 0.0
          END
        ELSE NULL
      END
    ),
    0.0
  ) AS tool_success_rate
FROM agent_spans;

CREATE OR REPLACE VIEW v_hallucination_rate AS
SELECT
  COALESCE(AVG(CASE WHEN hallucination THEN 1.0 ELSE 0.0 END), 0.0) AS hallucination_rate
FROM evaluation_scores;

-- ---------------------------------------------------------------------------
-- 005_intent_eval.sql
-- ---------------------------------------------------------------------------

-- Phase 6+: intent evaluation schema for Bitext-based intent classification runs.

CREATE TABLE IF NOT EXISTS intent_eval_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  experiment_name VARCHAR(200),
  dataset_key VARCHAR(50),
  model_provider VARCHAR(50),
  model_name VARCHAR(200),
  prompt_version VARCHAR(100),
  metadata JSONB,
  accuracy DOUBLE PRECISION,
  macro_precision DOUBLE PRECISION,
  macro_recall DOUBLE PRECISION,
  macro_f1 DOUBLE PRECISION,
  total_examples INTEGER,
  correct_examples INTEGER,
  failed_examples INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intent_eval_predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES intent_eval_runs(id) ON DELETE CASCADE,
  test_id VARCHAR(200),
  split VARCHAR(100),
  question TEXT,
  expected_intent VARCHAR(100),
  predicted_intent VARCHAR(100),
  is_correct BOOLEAN,
  session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intent_eval_runs_experiment_created_at
  ON intent_eval_runs (experiment_name, created_at);

CREATE INDEX IF NOT EXISTS idx_intent_eval_runs_metadata_gin
  ON intent_eval_runs USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_intent_eval_predictions_run_id
  ON intent_eval_predictions (run_id);

CREATE INDEX IF NOT EXISTS idx_intent_eval_predictions_expected_intent
  ON intent_eval_predictions (expected_intent);

CREATE INDEX IF NOT EXISTS idx_intent_eval_predictions_predicted_intent
  ON intent_eval_predictions (predicted_intent);

-- ---------------------------------------------------------------------------
-- 006_add_company_id.sql
-- ---------------------------------------------------------------------------

-- Add company_id columns to core tables for multi-tenant scoping.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS company_id VARCHAR(100);

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS company_id VARCHAR(100);

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS company_id VARCHAR(100);

-- Basic indexes to keep scoped queries efficient.
CREATE INDEX IF NOT EXISTS idx_documents_company_id ON documents (company_id);
CREATE INDEX IF NOT EXISTS idx_orders_company_id ON orders (company_id);
