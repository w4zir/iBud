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

