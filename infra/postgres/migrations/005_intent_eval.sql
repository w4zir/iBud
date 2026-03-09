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

