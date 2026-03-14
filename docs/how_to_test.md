## Testing Guide

This document describes how to test the system across different layers:

- Unit tests (functions, components)
- Integration tests (database, retrieval)
- API-level use-case tests (end-to-end flows)
- Observability checks (metrics, traces)
- RAG quality evaluation (RAGAS)
 - Company-scoped/Foodpanda policy tests

All examples assume a PowerShell shell on Windows from the project root.

## 1. Prerequisites

- Python 3.11+ and a virtual environment at `venv/`
- Backend dependencies installed:

  ```powershell
  .\venv\Scripts\python -m pip install -r backend\requirements.txt
  ```

- Docker services up when running integration or API-level tests (see `docs/how_to_run.md`).
- WixQA KB ingested and mock orders seeded for retrieval- and order-related tests.

## 2. Unit tests (fast feedback)

Unit tests live throughout `tests/` and cover:

- RAG components (`tests/test_retriever.py`, `tests/test_chunker.py`, `tests/test_ingest_transform.py`)
- Redis caching helpers (`tests/test_redis_client.py`)
- Seed scripts (`tests/test_seed_mock_data.py`)
- Phase 6 observability and tracing (`tests/test_langsmith_tracing.py`, `tests/test_observability_metrics.py`)
- Warehouse/OTel/analytics layers (`tests/test_warehouse.py`, `tests/test_otel_tracing.py`, `tests/test_analytics_metrics.py`)
- Request-ID middleware (`tests/test_request_id_middleware.py`)
- Evaluation utilities (`tests/test_build_wixqa_testset.py`, `tests/test_ragas_eval.py`)
- Async evaluation pipeline (`tests/test_evaluation_pipeline.py`)

Run all unit tests (integration tests are excluded by default via `pytest.ini`):

```powershell
.\venv\Scripts\python -m pytest
```

## 3. Integration tests (live data)

Integration tests are marked with `@pytest.mark.integration` and assume:

- Postgres and Redis are running
- WixQA corpus is ingested into pgvector

Examples include:

- `tests/test_vector_query_smoke.py`
- `tests/test_seed_mock_data.py`

Run integration tests explicitly:

```powershell
.\venv\Scripts\python -m pytest -m integration
```

## 4. API-level use-case tests

Automated use-case tests are implemented in `tests/test_use_cases.py`. They drive the FastAPI app directly and validate:

- Session creation and message persistence via `POST /chat`
- Request-level Redis caching for identical queries in the same session
- Session history retrieval via `GET /sessions/{session_id}/history`

Run them with:

```powershell
.\venv\Scripts\python -m pytest tests/test_use_cases.py
```

These tests stub the underlying agent runtime (`run_orchestrated_agent`) to avoid external LLM calls while still exercising the API surface and database interactions.

### 4.1 Foodpanda API integration tests

Foodpanda-specific API integration tests live in `tests/test_foodpanda_rag_integration.py`. They validate that:

- `/chat` accepts `dataset="foodpanda"` and `company="foodpanda"`.
- End-to-end responses are returned successfully for Foodpanda RAG scenarios.

These tests require the **company-scoped schema migration**
`infra/postgres/migrations/006_add_company_id.sql` to be applied so that the
`company_id` column exists on `documents`, `sessions`, and `orders`. If you
haven’t run it yet, see `docs/how_to_run.md` §7.4 for the exact commands.

Run them (with integration marker) using:

```powershell
.\venv\Scripts\python -m pytest tests/test_foodpanda_rag_integration.py -m integration
```

## 5. Manual end-to-end scenarios

In addition to automated checks, you can manually validate key user journeys once the backend and frontend are running.

### 5.1 Chat via Streamlit UI

1. Ensure Docker services are running (including `backend` and `frontend`).
2. Open the Streamlit UI at `http://localhost:8501`.
3. In the sidebar:
   - Set a `User ID` (e.g. `user-1`).
   - Choose a **Knowledge base dataset**:
     - `WixQA KB (articles)` — help-center articles ingested from `Wix/WixQA`
     - `Bitext QA pairs` — labeled customer-support Q&A pairs from Bitext
   - Optionally toggle “Show sources” on or off.
4. In the main chat area:
   - Ask an order-related question (e.g. “What is the status of my order?”).
   - Ask a return-related question (e.g. “Can I return my last order?”).
   - Ask a product or policy question (e.g. “What is your return policy?”).
5. Observe:
   - Assistant responses appear under your messages.
   - When applicable, a “Tools used” line indicates which tools were called.
   - When the agent escalates, a banner indicates a ticket was created and shows the ticket ID.
   - Under **Sources**, each document includes a `dataset` label (the document `source`, for example `wixqa` or `bitext`) alongside category, tier, and score.

### 5.2 Chat via HTTP API

You can also call the API directly using `curl` or `Invoke-RestMethod`.

```powershell
curl -X POST http://localhost:8000/chat/ `
  -H "Content-Type: application/json" `
  -d '{ "session_id": null, "user_id": "manual-user", "message": "Hello", "dataset": "wixqa" }'
```

The response includes:

- `session_id` — use this to continue the conversation.
- `response` — assistant answer text.
- `sources` — retrieved documents (when available), including `dataset` (document `source`), `category`, `doc_tier`, and `score`.
- `tools_used` — list of tool names invoked.
- `escalated` / `ticket_id` — escalation status.

### 5.3 Inspect session history

Given a `session_id` from a chat response, fetch the full message history:

```powershell
curl http://localhost:8000/sessions/<session_id>/history
```

You should see a chronological list of `{role, content, created_at}` entries for the session.

## 6. Observability checks

### 6.1 Prometheus metrics

1. Start Prometheus and backend:

   ```powershell
   docker compose up -d backend prometheus
   ```

2. Hit the metrics endpoint directly:

   ```powershell
   curl http://localhost:8000/metrics
   ```

   You should see:

   - `chat_requests_total`
   - `chat_latency_seconds_*` buckets
   - `retrieval_latency_seconds_*` buckets
   - `redis_cache_hits_total`
   - `agent_tool_calls_total`
   - `tool_outcome_total`
   - `intent_distribution_total`
   - `task_outcome_total`
   - `errors_total`
   - `escalations_total`
   - `llm_tokens_total`
   - `db_operation_latency_seconds`
   - `redis_operation_latency_seconds`

3. Open the Prometheus UI (default `http://localhost:9090`) and verify that the `backend` target is `UP`.

### 6.2 Grafana dashboard

1. Start Grafana alongside Prometheus:

   ```powershell
   docker compose up -d prometheus grafana
   ```

2. Log into Grafana (default `http://localhost:3000`).
3. Import the dashboard JSON from `infra/grafana/dashboards/phase6-observability.json`.
4. Verify panels update as you send chat traffic:
   - Chat rate and latency
   - Retrieval latency
   - Cache hits
   - Tool calls
   - Escalations
   - LLM tokens by provider
   - Executive, product, and AI-quality dashboard panels for audience-specific views

### 6.3 LangSmith traces

1. Set the following in `.env`:

   - `LANGCHAIN_TRACING_V2=true`
   - `LANGCHAIN_API_KEY=ls__...`
   - `LANGCHAIN_PROJECT=ecom-support-rag`

2. Restart the backend so it picks up the new environment.
3. Send one or more chat requests (via UI or API).
4. In the LangSmith UI, confirm that new traces appear with metadata such as `session_id`, `user_id`, and `intent` attached.

### 6.4 OTel tracing verification

With `OTEL_ENABLED=true` and an OTLP endpoint configured:

1. Send chat traffic through `POST /chat`.
2. Inspect your OTel backend/collector output for:
   - A root `conversation` span
   - Child spans for intent, retrieval, tool calls, synthesis, and outcome
3. Confirm LangSmith metadata includes `otel_trace_id` and `otel_span_id` when tracing is enabled.

## 7. RAG quality evaluation (RAGAS)

You can now evaluate against both the WixQA and Bitext datasets.

### 7.1 WixQA evaluation

1. Ensure WixQA KB is ingested and backend is running.
2. Build the WixQA testset:

   ```powershell
   .\venv\Scripts\python -m evaluation.build_wixqa_testset
   ```

3. Run the RAGAS evaluation (WixQA remains the default dataset):

   ```powershell
   .\venv\Scripts\python -m evaluation.ragas_eval --backend-url http://localhost:8000 --limit 50
   ```

4. Inspect the summary JSON written under `evaluation/results/` (e.g. `run_<timestamp>.json`) to view:

   - Overall scores for `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
   - Per-split breakdown for ExpertWritten vs Simulated questions

### 7.2 Bitext evaluation

Bitext provides labeled customer-support Q&A pairs. You can evaluate the agent
with either a full or sampled Bitext testset.

1. Ensure the Bitext corpus is ingested (optional but recommended for realism):

   ```powershell
   docker compose exec -T backend python -m backend.rag.ingest_bitext
   ```

2. Build the Bitext testsets:

   ```powershell
   # Build both full and sampled Bitext testsets
   .\venv\Scripts\python -m evaluation.build_bitext_testset --mode both --max-per-intent 50
   ```

   This writes:

   - `evaluation/bitext_testset_full.json`
   - `evaluation/bitext_testset_sampled.json`

3. Run Bitext RAGAS evaluation with the sampled testset (recommended default):

   ```powershell
   .\venv\Scripts\python -m evaluation.ragas_eval `
     --backend-url http://localhost:8000 `
     --dataset-key bitext `
     --testset-path evaluation/bitext_testset_sampled.json `
     --limit 100
   ```

4. (Optional) Run a full Bitext evaluation:

   ```powershell
   .\venv\Scripts\python -m evaluation.ragas_eval `
     --backend-url http://localhost:8000 `
     --dataset-key bitext `
     --testset-path evaluation/bitext_testset_full.json
   ```

### 7.3 Evaluation result JSON schema

Each `evaluation/results/run_<timestamp>.json` file includes:

- `metrics`: overall metric means (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), with `null` when no valid values are available
- `by_split`: per-split metric means (`expertwritten`, `simulated`)
- `num_rows`: number of rows that reached the evaluator dataframe
- `valid_counts`: non-NaN value count per metric used for aggregation
- `failed_rows`: per-question failures (for example backend call or evaluator failure) with error reason
- `thresholds`: configured pass/fail thresholds
- `threshold_details`: actual vs required values for each threshold check
- `passed`: overall boolean pass/fail for the run

5. (Optional) Configure a separate judge model/provider for RAGAS:

   - By default, if no judge-specific env vars are set, RAGAS will:
     - Use a local Ollama-backed judge when `LLM_PROVIDER=ollama` (the default for agents), reusing `OLLAMA_BASE_URL`/`OLLAMA_MODEL`.
     - Use a Cerebras-backed judge when `LLM_PROVIDER=cerebras`, reusing `CEREBRAS_*` env vars.
     - Fall back to RAGAS' own OpenAI-backed default evaluator otherwise.
   - To force RAGAS to use a different judge than the runtime agent, set in `.env`:
     - `RAGAS_LLM_PROVIDER=ollama`, `openai`, or `cerebras`.
     - Optionally, the following judge-specific overrides:
       - For Ollama: `RAGAS_OLLAMA_MODEL`, `RAGAS_OLLAMA_BASE_URL`, `RAGAS_OLLAMA_API_KEY` (API key is typically ignored by local Ollama but required by the OpenAI client).
       - For OpenAI: `RAGAS_OPENAI_MODEL`, `RAGAS_OPENAI_BASE_URL` (credentials are read from `OPENAI_API_KEY`).
       - For Cerebras: `RAGAS_CEREBRAS_MODEL`, `RAGAS_CEREBRAS_API_KEY`, `RAGAS_CEREBRAS_BASE_URL`.

   These RAGAS-specific env vars only affect evaluation runs and do not change which model the agent uses for live chats.

## 8. Intent classification evaluation (Bitext)

In addition to RAGAS, you can directly evaluate the agent's **intent classification** against the Bitext testsets.

### 8.1 Run Bitext intent evaluation

With the backend running and Bitext ingested/testsets built:

> **Database migration prerequisite:** Ensure the intent-eval migration
> `infra/postgres/migrations/005_intent_eval.sql` has been applied so that the
> `intent_eval_runs` and `intent_eval_predictions` tables exist. You can apply
> it via the commands in `docs/how_to_run.md` §7.4, or directly:
>
> ```powershell
> Get-Content infra/postgres/migrations/005_intent_eval.sql | docker compose exec -T postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB
> ```

```powershell
.\venv\Scripts\python -m evaluation.intent_eval `
  --backend-url http://localhost:8000 `
  --dataset-key bitext `
  --testset-path evaluation/bitext_testset_sampled.json `
  --limit 200 `
  --experiment-name "bitext-intent-v1" `
  --model-provider ollama `
  --model-name llama3.2 `
  --prompt-version "intent-v1" `
  --intent-prompt-profile bitext
```

This:

- Sends each Bitext question to `/chat/intent` with `dataset="bitext"` and the `bitext` prompt profile.
- Reads the classified `intent` from the intent-only response.
- Compares it to the ground-truth `intent` in the testset.
- Prints a JSON summary (accuracy, macro precision/recall/F1, counts, etc.) to the terminal. Confusion is omitted by default; use `--print-confusion` to include it.
- Writes a summary artifact under `evaluation/results/intent_run_<timestamp>.json`.
- Persists run + per-example rows into Postgres tables:
  - `intent_eval_runs`
  - `intent_eval_predictions`
  - and a summary row in `evaluation_scores.metadata` with `pipeline="intent_eval"`.

### 8.2 Regenerate summaries from Postgres

To rebuild a summary for a previously stored run without re-calling the backend:

```powershell
.\venv\Scripts\python -m evaluation.intent_eval --from-db --run-id <run_uuid>
```

Or, to regenerate the most recent run for a given experiment name:

```powershell
.\venv\Scripts\python -m evaluation.intent_eval --from-db --from-experiment "bitext-intent-v1"
```

### 8.3 Example SQL queries

Some common patterns:

```sql
-- List recent intent-eval runs with basic metrics
SELECT
  id,
  experiment_name,
  dataset_key,
  model_provider,
  model_name,
  prompt_version,
  accuracy,
  macro_f1,
  created_at
FROM intent_eval_runs
ORDER BY created_at DESC
LIMIT 20;

-- Inspect misclassified examples for a specific run
SELECT
  p.test_id,
  p.split,
  p.question,
  p.expected_intent,
  p.predicted_intent,
  p.error
FROM intent_eval_predictions p
WHERE p.run_id = '<run_uuid>'
  AND COALESCE(p.is_correct, FALSE) = FALSE;
```

## 9. Troubleshooting

- **API not reachable (`connection refused`)**
  - Ensure `docker compose ps` shows `backend` as `Up`.
  - Confirm you are calling `http://localhost:8000` (or your configured host/port).

- **Tests failing with `ModuleNotFoundError: No module named 'backend'`**
  - Make sure you run `pytest` from the project root (`d:\ai_ws\projects\iBud`) with the virtual environment activated so `backend` is on `PYTHONPATH`.

- **Use-case tests failing due to database errors**
  - Verify Postgres is running and accessible from the host.
  - Confirm ingestion and seeding have completed successfully.

- **Async evaluation pipeline not producing scores**
  - Trigger manually via `POST /admin/eval/trigger`.
  - Confirm rows exist in `outcomes` and are old enough for `min_age_minutes`.
  - Verify no existing `evaluation_scores` row already exists for the same `session_id`.

- **Streamlit UI cannot reach backend**
  - Check that `BACKEND_BASE_URL` (if set) points to the correct backend URL.
  - Ensure CORS is configured via `CORS_ORIGINS` in `.env` (e.g. `http://localhost:8501`).

## 10. Alerting and SLO tests (Phase 10)

Run alert and dashboard structure tests:

```powershell
.\venv\Scripts\python -m pytest tests/test_alert_rules.py tests/test_dashboards.py
```

Validate that:

- All required alerts exist with expressions, severity labels, and runbook annotations.
- Audience dashboard JSON files are valid and include expected key panels.

## 11. Remediation tests (Phase 11)

Run remediation unit tests:

```powershell
.\venv\Scripts\python -m pytest tests/test_remediation.py
```

These tests validate:

- Rule evaluation and trigger behavior
- Governance controls (`manual_override`, global enable/disable, action budget)
- Dry-run and execution behavior of remediation engine
- Intervention recording path

