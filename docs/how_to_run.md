## Table of contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [1. Environment setup](#1-environment-setup)
- [2. Start Docker services](#2-start-docker-services)
  - [2.1 Core data infra only (Postgres + Redis + Elasticsearch)](#21-core-data-infra-only-postgres--redis--elasticsearch)
  - [2.2 Backend API only (requires data infra + Ollama)](#22-backend-api-only-requires-data-infra--ollama)
  - [2.3 Frontend only (requires backend)](#23-frontend-only-requires-backend)
  - [2.4 Full stack (backend + frontend + infra)](#24-full-stack-backend--frontend--infra)
  - [2.5 Full stack with observability](#25-full-stack-with-observability)
- [3. Prepare Ollama models (for embeddings)](#3-prepare-ollama-models-for-embeddings)
- [3.1 Agent model routing (runtime)](#31-agent-model-routing-runtime)
- [3.3 Query classification routing (ModernBERT)](#33-query-classification-routing-modernbert)
- [3.4 External human handoff integration (optional)](#34-external-human-handoff-integration-optional)
- [4. Ingest knowledge-base datasets into Elasticsearch](#4-ingest-knowledge-base-datasets-into-elasticsearch)
- [5. Seed mock orders](#5-seed-mock-orders)
- [6. Run automated tests](#6-run-automated-tests)
  - [6.1 Unit / fast tests (default)](#61-unit--fast-tests-default)
  - [6.2 Integration tests (require running services)](#62-integration-tests-require-running-services)
  - [6.3 API-level use-case tests](#63-api-level-use-case-tests)
- [7. Observability & evaluation](#7-observability--evaluation)
  - [7.1 Prometheus & Grafana](#71-prometheus--grafana)
  - [7.2 LangSmith tracing](#72-langsmith-tracing)
  - [7.3 RAGAS evaluation](#73-ragas-evaluation)
  - [7.4 Database & warehouse migrations (Phases 5-6+)](#74-database--warehouse-migrations-phases-5-6)
  - [7.5 Asynchronous evaluation pipeline](#75-asynchronous-evaluation-pipeline)
  - [7.6 OpenTelemetry collector (Phase 8)](#76-opentelemetry-collector-phase-8)
  - [7.7 Audience dashboards (Phase 9)](#77-audience-dashboards-phase-9)
  - [7.8 Alerting and SLO setup (Phase 10)](#78-alerting-and-slo-setup-phase-10)
  - [7.9 Self-healing remediation endpoints (Phase 11)](#79-self-healing-remediation-endpoints-phase-11)
- [8. Quick verification checklist](#8-quick-verification-checklist)

---

## Overview

This document explains how to bring up the stack with Docker services, ingest the **WixQA** KB corpus into **Elasticsearch** (with Postgres kept for relational data), seed mock orders, run the agentic backend API, use the Streamlit UI, and enable observability and evaluation.

The primary workflow uses Docker Compose for infrastructure (Postgres, Redis, Elasticsearch, Ollama, backend, frontend, Prometheus, Grafana) and a local Python virtual environment for management commands and tests.

## Prerequisites

- Docker and Docker Compose v2 (`docker compose` CLI)
- Python 3.11+ on your host
- A virtual environment at `venv/` in the project root
- Network access for Python packages, HuggingFace datasets, and Ollama models

## 1. Environment setup

1. From the project root, create or activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install backend dependencies:

   ```powershell
   .\venv\Scripts\python -m pip install -r backend\requirements.txt
   ```

3. Copy and configure environment variables:

   ```powershell
   copy .env.example .env
   ```

   For Phase 1/2 with Ollama embeddings, set in `.env`:

   - `EMBEDDING_PROVIDER=ollama`
   - `EMBEDDING_MODEL_OLLAMA=nomic-embed-text`
   - `EMBEDDING_DIM=768`
   - `HF_DATASET_PRIMARY=Wix/WixQA`
   - `POSTGRES_DB=ecom_support`
   - `POSTGRES_USER` / `POSTGRES_PASSWORD` as desired

For retrieval (Elasticsearch-backed) and caching (Redis-backed), also configure:

   - `REDIS_HOST=redis`
   - `REDIS_PORT=6379`
   - `REDIS_CACHE_TTL=300` (seconds)

Elasticsearch + retrieval settings:
- `ES_HOST=elasticsearch`
- `ES_PORT=9200`
- `ES_INDEX_NAME=ecom-support-documents`
- `ES_RETRIEVAL_TOP_K=40` (candidate docs before reranking)
- `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `RERANK_TOP_K=5`

ModernBERT query classification (issue vs non-issue pipeline routing):
- `CLASSIFIER_MODEL=MoritzLaurer/ModernBERT-base-zeroshot-v2.0`
- `CLASSIFIER_THRESHOLD=0.7`

  **Backend structured logging:** Logs are emitted as JSON with stable fields (for example `request_id`, `session_id`, `user_id`, `intent`, `tool_name`, `status`, `latency_ms`, `error_type`). Set `DEBUG=true` in `.env` for debug-level events; default is `false` (INFO level) for normal runs.

  **Request correlation:** Every request is assigned a request ID. You can provide your own via `X-Request-ID`; otherwise the backend generates one and returns it in the response headers.

  **OpenTelemetry (optional):** To emit OTel traces, set:
  - `OTEL_ENABLED=true`
  - `OTEL_SERVICE_NAME=ibud-backend`
  - `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`

## 2. Start Docker services

You can start different slices of the system depending on what you are working on.

### 2.1 Core data infra only (Postgres + Redis + Elasticsearch)

```powershell
docker compose up -d postgres redis elasticsearch
```

### 2.2 Backend API only (requires data infra + Ollama)

```powershell
docker compose up -d postgres redis elasticsearch ollama backend
```

Backend API will be available at `http://localhost:8000`.

### 2.3 Frontend only (requires backend)

```powershell
docker compose up -d backend frontend
```

Streamlit UI will be available at `http://localhost:8501`.

### 2.4 Full stack (backend + frontend + infra)

```powershell
docker compose up -d postgres redis elasticsearch ollama backend frontend
```

### 2.5 Full stack with observability

```powershell
docker compose up -d postgres redis elasticsearch ollama backend frontend prometheus grafana alertmanager
```

Verify services:

```powershell
docker compose ps
```

You should see at least `postgres`, `redis`, `elasticsearch`, `ollama`, `backend`, and `frontend` in `Up` state; for observability, also `prometheus`, `grafana`, and `alertmanager`.

## 3. Prepare Ollama models (for embeddings)

Inside the Ollama container:

```powershell
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

Optionally, pull a larger model for the planner and set `OLLAMA_PLANNER_MODEL` in `.env` (e.g. `llama3.1:70b`).

## 3.1 Agent model routing (runtime)

The runtime uses role-based model selection:

- **Planner**: `OLLAMA_PLANNER_MODEL` (default: same as `OLLAMA_MODEL`, e.g. `llama3.2`)
- **Small model**: `OLLAMA_SMALL_MODEL` (default: same as `OLLAMA_MODEL`, e.g. `llama3.2`)

You can override these in `.env` as needed.

## 3.3 Query classification routing (ModernBERT)

Before the planner runs, the backend uses a ModernBERT zero-shot classifier to
route the query into one of two pipelines:

- `issue` pipeline: allows order/return tools in addition to KB search
- `non_issue` pipeline: restricts tools to KB/FAQ search only

Configure via:

- `CLASSIFIER_MODEL` (default `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`)
- `CLASSIFIER_THRESHOLD` (default `0.7`)

## 3.4 External human handoff integration (optional)

To enable external human escalation calls (in addition to creating a local DB ticket), set:

- `HUMAN_HANDOFF_URL=<https endpoint>`
- (optional) `HUMAN_HANDOFF_API_KEY=<token>`
- (optional) `HUMAN_HANDOFF_TIMEOUT_SECONDS=5`

## 4. Ingest knowledge-base datasets into Elasticsearch

Run the **WixQA** ingestion pipeline inside the backend container:

```powershell
docker compose exec -T backend python -m backend.rag.ingest_wixqa
```

This now performs a dual-write:

- Loads the HuggingFace dataset `Wix/WixQA` (config `wix_kb_corpus`)
- Extracts article fields (`id`, `url`, `contents`, `article_type`)
- Chunks with section-aware and parent-document logic (chunk size ~400 tokens, overlap 50)
- Embeds child chunks (Ollama `nomic-embed-text`, 768-dim by default)
- Inserts parent/child rows into Postgres `documents` for relational needs
- Indexes parent/child content + embeddings into Elasticsearch for retrieval
- Uses `source="wixqa"` and `doc_tier=1` consistently across both stores

On success you’ll see a line like:

```text
WixQA ingestion complete. documents (source=wixqa) count=<N> ...
```

Optionally, you can also ingest the **Bitext** customer-support QA dataset as a
secondary corpus:

```powershell
docker compose exec -T backend python -m backend.rag.ingest_bitext
```

This:

- Loads the HuggingFace dataset `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Builds a support record per row from `instruction` + `response`
- Embeds each record and inserts rows into Postgres `documents` with `source="bitext"`, `doc_tier=1`
- Indexes those records into Elasticsearch (`ES_INDEX_NAME`) for retrieval

## 5. Seed mock orders

Run seeding inside the backend container:

```powershell
docker compose exec -T backend python /app/scripts/seed_mock_data.py
```

Or run ingestion and seeding in one go (bash-capable shell):

```bash
bash scripts/ingest_wixqa.sh
```

> If you see errors about `set` or `$'\r'`, use UNIX line endings (e.g. `dos2unix scripts/ingest_wixqa.sh`).

## 6. Run automated tests

All test commands assume you are in the project root with the virtual environment activated (`.\venv\Scripts\activate`) so that the `backend` package is importable.

### 6.1 Unit / fast tests (default)

```powershell
.\venv\Scripts\python -m pytest
```

This runs fast unit tests only (integration tests are deselected by default via `pytest.ini`).

### 6.2 Integration tests (require running services)

After starting Postgres/Redis/Elasticsearch and performing ingestion and seeding:

```powershell
.\venv\Scripts\python -m pytest -m integration
```

Integration tests cover vector queries, retrieval quality, and other flows that depend on live data.

### 6.3 API-level use-case tests

With Docker services running and data ingested/seeded:

```powershell
.\venv\Scripts\python -m pytest tests/test_use_cases.py
```

This exercises the `/chat` API and session history using a stubbed agent, validating session creation, caching behaviour, and history retrieval.

## 7. Observability & evaluation

### 7.1 Prometheus & Grafana

- Prometheus is configured to scrape `backend:8000/metrics` inside the Docker network (see `infra/prometheus/prometheus.yml`).
- Grafana can import the dashboard JSON at `infra/grafana/dashboards/phase6-observability.json` to visualise:
  - Chat request rate and latency
  - Retrieval latency
  - Redis cache hits
  - Tool calls
  - Escalations
  - LLM tokens by provider

### 7.2 LangSmith tracing

To enable LangSmith traces for every agent run, set in `.env`:

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=ls__...`
- `LANGCHAIN_PROJECT=ecom-support-rag`

With these set, `backend/observability/langsmith_tracer.py` attaches metadata and tags per run (including a `chat_request` parent span) so traces appear in the LangSmith UI.

### 7.3 RAGAS evaluation

1. Build the WixQA evaluation testset:

   ```powershell
   .\venv\Scripts\python -m scripts.datasets.build_wixqa_testset
   ```

2. Ensure the backend is running locally (for example via full stack Docker compose).

3. Run the RAGAS evaluation:

   ```powershell
   .\venv\Scripts\python -m evaluation.ragas_eval --backend-url http://localhost:8000 --limit 50
   ```

Results are written under `evaluation/results/` as `run_<timestamp>.json`, including overall metric means and per-split breakdowns.

4. (Optional) Configure a separate judge model/provider for RAGAS:

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

### 7.4 Database & warehouse migrations (Phases 5-6+)

Apply the new analytics/warehouse and evaluation/multi-tenant migrations in order
(all files are idempotent and safe to re-run):

```powershell
Get-Content infra/postgres/migrations/003_observability_warehouse.sql | docker compose exec -T postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB
Get-Content infra/postgres/migrations/004_analytics_views.sql         | docker compose exec -T postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB
Get-Content infra/postgres/migrations/005_intent_eval.sql             | docker compose exec -T postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB
Get-Content infra/postgres/migrations/006_add_company_id.sql          | docker compose exec -T postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB
```

These additional migrations enable:

- **005_intent_eval.sql**: creates `intent_eval_runs` and `intent_eval_predictions`
  tables used by `evaluation/intent_eval.py` and the intent-eval sections in
  `docs/how_to_test.md`.
- **006_add_company_id.sql**: adds `company_id` columns and indexes to
  `documents`, `sessions`, and `orders` for multi-tenant/company-scoped flows
  (for example, Foodpanda policy RAG and company-aware tools).

These migrations add:
- Warehouse tables: `agent_spans`, `outcomes`, `evaluation_scores`
- Session analytics columns (`intent`, `escalated`, `resolved_at`, `csat_score`, `nps_score`)
- Analytics SQL views for KPI queries

### 7.5 Asynchronous evaluation pipeline

Continuous scoring is separated from `evaluation/ragas_eval.py` (benchmark runner).

- Manual trigger:

```powershell
curl -X POST "http://localhost:8000/admin/eval/trigger?limit=25&min_age_minutes=5"
```

- Cron-style example (runs every 30 minutes):

```text
*/30 * * * * curl -X POST "http://backend:8000/admin/eval/trigger?limit=25&min_age_minutes=5"
```

### 7.6 OpenTelemetry collector (Phase 8)

An OTel collector config is provided at `infra/otel/otel-collector-config.yaml`.

Minimal local run example:

```powershell
docker run --rm -p 4317:4317 -p 4318:4318 -p 9464:9464 `
  -v ${PWD}/infra/otel/otel-collector-config.yaml:/etc/otelcol/config.yaml `
  otel/opentelemetry-collector:latest `
  --config /etc/otelcol/config.yaml
```

When enabled, the backend emits:
- Root `conversation` spans from chat handling
- Child spans for `intent_detection`, `retrieval`, `tool_calls`, `response_synthesis`, and `outcome` steps
- OTel trace IDs are also attached to LangSmith metadata for cross-correlation

### 7.7 Audience dashboards (Phase 9)

Grafana dashboards now include audience-specific views:

- `infra/grafana/dashboards/phase6-observability.json` (operations baseline)
- `infra/grafana/dashboards/executive-observability.json`
- `infra/grafana/dashboards/product-observability.json`
- `infra/grafana/dashboards/ai-quality-observability.json`

Import each dashboard in Grafana and validate panel queries against `/metrics` and warehouse-backed KPI definitions in `docs/qa.md`.

### 7.8 Alerting and SLO setup (Phase 10)

Prometheus rules and Alertmanager config are provided:

- `infra/prometheus/alert_rules.yml`
- `infra/alertmanager/alertmanager.yml`
- `docs/slos.md` (availability, latency, and quality objectives)

Start alerting components with:

```powershell
docker compose up -d prometheus alertmanager
```

Then verify:

1. Prometheus loads `rule_files` from `infra/prometheus/prometheus.yml`
2. Alertmanager is reachable at `http://localhost:9093`
3. Alert annotations reference runbooks in `docs/runbook_*.md`

### 7.9 Self-healing remediation endpoints (Phase 11)

Use admin endpoints to inspect and run remediation logic:

- Dry-run checks: `POST /admin/remediation/check`
- Execute eligible remediations: `POST /admin/remediation/trigger`
- Intervention history: `GET /admin/remediation/history?hours=24`
- Governance config: `GET /admin/remediation/config`

Examples:

```powershell
curl -X POST http://localhost:8000/admin/remediation/check
curl -X POST http://localhost:8000/admin/remediation/trigger
curl http://localhost:8000/admin/remediation/history?hours=24
```

## 8. Quick verification checklist

- **Core services**: `docker compose up -d` and `docker compose ps` show `postgres`, `redis`, `ollama`, `backend`, and `frontend` as `Up`.
- **Ingestion**: `docker compose exec -T backend python -m backend.rag.ingest_wixqa` completes with a non-zero document count.
- **Seeding**: `docker compose exec -T backend python /app/scripts/seed_mock_data.py` completes with a non-zero orders count.
- **Agent API**: `curl http://localhost:8000/health` returns a JSON object with `status`, `postgres`, `redis`, and `ollama` fields.
- **UI**: navigate to `http://localhost:8501` and send a chat message; you should receive a model response and (when available) sources/tool activity.
- **Metrics**: Prometheus shows `backend` target as `UP` and `/metrics` exposes Phase 6 counters/histograms.
- **Traces**: with LangSmith env vars set, new chats appear as traces in the LangSmith project.
- **Evaluation**: RAGAS run completes and writes a `run_<timestamp>.json` summary under `evaluation/results/`.
