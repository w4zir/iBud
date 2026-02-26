## Overview

This document explains how to bring up the stack with Docker services, ingest the **WixQA** KB corpus into pgvector, seed mock orders, run the agentic backend API (Phases 3–4), use the Streamlit UI (Phase 5), and enable observability and evaluation (Phase 6).

The primary workflow uses Docker Compose for infrastructure (Postgres, Redis, Ollama, backend, frontend, Prometheus, Grafana) and a local Python virtual environment for management commands and tests.

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

   For Phase 2 Redis-backed retrieval, also configure:

   - `REDIS_HOST=redis`
   - `REDIS_PORT=6379`
   - `REDIS_CACHE_TTL=300` (seconds)

   **Backend debug logging:** Set `DEBUG=true` in `.env` to print minimal flow logs to the backend console (e.g. `[DEBUG][chat]` and `[DEBUG][agent]`). Logs include request/session metadata, cache hit/miss, agent stage names, and tool name + parameters only. Default is `false`; leave unset or `false` in production.

## 2. Start Docker services

You can start different slices of the system depending on what you are working on.

### 2.1 Core data infra only (Postgres + Redis)

```powershell
docker compose up -d postgres redis
```

### 2.2 Backend API only (requires data infra + Ollama)

```powershell
docker compose up -d postgres redis ollama backend
```

Backend API will be available at `http://localhost:8000`.

### 2.3 Frontend only (requires backend)

```powershell
docker compose up -d backend frontend
```

Streamlit UI will be available at `http://localhost:8501`.

### 2.4 Full stack (backend + frontend + infra)

```powershell
docker compose up -d postgres redis ollama backend frontend
```

### 2.5 Full stack with observability

```powershell
docker compose up -d postgres redis ollama backend frontend prometheus grafana
```

Verify services:

```powershell
docker compose ps
```

You should see at least `postgres`, `redis`, `ollama`, `backend`, and `frontend` in `Up` state; for observability, also `prometheus` and `grafana`.

## 3. Prepare Ollama models (for embeddings)

Inside the Ollama container:

```powershell
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

## 4. Ingest WixQA KB into pgvector

Run the **WixQA** ingestion pipeline inside the backend container:

```powershell
docker compose exec -T backend python -m backend.rag.ingest_wixqa
```

This:

- Loads the HuggingFace dataset `Wix/WixQA` (config `wix_kb_corpus`)
- Extracts article fields (`id`, `url`, `contents`, `article_type`)
- Chunks with section-aware and parent-document logic (chunk size ~400 tokens, overlap 50)
- Embeds child chunks (Ollama `nomic-embed-text`, 768-dim by default)
- Inserts parent and child rows into `documents` with `source="wixqa"`, `doc_tier=1`

On success you’ll see a line like:

```text
WixQA ingestion complete. documents (source=wixqa) count=<N> ...
```

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

After starting Postgres/Redis and performing ingestion and seeding:

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

With these set, `backend/agent/graph.py` attaches metadata and tags per run so traces appear in the LangSmith UI.

### 7.3 RAGAS evaluation

1. Build the WixQA evaluation testset:

   ```powershell
   .\venv\Scripts\python -m evaluation.build_wixqa_testset
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

## 8. Quick verification checklist

- **Core services**: `docker compose up -d` and `docker compose ps` show `postgres`, `redis`, `ollama`, `backend`, and `frontend` as `Up`.
- **Ingestion**: `docker compose exec -T backend python -m backend.rag.ingest_wixqa` completes with a non-zero document count.
- **Seeding**: `docker compose exec -T backend python /app/scripts/seed_mock_data.py` completes with a non-zero orders count.
- **Agent API**: `curl http://localhost:8000/health` returns a JSON object with `status`, `postgres`, `redis`, and `ollama` fields.
- **UI**: navigate to `http://localhost:8501` and send a chat message; you should receive a model response and (when available) sources/tool activity.
- **Metrics**: Prometheus shows `backend` target as `UP` and `/metrics` exposes Phase 6 counters/histograms.
- **Traces**: with LangSmith env vars set, new chats appear as traces in the LangSmith project.
- **Evaluation**: RAGAS run completes and writes a `run_<timestamp>.json` summary under `evaluation/results/`.
