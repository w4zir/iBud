## Overview

This document explains how to bring up the stack with Docker services, ingest the **WixQA** KB corpus into pgvector, seed mock orders, run the agentic backend API (Phases 3–4), and use the Streamlit UI (Phase 5). It also links to the use-case testing guide.

The primary workflow uses Docker Compose for infrastructure (Postgres, Redis, Ollama, backend, frontend) and a local Python virtual environment for management commands and tests.

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

## 2. Start Docker services

From the project root, start all core services, including the backend API and Streamlit frontend:

```powershell
docker compose up -d postgres redis ollama backend frontend
```

Verify services:

```powershell
docker compose ps
```

You should see `postgres`, `redis`, `ollama`, `backend`, and `frontend` in `Up` state.

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

## 6. Run use-case tests

For detailed testing flows focused on end-to-end use cases (agent + API), see `docs/how_to_test.md`.

At a glance, once Docker services are running and data is ingested/seeded:

```powershell
.\venv\Scripts\python -m pytest tests/test_use_cases.py
```

This exercises the `/chat` API and session history using a stubbed agent, validating session creation, caching behaviour, and history retrieval.

## 7. Quick verification checklist

- **Core services**: `docker compose up -d` and `docker compose ps` show `postgres`, `redis`, `ollama`, `backend`, and `frontend` as `Up`.
- **Ingestion**: `docker compose exec -T backend python -m backend.rag.ingest_wixqa` completes with a non-zero document count.
- **Seeding**: `docker compose exec -T backend python /app/scripts/seed_mock_data.py` completes with a non-zero orders count.
- **Agent API**: `curl http://localhost:8000/health` returns a JSON object with `status`, `postgres`, `redis`, and `ollama` fields.
- **UI**: navigate to `http://localhost:8501` and send a chat message; you should receive a model response and (when available) sources/tool activity.
