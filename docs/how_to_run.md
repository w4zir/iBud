## Overview

This document explains how to bring up the Phase 0/1/2 stack (Docker services), ingest the **WixQA** KB corpus into pgvector, seed mock orders, enable the RAG retriever with Redis caching, and run the test suite.

The primary workflow uses Docker Compose for infrastructure (Postgres, Redis, Ollama, backend) and a local Python virtual environment for management commands and tests.

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

## 2. Start Docker services (Phase 0)

From the project root:

```powershell
docker compose up -d postgres redis ollama backend
```

Verify services:

```powershell
docker compose ps
```

You should see `postgres`, `redis`, `ollama`, and `backend` in `Up` state.

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

## 6. Run tests

For detailed testing flows (unit, integration, and evaluation/quality tests for the retriever and Redis cache), see `docs/how_to_test.md`.

At a glance:

- **Default (unit / mocked, no Postgres or Ollama required):**

  ```powershell
  .\venv\Scripts\python -m pytest tests/
  ```

- **With Postgres/Redis (integration tests):**

  1. Set `POSTGRES_HOST=localhost` so the host can reach the containerized Postgres.
  2. Ensure `docker compose up -d postgres redis ollama backend` is running.
  3. Run integration tests only (includes seeding and retrieval quality checks):

     ```powershell
     $env:POSTGRES_HOST = "localhost"
     .\venv\Scripts\python -m pytest tests/ -m "integration"
     ```

## 7. Quick verification checklist

- **Phase 0**: `docker compose up -d` and `docker compose ps` show all services `Up`.
- **Ingestion**: `docker compose exec -T backend python -m backend.rag.ingest_wixqa` completes with a non-zero document count.
- **Seeding**: `docker compose exec -T backend python /app/scripts/seed_mock_data.py` completes with a non-zero orders count.
- **Phase 2 retrieval**: with services up and data ingested, `pytest evaluation/test_retrieval_quality.py -m "integration"` passes.

Phase 0, Phase 1 (WixQA data and storage foundation), and Phase 2 (RAG core with Redis caching) are then ready for later RAG and agent phases.
