## Overview

This document explains how to bring up the Phase 1 data/storage stack, ingest the HuggingFace dataset into pgvector, seed mock orders, and run the Phase 1 tests.

The primary workflow uses Docker Compose for infrastructure (Postgres, Redis, Ollama, backend) and a local Python virtual environment for running management commands and tests.

## Prerequisites

- Docker and Docker Compose v2 (`docker compose` CLI)
- Python 3.11 on your host
- A virtual environment created at `venv/` in the project root
- Network access to download Python packages, HuggingFace datasets, and Ollama models

## 1. Environment setup

1. From the project root (`d:\ai_ws\projects\iBud`), create or activate a virtual environment (if not already done):

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install backend dependencies (including LangChain, pgvector, asyncpg, Ollama/OpenAI clients, and pytest):

   ```powershell
   .\venv\Scripts\python -m pip install -r backend\requirements.txt
   ```

3. Copy and configure environment variables:

   ```powershell
   copy .env.example .env
   ```

   Then edit `.env` as needed. For Phase 1 with Ollama embeddings:

   - `EMBEDDING_PROVIDER=ollama`
   - `EMBEDDING_MODEL_OLLAMA=nomic-embed-text`
   - `POSTGRES_DB=ecom_support`
   - `POSTGRES_USER` / `POSTGRES_PASSWORD` as desired

## 2. Start Docker services

From the project root:

```powershell
docker compose up -d postgres redis ollama backend
```

You can verify services are running with:

```powershell
docker compose ps
```

You should see at least `postgres`, `redis`, `ollama`, and `backend` in `Up` state.

## 3. Prepare Ollama models (for embeddings)

Inside the Ollama container, pull the required models:

```powershell
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

This may take several minutes on first run.

## 4. Ingest dataset into pgvector

Run the ingestion pipeline inside the backend container:

```powershell
docker compose exec -T backend python -m rag.ingest
```

What this does:

- Loads the HuggingFace dataset (`rjac/e-commerce-customer-support-qa` by default)
- Maps rows to `Q:/A:` text plus metadata
- Chunks documents (`chunk_size=500`, `chunk_overlap=50`)
- Calls the embedding model (Ollama by default) to generate 768‑dim embeddings
- Upserts into the `documents` table with pgvector `VECTOR(768)`

On success you’ll see a summary line like:

```text
Ingestion complete. documents count=<N>
```

## 5. Seed mock orders

The mock order seeding script lives at `scripts/seed_mock_data.py` and is mounted into the backend container.

Run it inside the backend container:

```powershell
docker compose exec -T backend python /app/scripts/seed_mock_data.py
```

This will:

- Insert (or re-use) 50 mock orders into the `orders` table
- Ensure all four statuses are represented: `processing`, `in-transit`, `delivered`, `returned`
- Print a final count:

```text
Seeded mock orders. orders count=<N>
```

You can also use the orchestration script (from a bash-capable shell such as Git Bash or WSL) to run ingestion + seed in one go:

```bash
bash scripts/ingest_data.sh
```

> Note: If you see errors about `set` or `$'\r'`, ensure the script has UNIX line endings (e.g. run `dos2unix scripts/ingest_data.sh`).

## 6. Run Phase 1 tests

Tests are run from the host using the local virtual environment, connecting to the Postgres container via the published port.

1. Ensure the virtual environment is active:

   ```powershell
   .\venv\Scripts\activate
   ```

2. Override `POSTGRES_HOST` so the test code connects to the dockerized Postgres via `localhost`:

   ```powershell
   $env:POSTGRES_HOST = "localhost"
   ```

3. Run the focused Phase 1 tests:

   ```powershell
   .\venv\Scripts\python -m pytest tests/test_seed_mock_data.py tests/test_ingest_transform.py tests/test_vector_query_smoke.py -q
   ```

   - `test_seed_mock_data.py` validates that seeding creates at least 50 orders, includes all statuses, and keeps `order_number` unique.
   - `test_ingest_transform.py` checks the dataset row → document mapping and chunking behavior.
   - `test_vector_query_smoke.py` (after ingestion) runs a simple pgvector similarity query using an embedded query vector to ensure the end‑to‑end path works.

## 7. Quick verification checklist

After completing the steps above, you should be able to confirm:

- **Docker services**: `docker compose ps` shows `postgres`, `redis`, `ollama`, and `backend` as `Up`.
- **Ingestion**: `docker compose exec -T backend python -m rag.ingest` completes and prints a non‑zero `documents count`.
- **Seeding**: `docker compose exec -T backend python /app/scripts/seed_mock_data.py` completes and prints a non‑zero `orders count`.
- **Tests**: The pytest command exits with status 0 and reports all selected tests as passed.

At this point, Phase 1 (data and storage foundation) is complete and ready to be used by later RAG and agent phases.

