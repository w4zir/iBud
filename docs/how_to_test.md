## Testing Guide

This document explains how to run unit, integration, and evaluation/quality tests for the backend, including the Phase 2 RAG retriever and Redis cache.

All examples assume a PowerShell shell on Windows from the project root.

## 1. Prerequisites

- Python 3.11+ and a virtual environment at `venv/`
- Backend dependencies installed:

  ```powershell
  .\venv\Scripts\python -m pip install -r backend\requirements.txt
  ```

- For integration/evaluation tests:
  - Docker services up (at least `postgres`, `redis`, `ollama`, `backend`)
  - WixQA KB ingested into Postgres
  - Mock orders seeded

## 2. Unit tests (fast, no external services)

These tests run entirely against in-memory mocks and do not require Postgres, Redis, or Ollama.

Run all unit tests (default marker configuration excludes `integration`):

```powershell
.\venv\Scripts\python -m pytest tests/
```

This includes:

- `test_chunker.py` — section-aware and parent-document chunking
- `test_ingest_transform.py` — WixQA row mapping and chunker integration
- `test_vector_query_smoke.py` — embedding client and vector query construction
- `test_redis_client.py` — cache key generation and Redis get/set behavior (mocked)
- `test_retriever.py` — retriever search, caching, parent expansion, and reranking (mocked)

## 3. Integration tests (Postgres/Redis)

Integration tests require a live Postgres and Redis, plus ingested data.

1. Start the core services:

   ```powershell
   docker compose up -d postgres redis ollama backend
   ```

2. Ensure `.env` is configured for Phase 1/2 and that WixQA is ingested and mock orders are seeded (see `docs/how_to_run.md` for commands).

3. From the host, point tests at the containerized Postgres:

   ```powershell
   $env:POSTGRES_HOST = "localhost"
   .\venv\Scripts\python -m pytest tests/ -m "integration"
   ```

Key integration tests:

- `tests/test_seed_mock_data.py` — verifies seeded orders and idempotency

## 4. Retrieval quality tests (Phase 2)

Phase 2 adds evaluation-style tests that validate end-to-end retrieval quality against a small sample of real questions from the WixQA ExpertWritten split.

Prerequisites:

- Docker services: `postgres`, `redis`, `ollama`, `backend` up
- WixQA KB fully ingested into `documents`

Run the retrieval quality test:

```powershell
$env:POSTGRES_HOST = "localhost"
.\venv\Scripts\python -m pytest evaluation/test_retrieval_quality.py -m "integration"
```

This test:

- Draws 20 sample queries from `Wix/WixQA` (`wixqa_expertwritten`)
- Uses the Phase 2 `Retriever` to fetch top-k results
- Asserts that the majority of queries return at least one document

## 5. Suggested workflows

- **During development (fast loop):**

  ```powershell
  .\venv\Scripts\python -m pytest tests/ -k "retriever or redis_client"
  ```

- **Before merging Phase 2 changes:**

  ```powershell
  .\venv\Scripts\python -m pytest tests/
  $env:POSTGRES_HOST = "localhost"
  .\venv\Scripts\python -m pytest tests/ -m "integration"
  .\venv\Scripts\python -m pytest evaluation/test_retrieval_quality.py -m "integration"
  ```

## 6. Troubleshooting

- **Redis connection errors in tests:**
  - Unit tests mock Redis and should not require a live instance.
  - For integration tests, ensure `docker compose ps` shows `redis` as `Up` and that `REDIS_HOST` / `REDIS_PORT` are correct in `.env`.

- **No results in retrieval quality tests:**
  - Confirm WixQA ingestion has run successfully (non-zero `documents` count for `source='wixqa'`).
  - Ensure embeddings were generated with the same `EMBEDDING_DIM` as your schema.

- **HuggingFace dataset download issues:**
  - Check network access.
  - Try running `python -c "from datasets import load_dataset; load_dataset('Wix/WixQA', 'wixqa_expertwritten', split='train')"` inside the backend container to verify access.

