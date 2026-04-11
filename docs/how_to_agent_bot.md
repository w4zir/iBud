# How To Run Agent Bot (ModernBERT + no_issue Path)

This guide is focused only on running the current agentic chat system through:

- ModernBERT query classification (`is_issue` vs `no_issue`)
- `no_issue` pipeline execution (`simple_chat` route)

It is intentionally scoped and self-contained.

## What This Covers

For each `/chat` request, the backend currently does this:

1. Calls ModernBERT BentoML classifier (`CLASSIFIER_BENTOML_URL`)
2. Gets `is_issue` decision
3. Maps routing:
   - `is_issue=true` -> `issue`
   - `is_issue=false` -> `non_issue` (`no_issue` behavior)
4. Selects runtime route:
   - `non_issue` -> `run_simple_chat_agent`
   - `issue` -> orchestrated planner path (only if enabled)

In this document, we run and verify the `non_issue` path, and show how to disable the `issue` path using env vars.

## Prerequisites

- Docker Desktop with `docker compose`
- Python virtual environment at `venv/`
- Ollama running on host

## 1) Configure Environment

Create `.env` (or update existing) with at least:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
OLLAMA_SMALL_MODEL=llama3.2

CLASSIFIER_BENTOML_URL=http://modernbert:3000/classify
CLASSIFIER_BENTOML_TIMEOUT_SECONDS=5
CLASSIFIER_THRESHOLD=0.7

AGENT_PLANNING_ENABLED=true
DEFAULT_DATASET=wixqa
DEFAULT_COMPANY=default
```

Keep your existing DB/Redis/Elasticsearch values in `.env` as-is.

## Postgres migrations (agent warehouse and related schema)

The agent writes spans, outcomes, and session analytics to Postgres. If your database was created before the full migration set was applied, run the combined script once (from the repo root, with Postgres up and `.env` loaded so `POSTGRES_USER` and `POSTGRES_DB` match your compose file):

```powershell
Get-Content infra/postgres/migrations/all_migrations.sql -Raw `
  | docker compose exec -T postgres psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB -v ON_ERROR_STOP=1
```

Or use the helper script (same effect):

```powershell
.\scripts\apply_postgres_warehouse.ps1
```

The file `infra/postgres/migrations/all_migrations.sql` concatenates migrations `001`–`006` (documents columns, embedding size, observability tables, analytics views, intent-eval tables, `company_id` indexes). Re-running is mostly safe where the underlying scripts use `IF NOT EXISTS`; migration `002` can clear document embeddings when resizing the vector column—avoid re-running that section on a production DB unless you intend to re-embed.

## 2) Start Required Services

Run from repo root:

```powershell
docker compose up -d postgres redis elasticsearch modernbert backend
```

Check status:

```powershell
docker compose ps
```

Expected `Up`: `postgres`, `redis`, `elasticsearch`, `modernbert`, `backend`.

## 3) Prepare Ollama Models

On host machine:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

## 4) Ingest KB Data (Needed for FAQ/KB-style responses)

```powershell
docker compose exec -T backend python -m backend.rag.ingest_wixqa
```

### Foodpanda policy docs → Elasticsearch (and Postgres)

To index markdown policies from `data/foodpanda/policy_docs` into Elasticsearch (and upsert matching rows in Postgres for retrieval), use the Foodpanda ingest module. It loads every `*.md` file in that folder, chunks content, embeds chunks via your configured embedding client (Ollama when using the stack from this guide), calls `ensure_index()` on the ES index named by `ES_INDEX_NAME` (default `ecom-support-documents`), and `bulk_index`es parent documents plus child chunks.

Prerequisites: Elasticsearch and Postgres are up (same compose stack as section 2), and embeddings are available (Ollama with `nomic-embed-text` per section 3).

```powershell
docker compose exec -T backend python -m backend.rag.ingest_foodpanda
```

- **Source directory:** defaults to `data/foodpanda/policy_docs` inside the repo. Override with `FOODPANDA_POLICY_DIR` if policies live elsewhere.
- **Indexed identity:** documents are stored with `company_id=foodpanda`, `source=foodpanda`, `category=policy`.
- For broader Foodpanda UI and integration-test notes, see `docs/foodpanda_testing.md`.

## 5) Verify ModernBERT + no_issue Routing

Send a clearly non-issue prompt:

```powershell
curl -X POST http://localhost:8000/chat/ `
  -H "Content-Type: application/json" `
  -d "{\"user_id\":\"u1\",\"message\":\"hello there\"}"
```

In current routing logic, non-issue messages are sent to `simple_chat` (no planner workflow).

Optional: inspect backend logs to confirm route selection fields (`query_type`, `route`, classifier label/confidence):

```powershell
docker compose logs backend --tail 200
```

## Disable is_issue Pipeline (Env-Only)

If you want to disable the `is_issue` orchestrated/planner path, set:

```env
AGENT_PLANNING_ENABLED=false
```

Then restart backend:

```powershell
docker compose up -d backend
```

### Important Current-System Behavior

- The ModernBERT classifier is still called on each request.
- With `AGENT_PLANNING_ENABLED=false`, even `is_issue=true` requests are forced to `simple_chat` route.
- This is the available env-based switch in the current codebase for disabling the `is_issue` pipeline behavior.

## Quick Health Check

```powershell
curl http://localhost:8000/health
```

You should see classifier and backend dependencies reported in the JSON response.
