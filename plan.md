# Agentic RAG — E-Commerce Customer Support System
## Project Plan for Cursor IDE

> **Stack Summary:** FastAPI · LangGraph · LangSmith · pgvector · Redis · Ollama / OpenAI / Cerebras · Streamlit · RAGAS · Prometheus · Docker · GCP

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase 0 — Project Bootstrap](#phase-0--project-bootstrap)
5. [Phase 1 — Data & Storage Foundation](#phase-1--data--storage-foundation)
6. [Phase 2 — RAG Core](#phase-2--rag-core)
7. [Phase 3 — Agentic Layer](#phase-3--agentic-layer)
8. [Phase 4 — REST API Backend](#phase-4--rest-api-backend)
9. [Phase 5 — Streamlit Frontend](#phase-5--streamlit-frontend)
10. [Phase 6 — Observability & Evaluation](#phase-6--observability--evaluation)
11. [Phase 7 — GCP Deployment](#phase-7--gcp-deployment)
12. [Environment Variables Reference](#environment-variables-reference)
13. [LLM Provider Config](#llm-provider-config)
14. [Milestones & Checklist](#milestones--checklist)

---

## 1. Project Overview

Build a production-ready agentic RAG system for e-commerce customer support that can:

- Answer product and policy questions via semantic retrieval (RAG)
- Look up order status via a mock Order API tool
- Initiate returns and refunds via a mock Returns API tool
- Escalate unresolved tickets to human agents
- Be evaluated continuously with RAGAS
- Be observed end-to-end via LangSmith and Prometheus

**Primary Dataset:** `Wix/WixQA` from Hugging Face. Provides real KB help articles as the document store (Tier 1) and three tiers of grounded QA pairs for RAGAS evaluation. The system is fully functional using only this dataset.

**Secondary Dataset (Optional):** `rjac/e-commerce-customer-support-qa` — can be added in later stages as a supplementary historical Q&A store (Tier 2) to extend retrieval coverage with resolved support conversations. The system does not depend on it.

---

## 2. Repository Structure

```
ecom-support-rag/
├── .env.example
├── .env                          # gitignored
├── docker-compose.yml
├── plan.md                       # this file
│
├── infra/
│   ├── postgres/
│   │   └── init.sql              # schema + pgvector extension
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                   # FastAPI entrypoint
│   ├── config.py                 # LLM provider config (switchable)
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py           # POST /chat
│   │   │   ├── sessions.py       # GET/DELETE /sessions
│   │   │   └── health.py         # GET /health + metrics
│   │   └── models.py             # Pydantic schemas
│   ├── agent/
│   │   ├── graph.py              # LangGraph state machine
│   │   ├── nodes.py              # Node functions (retrieve, plan, act, respond)
│   │   ├── state.py              # AgentState TypedDict
│   │   └── prompts.py            # System prompts per node
│   ├── tools/
│   │   ├── order_lookup.py       # Mock order API tool
│   │   ├── return_initiate.py    # Mock returns API tool
│   │   ├── faq_search.py         # Vector search tool
│   │   └── ticket_create.py      # Human escalation tool
│   ├── rag/
│   │   ├── ingest_wixqa.py       # PRIMARY: WixQA KB articles → chunking → pgvector
│   │   ├── ingest_rjac.py        # OPTIONAL: rjac Q&A pairs → pgvector (Tier 2)
│   │   ├── chunker.py            # Section-aware + parent-document chunking logic
│   │   ├── retriever.py          # pgvector semantic search + reranking (searches both tiers)
│   │   └── embeddings.py         # Embedding model abstraction
│   ├── db/
│   │   ├── postgres.py           # SQLAlchemy async engine
│   │   ├── redis_client.py       # Redis cache client
│   │   └── models.py             # ORM models
│   └── observability/
│       ├── langsmith_tracer.py
│       └── prometheus_metrics.py
│
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                    # Streamlit UI
│
├── evaluation/
│   ├── ragas_eval.py             # RAGAS evaluation runner
│   ├── build_wixqa_testset.py    # PRIMARY: Build eval set from WixQA ExpertWritten + Simulated
│   ├── build_rjac_testset.py     # OPTIONAL: Build supplementary eval set from rjac dataset
│   └── results/                  # JSON eval outputs
│
└── scripts/
    ├── ingest_wixqa.sh           # PRIMARY: ingest WixQA KB articles
    ├── ingest_rjac.sh            # OPTIONAL: ingest rjac Q&A (run after wixqa)
    └── seed_mock_data.py         # Seed orders/users for mock tools
```

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
│                      Streamlit Frontend                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend                             │
│   POST /chat  ·  GET /sessions  ·  GET /health  ·  /metrics    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              LangGraph Agent (Stateful Graph)                   │
│                                                                 │
│  [intent_classify] → [retrieve_context] → [plan_action]        │
│        ↓                    ↓                   ↓              │
│  [tool_executor] ←──────────────────────── [tool_select]       │
│        ↓                                                        │
│  [synthesize_response] → [escalation_check] → [respond]        │
│                                                                 │
│  Tools: order_lookup · return_initiate · faq_search ·          │
│         ticket_create                                           │
└──────────────┬──────────────────┬───────────────────────────────┘
               │                  │
  ┌────────────▼──┐   ┌──────────▼───────────────────────────────┐
  │  Redis Cache  │   │             Data Layer                    │
  │ (recent Q&A)  │   │  PostgreSQL + pgvector                   │
  └───────────────┘   │  · documents table — Tier 1              │
                      │    (WixQA KB articles, policy docs)      │
                      │  · documents table — Tier 2 (optional)   │
                      │    (rjac resolved Q&A pairs)             │
                      │  · orders table (mock data)              │
                      │  · sessions table                        │
                      │  · tickets table                         │
                      └──────────────────────────────────────────┘
                      
  LangSmith ←── traces from every agent run
  Prometheus ←── metrics from FastAPI + agent
```

---

## Phase 0 — Project Bootstrap

**Goal:** Reproducible local dev environment with all services running.

### Tasks

- [ ] `P0-1` Initialize Git repo, `.gitignore`, `.env.example`
- [ ] `P0-2` Write `docker-compose.yml` with the following services:
  - `postgres` — postgres:16 with pgvector extension, port 5432
  - `redis` — redis:7-alpine, port 6379
  - `ollama` — ollama/ollama, port 11434, with GPU passthrough comment
  - `backend` — build from `backend/Dockerfile`, port 8000
  - `frontend` — build from `frontend/Dockerfile`, port 8501
  - `prometheus` — prom/prometheus, port 9090
  - `grafana` — grafana/grafana, port 3000
- [ ] `P0-3` Write `backend/Dockerfile` (Python 3.11-slim, pip install, non-root user)
- [ ] `P0-4` Write `frontend/Dockerfile` (Python 3.11-slim, Streamlit)
- [ ] `P0-5` Write `infra/postgres/init.sql`:
  - Enable `pgvector` extension
  - Create `documents`, `sessions`, `messages`, `orders`, `tickets` tables
- [ ] `P0-6` Write `backend/config.py` — LLM provider switcher (see [LLM Provider Config](#llm-provider-config))
- [ ] `P0-7` Confirm `docker compose up` boots all services cleanly

### Docker Compose Skeleton

```yaml
# docker-compose.yml
version: "3.9"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ecom_support
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes: ["ollama_models:/root/.ollama"]
    # Uncomment for GPU: deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] } } }

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, redis, ollama]
    volumes: ["./backend:/app"]

  frontend:
    build: ./frontend
    ports: ["8501:8501"]
    env_file: .env
    depends_on: [backend]

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes: ["./infra/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml"]

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    depends_on: [prometheus]

volumes:
  pgdata:
  ollama_models:
```

---

## Phase 1 — Data & Storage Foundation

**Goal:** Load the WixQA KB corpus as the primary document store, chunk it with structure-awareness, embed it, and store in pgvector. Seed mock relational data. The `rjac` dataset is explicitly out of scope here and handled in Phase 1b (optional).

### Tasks

- [ ] `P1-1` Install dependencies: `datasets`, `langchain`, `langchain-community`, `pgvector`, `sqlalchemy[asyncio]`, `asyncpg`, `pypdf`, `beautifulsoup4`
- [ ] `P1-2` Write `backend/rag/embeddings.py` — abstraction over embedding models:
  - Ollama: `nomic-embed-text` (local, 768-dim)
  - OpenAI: `text-embedding-3-small` (1536-dim)
  - Selected via `EMBEDDING_PROVIDER` env var
  - **Note:** embedding dimension must match `VECTOR(n)` in schema — set via `EMBEDDING_DIM` env var (default 768)
- [ ] `P1-3` Write `backend/rag/chunker.py` — section-aware chunking:
  - WixQA articles have titles and structured HTML/markdown body
  - Use LangChain `RecursiveCharacterTextSplitter` with `HTMLHeaderTextSplitter` for structure
  - Implement parent-document chunking: store full article as parent, split into child chunks for embedding
  - Child chunks carry `parent_id` pointer for context expansion at retrieval time
  - Chunk size: 400 tokens, overlap: 50 tokens
- [ ] `P1-4` Write `backend/rag/ingest_wixqa.py`:
  - Load `Wix/WixQA` corpus split: `load_dataset("Wix/WixQA", "wix_kb_corpus")`
  - For each article: extract `title`, `body`, `url`, `category` fields
  - Run through `chunker.py` to produce child chunks with parent references
  - Generate embeddings via `embeddings.py`
  - Upsert into `documents` table with `source="wixqa"`, `doc_tier=1` metadata
  - Hold out nothing here — the WixQA eval QA pairs are separate splits, not derived from a held-out portion of the KB corpus
- [ ] `P1-5` Write `backend/db/postgres.py` — async SQLAlchemy engine + session factory
- [ ] `P1-6` Write `backend/db/models.py` — ORM models for all tables
- [ ] `P1-7` Write `scripts/seed_mock_data.py` — seed 50 mock orders with statuses
- [ ] `P1-8` Write `scripts/ingest_wixqa.sh` — runs `ingest_wixqa.py` then `seed_mock_data.py`
- [ ] `P1-9` Validate: query pgvector with a sample embedding and confirm top-k returns semantically relevant WixQA articles

### Phase 1b — Optional: rjac Dataset (Tier 2)

> **This phase is entirely optional.** The system is fully functional without it. Add this after Phase 6 is stable and you want to extend retrieval coverage with resolved support Q&A.

- [ ] `P1b-1` Write `backend/rag/ingest_rjac.py`:
  - Load `rjac/e-commerce-customer-support-qa`
  - Extract `qa.knowledge[]` pairs: format as `"Q: {question}\nA: {solution}"`
  - Embed and upsert with `source="rjac"`, `doc_tier=2` metadata
  - This populates the **same `documents` table** — retriever searches both tiers transparently
- [ ] `P1b-2` Write `scripts/ingest_rjac.sh`
- [ ] `P1b-3` Validate: confirm retriever returns a mix of Tier 1 (KB articles) and Tier 2 (Q&A) results for the same query, and that `doc_tier` metadata is visible in API responses

### Database Schema

```sql
-- infra/postgres/init.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- EMBEDDING_DIM: 768 for nomic-embed-text (Ollama), 1536 for OpenAI text-embedding-3-small
-- Set the correct dimension before running. Default below is 768.

CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,               -- child chunk text
    parent_id   UUID,                        -- pointer to parent article (self-referential)
    embedding   VECTOR(768),                 -- change to 1536 if using OpenAI embeddings
    source      VARCHAR(50),                 -- "wixqa" | "rjac"
    doc_tier    INTEGER DEFAULT 1,           -- 1 = KB article (primary), 2 = Q&A pair (optional)
    category    VARCHAR(100),
    source_id   VARCHAR(200),                -- original article URL or rjac row id
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON documents (doc_tier);        -- fast filter by tier
CREATE INDEX ON documents (source);

CREATE TABLE sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     VARCHAR(100),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    role        VARCHAR(20) NOT NULL,        -- user | assistant | tool
    content     TEXT NOT NULL,
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_number    VARCHAR(50) UNIQUE NOT NULL,
    user_id         VARCHAR(100),
    status          VARCHAR(50),             -- processing | in-transit | delivered | returned
    items           JSONB,
    total_amount    DECIMAL(10,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    estimated_delivery TIMESTAMPTZ
);

CREATE TABLE tickets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    issue_type  VARCHAR(100),
    summary     TEXT,
    status      VARCHAR(50) DEFAULT 'open',
    priority    VARCHAR(20) DEFAULT 'normal',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Phase 2 — RAG Core

**Goal:** Working retrieval pipeline with semantic search, metadata filtering, and optional reranking.

### Tasks

- [ ] `P2-1` Write `backend/rag/retriever.py`:
  - `similarity_search(query, top_k=5, category_filter=None, tier_filter=None)` — cosine similarity via pgvector
  - `tier_filter=1` searches only WixQA KB articles; `tier_filter=None` searches all tiers (used when rjac is ingested)
  - `rerank(query, docs)` — cross-encoder reranking using `sentence-transformers` (local)
  - Return list of `Document(content, metadata, score, source, doc_tier)`
  - Parent-document expansion: if a child chunk scores highly, optionally fetch its parent article for broader context
- [ ] `P2-2` Write `backend/db/redis_client.py`:
  - Cache key: `md5(query + filters)`
  - TTL: 300 seconds (configurable via `REDIS_CACHE_TTL` env var)
  - `get_cached(key)` / `set_cached(key, value)`
  - Cache hit short-circuits the pgvector call entirely
- [ ] `P2-3` Add metadata filtering support — filter by `category` (derived from WixQA article categories) and by `doc_tier` (1=KB articles, 2=rjac Q&A pairs)
- [ ] `P2-4` Write unit tests in `evaluation/` to verify retrieval quality on 20 sample queries drawn from WixQA ExpertWritten eval split
- [ ] `P2-5` Log retrieval latency to Prometheus counter/histogram (see Phase 6)

### Retriever Interface

```python
# backend/rag/retriever.py

class Retriever:
    async def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        use_cache: bool = True,
        rerank: bool = True,
    ) -> list[RetrievedDoc]:
        ...
```

---

## Phase 3 — Agentic Layer

**Goal:** LangGraph state machine that orchestrates intent detection, retrieval, tool use, response synthesis, and escalation.

### Tasks

- [ ] `P3-1` Write `backend/agent/state.py` — `AgentState` TypedDict:
  - `messages`, `session_id`, `intent`, `retrieved_docs`, `tool_calls`, `tool_results`, `final_response`, `should_escalate`
- [ ] `P3-2` Write `backend/agent/prompts.py` — system prompts for each node
- [ ] `P3-3` Write `backend/agent/nodes.py` — one async function per node:
  - `classify_intent` — categorize into: order_status, return_request, product_qa, account_issue, complaint, other
  - `retrieve_context` — call `Retriever.search()` with intent-based category hint
  - `plan_action` — LLM decides which tool(s) to call, or goes straight to response
  - `execute_tool` — dispatches to appropriate tool function
  - `synthesize_response` — LLM generates grounded answer with retrieved context + tool results
  - `check_escalation` — rule-based: escalate if frustration detected, tool failed twice, or intent=complaint with no resolution
  - `create_ticket` — if escalation triggered, call `ticket_create` tool and notify user
- [ ] `P3-4` Write `backend/agent/graph.py` — assemble the LangGraph:
  - Define nodes, edges, and conditional routing
  - Attach LangSmith tracer
  - Compile graph with `checkpointer` for session memory
- [ ] `P3-5` Write the four tools in `backend/tools/`:
  - `order_lookup.py` — queries `orders` table by order number or user_id
  - `return_initiate.py` — validates eligibility (order must be delivered, < 30 days old), updates order status to `return_initiated`, returns RMA number
  - `faq_search.py` — thin wrapper over `Retriever.search()` as a LangGraph tool
  - `ticket_create.py` — inserts into `tickets` table, returns ticket ID
- [ ] `P3-6` Test full agent round-trip locally with 5 manual queries covering each use case

### LangGraph Node Flow

```
START
  │
  ▼
classify_intent
  │
  ▼
retrieve_context  ──(low relevance?)──► create_ticket → END
  │
  ▼
plan_action
  │
  ├──(needs tool)──► execute_tool ──► synthesize_response
  │                                         │
  └──(no tool needed)──────────────────────►┤
                                            │
                                    check_escalation
                                            │
                              ┌─────────────┴──────────────┐
                              │                            │
                         (resolved)               (escalate)
                              │                            │
                           respond                  create_ticket
                              │                            │
                             END                          END
```

### Tool Schema Example

```python
# backend/tools/order_lookup.py
from langchain_core.tools import tool

@tool
async def order_lookup(order_number: str) -> dict:
    """Look up the status and details of a customer order by order number."""
    # Queries orders table in PostgreSQL
    ...
```

---

## Phase 4 — REST API Backend

**Goal:** Production-grade FastAPI server exposing the agent over HTTP with proper session management, error handling, and metrics.

### Tasks

- [ ] `P4-1` Write `backend/main.py` — FastAPI app with lifespan (db pool init/close), CORS, middleware
- [ ] `P4-2` Write `backend/api/models.py` — Pydantic request/response schemas
- [ ] `P4-3` Write `backend/api/routes/chat.py`:
  - `POST /chat` — accepts `{session_id, user_id, message}`, runs agent graph, returns `{response, session_id, sources, tool_calls_used}`
  - `POST /chat/stream` — SSE streaming version (stretch goal)
- [ ] `P4-4` Write `backend/api/routes/sessions.py`:
  - `GET /sessions/{session_id}/history` — return full message history
  - `DELETE /sessions/{session_id}` — clear session
- [ ] `P4-5` Write `backend/api/routes/health.py`:
  - `GET /health` — service health check
  - `GET /metrics` — Prometheus metrics endpoint (via `prometheus_client`)
- [ ] `P4-6` Add request-level Redis caching: if identical query asked within TTL window in same session, return cached response
- [ ] `P4-7` Add global exception handler — return structured error JSON, log to Prometheus error counter
- [ ] `P4-8` Write `backend/api/routes/admin.py` (optional but useful):
  - `POST /admin/ingest` — trigger re-ingestion
  - `GET /admin/tickets` — list open escalation tickets

### API Contracts

```
POST /chat
  Body:  { session_id: str | null, user_id: str, message: str }
  Response: {
    session_id: str,
    response: str,
    sources: [{ content: str, category: str, score: float }],
    tools_used: [str],
    escalated: bool,
    ticket_id: str | null
  }

GET /sessions/{session_id}/history
  Response: { messages: [{ role, content, created_at }] }

GET /health
  Response: { status: "ok", postgres: bool, redis: bool, ollama: bool }
```

---

## Phase 5 — Streamlit Frontend

**Goal:** Clean, functional chat UI with session management, source display, and tool activity visibility.

### Tasks

- [ ] `P5-1` Write `frontend/app.py` with:
  - Sidebar: session selector (new / resume), user ID input, LLM provider badge
  - Chat area: message history with role-based styling
  - Input: text box + send button
  - Source panel: collapsible expander showing retrieved docs with relevance scores
  - Tool activity: show which tools were called (e.g. "🔍 Searched orders", "📦 Return initiated")
  - Escalation banner: red alert if ticket was created
- [ ] `P5-2` Backend communication via `requests` to `POST /chat` and `GET /sessions/{id}/history`
- [ ] `P5-3` Session state in `st.session_state` (session_id, user_id, message_history)
- [ ] `P5-4` Add `st.sidebar` config panel:
  - LLM provider selector (Ollama / OpenAI / Cerebras)
  - Show/hide sources toggle
  - Clear conversation button

---

## Phase 6 — Observability & Evaluation

**Goal:** Full traceability of agent runs via LangSmith, operational metrics via Prometheus, and RAG quality scores via RAGAS.

### LangSmith

- [ ] `P6-1` Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` in `.env`
- [ ] `P6-2` Wrap `graph.invoke()` calls so every agent run creates a trace in LangSmith with:
  - Input message
  - Each node's input/output
  - Tool calls and results
  - Final response + latency

### Prometheus Metrics

- [ ] `P6-3` Write `backend/observability/prometheus_metrics.py` and instrument:

```python
# Metrics to track
request_count          = Counter("chat_requests_total", "Total chat requests", ["status"])
request_latency        = Histogram("chat_latency_seconds", "Request latency")
retrieval_latency      = Histogram("retrieval_latency_seconds", "RAG retrieval latency")
cache_hits             = Counter("redis_cache_hits_total", "Redis cache hits")
tool_calls             = Counter("agent_tool_calls_total", "Tool calls", ["tool_name"])
escalations            = Counter("escalations_total", "Human escalations triggered")
llm_tokens             = Counter("llm_tokens_total", "LLM tokens used", ["provider"])
```

- [ ] `P6-4` Configure `infra/prometheus/prometheus.yml` to scrape `backend:8000/metrics`
- [ ] `P6-5` Import Grafana dashboard for FastAPI + LangGraph metrics

### RAGAS Evaluation

WixQA provides three ready-made eval splits that map directly to RAGAS metrics without needing to hold out anything from the KB corpus. The KB corpus and eval QA pairs are separate dataset splits.

| WixQA Split | Rows | Use |
|---|---|---|
| `wixqa_expertwritten` | 200 | Primary eval — real user queries, expert answers. Hardest. |
| `wixqa_simulated` | 200 | Secondary eval — expert-validated, good for regression testing |
| `wixqa_synthetic` | 6,222 | Scale testing — LLM-generated, use for stress-testing retrieval |

- [ ] `P6-6` Write `evaluation/build_wixqa_testset.py`:
  - Load `wixqa_expertwritten` and `wixqa_simulated` splits
  - For each row: extract `question`, `answer` (ground truth), `supporting_article` (ground truth context)
  - Save as `evaluation/wixqa_testset.json`
  - This is the **primary eval set** — used for every RAGAS run
- [ ] `P6-7` Write `evaluation/ragas_eval.py`:
  - Load `wixqa_testset.json`
  - For each question: run the full retrieval pipeline, collect `answer`, `contexts`
  - Score with RAGAS: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
  - Save scores to `evaluation/results/run_{timestamp}.json`
  - Print summary table with per-split breakdown (ExpertWritten vs Simulated)
- [ ] `P6-8` Add `make eval` shortcut in `Makefile` or `scripts/`
- [ ] `P6-9` Target baseline scores: faithfulness > 0.75, answer_relevancy > 0.80

**Optional — Phase 6b: rjac Evaluation Extension**
> Add only after Phase 1b (rjac ingestion) is complete.
- [ ] `P6b-1` Write `evaluation/build_rjac_testset.py`:
  - Hold out 20% of `rjac` rows (never ingested in Phase 1b)
  - Extract `qa.knowledge[]` pairs as `{question, ground_truth}`
  - Save as `evaluation/rjac_testset.json`
- [ ] `P6b-2` Extend `ragas_eval.py` to optionally run against `rjac_testset.json` and produce a combined report showing Tier 1 vs Tier 2 contribution to answer quality

### RAGAS Evaluation Runner Outline

```python
# evaluation/ragas_eval.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Load wixqa_testset.json
# For each question: run RAG pipeline → collect {question, answer, contexts, ground_truths}
# Run: evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
# Export: evaluation/results/run_{timestamp}.json
# Optional: if --rjac flag passed, also evaluate against rjac_testset.json
```

---

## Phase 7 — GCP Deployment

**Goal:** Deploy all services to GCP using Cloud Run (stateless services) and managed GCP services.

### Infrastructure Map

| Local (Docker)     | GCP Equivalent                          |
|--------------------|-----------------------------------------|
| postgres+pgvector  | Cloud SQL (PostgreSQL 16 + pgvector)    |
| redis              | Memorystore (Redis)                     |
| ollama             | Cloud Run (GPU tier) or drop for OpenAI |
| backend            | Cloud Run                               |
| frontend           | Cloud Run                               |
| prometheus         | Cloud Monitoring + Managed Prometheus   |
| grafana            | Cloud Monitoring Dashboards             |

### Tasks

- [ ] `P7-1` Create `infra/gcp/` directory with Terraform or gcloud CLI scripts
- [ ] `P7-2` Provision Cloud SQL instance with pgvector, import schema
- [ ] `P7-3` Provision Memorystore Redis instance
- [ ] `P7-4` Push Docker images to Artifact Registry:
  ```bash
  docker build -t gcr.io/{PROJECT}/backend:latest ./backend
  docker push gcr.io/{PROJECT}/backend:latest
  ```
- [ ] `P7-5` Deploy backend to Cloud Run:
  - Min instances: 1, Max: 10
  - Set all env vars as Cloud Run secrets
  - VPC connector to reach Cloud SQL and Memorystore
- [ ] `P7-6` Deploy frontend to Cloud Run (stateless Streamlit)
- [ ] `P7-7` Set up Cloud Managed Service for Prometheus + connect to backend `/metrics`
- [ ] `P7-8` Configure LangSmith (remains cloud-hosted — no GCP migration needed)
- [ ] `P7-9` Set up Cloud Armor (WAF) and IAP if needed for frontend auth
- [ ] `P7-10` Configure CI/CD: Cloud Build trigger on `main` branch push

### Cloud Run Deployment Command

```bash
gcloud run deploy ecom-support-backend \
  --image gcr.io/${PROJECT_ID}/backend:latest \
  --region us-central1 \
  --platform managed \
  --set-env-vars LLM_PROVIDER=openai \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest" \
  --min-instances 1 \
  --max-instances 10 \
  --allow-unauthenticated
```

---

## Environment Variables Reference

```bash
# .env.example

# LLM Provider — "ollama" | "openai" | "cerebras"
LLM_PROVIDER=ollama

# Ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Cerebras
CEREBRAS_API_KEY=...
CEREBRAS_MODEL=llama3.1-8b

# Embeddings — "ollama" | "openai"
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL_OLLAMA=nomic-embed-text
EMBEDDING_MODEL_OPENAI=text-embedding-3-small

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ecom_support
POSTGRES_USER=admin
POSTGRES_PASSWORD=changeme

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_CACHE_TTL=300

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=ecom-support-rag

# HuggingFace — Primary Dataset
HF_DATASET_PRIMARY=Wix/WixQA

# HuggingFace — Secondary Dataset (optional, used in Phase 1b)
# HF_DATASET_SECONDARY=rjac/e-commerce-customer-support-qa

# Embedding dimension — must match VECTOR(n) in schema
# 768 for nomic-embed-text (Ollama), 1536 for OpenAI text-embedding-3-small
EMBEDDING_DIM=768

# App
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:8501
```

---

## LLM Provider Config

All LLM calls are routed through a single factory in `backend/config.py`. Switching providers requires only changing the `LLM_PROVIDER` env var — no code changes.

```python
# backend/config.py

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_cerebras import ChatCerebras  # community package
import os

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    match provider:
        case "openai":
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        case "cerebras":
            return ChatCerebras(
                model=os.getenv("CEREBRAS_MODEL", "llama3.1-8b"),
                api_key=os.getenv("CEREBRAS_API_KEY"),
            )
        case "ollama" | _:
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

def get_embedding_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL_OPENAI", "text-embedding-3-small"))
    else:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_OLLAMA", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
```

---

## Milestones & Checklist

| Phase | Milestone | Deliverable | Done |
|-------|-----------|-------------|------|
| 0 | Local env boots | `docker compose up` — all services green | ☐ |
| 1 | WixQA ingested | KB corpus embedded in pgvector (Tier 1) | ☐ |
| 1 | Mock data seeded | 50 orders queryable in postgres | ☐ |
| 1b *(opt)* | rjac ingested | Q&A pairs added to pgvector as Tier 2 | ☐ |
| 2 | RAG working | Top-k retrieval over WixQA KB confirmed, Redis caching confirmed | ☐ |
| 3 | Agent working | All 4 tools callable, full round-trip for each use case | ☐ |
| 3 | Escalation working | Frustrated user triggers ticket creation | ☐ |
| 4 | API live | `POST /chat` returns valid response with sources + `doc_tier` in response | ☐ |
| 5 | UI live | Streamlit chat with tool activity visible | ☐ |
| 6 | Traces in LangSmith | Every agent run traced end-to-end | ☐ |
| 6 | Metrics in Prometheus | All counters/histograms populating | ☐ |
| 6 | RAGAS baseline (WixQA) | Faithfulness > 0.75, Relevancy > 0.80 on ExpertWritten split | ☐ |
| 6b *(opt)* | RAGAS extended (rjac) | Combined Tier 1 + Tier 2 eval report generated | ☐ |
| 7 | GCP deployed | Backend + Frontend on Cloud Run, managed DB + Redis | ☐ |

---

## Developer Notes for Cursor

- Each Phase maps to a logical feature branch: `feat/phase-0-bootstrap`, `feat/phase-1-data`, etc.
- Use `# TODO(P3-2)` inline comments to link code to plan task IDs
- The `.env` file is gitignored; copy `.env.example` and fill in secrets
- Run `docker compose up --build` after any Dockerfile or requirements change
- Primary ingest (required): `bash scripts/ingest_wixqa.sh`
- Optional Tier 2 ingest: `bash scripts/ingest_rjac.sh` — run only after WixQA ingest is confirmed working
- Ollama models must be pulled manually inside the container: `docker exec -it ecom-support-rag-ollama-1 ollama pull llama3.2 && ollama pull nomic-embed-text`
- **Embedding dimension warning:** if switching from Ollama (768-dim) to OpenAI (1536-dim), you must drop and recreate the `documents` table with the updated `VECTOR(n)` size and re-run ingest
