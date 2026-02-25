## Overview

This document describes the quality assurance (QA) strategy for the iBud agentic RAG system. It complements `docs/how_to_run.md` and `docs/how_to_test.md` by explaining **what** we validate, **where**, and **how** we decide a change is safe to ship.

The goals are:

- Ensure the agent answers are **correct, grounded, and safe**.
- Keep the **API and data pipelines stable** across changes.
- Provide **repeatable checks** that can be run by any engineer.

## QA scope

- **Backend API (`backend/`)**
  - FastAPI endpoints (`/chat`, `/sessions`, `/health`, `/admin`).
  - LangGraph agent (intent classification, retrieval, planning, tools, escalation).
  - RAG pipeline (chunking, embeddings, pgvector similarity, Redis caching, reranking).
  - Observability hooks (Prometheus metrics, LangSmith tracing).
- **Frontend (`frontend/`)**
  - Streamlit chat UI, session handling, error surfacing.
- **Data & infra**
  - Postgres schema (`documents`, `sessions`, `messages`, `orders`, `tickets`).
  - Redis cache behaviour.
  - Ollama models (chat + embeddings).
- **Evaluation**
  - WixQA testset generation.
  - RAGAS evaluation metrics and regression tracking.

## QA layers and ownership

- **Unit tests** (developer-owned)
  - Validate small units: chunker, retriever, Redis client, seed scripts, observability helpers, evaluation utilities.
  - Run locally on every non-trivial change:

    ```powershell
    .\venv\Scripts\python -m pytest
    ```

- **Integration tests** (backend + data)
  - Validate behaviour with live Postgres/Redis and ingested WixQA data.
  - Focus on vector search, retrieval quality, and data access.
  - Run when changing database models, ingestion, retrieval, or cache logic:

    ```powershell
    .\venv\Scripts\python -m pytest -m integration
    ```

- **API-level use-case tests** (end-to-end without UI)
  - Drive the FastAPI app and validate:
    - Session creation and message persistence.
    - Cache hits for repeated queries.
    - Correct session history retrieval.
  - Run whenever changing `/chat` flow, agent graph, or session storage:

    ```powershell
    .\venv\Scripts\python -m pytest tests/test_use_cases.py
    ```

- **Manual UI testing**
  - Validate the Streamlit front-end and perceived UX for key journeys (see checklist below).

- **RAG quality evaluation (RAGAS)**
  - Validate answer quality and grounding against the WixQA corpus.
  - Run on demand before major releases or after substantial RAG/LLM changes.

## Environments

- **Local dev**
  - Run selected Docker services (`postgres`, `redis`, `ollama`, `backend`, `frontend`) and tests from your machine.
  - Used for fast iteration and debugging.
- **Shared / CI environment** (if configured)
  - Run the same test commands headless.
  - Recommended to compute RAGAS metrics here and track score history.

All QA commands assume:

- Python 3.11+ virtual environment at `venv/`.
- Backend requirements installed:

  ```powershell
  .\venv\Scripts\python -m pip install -r backend\requirements.txt
  ```

## Manual end-to-end QA checklist

With full stack running (`postgres`, `redis`, `ollama`, `backend`, `frontend`), validate:

- **Order-related flows**
  - Ask “What is the status of my order?” with a valid `user_id`.
  - Verify:
    - The agent calls `order_lookup` (see “Tools used” in UI if enabled).
    - The response includes plausible status and delivery information.
    - No unhandled errors in backend logs.

- **Return-related flows**
  - Ask “Can I return my last order?”.
  - Verify:
    - The agent calls `order_lookup` and `return_initiate` as needed.
    - The response correctly reflects return eligibility and next steps.

- **Product / policy questions**
  - Ask questions like “What is your return policy?” or “How long is the warranty?”.
  - Verify:
    - Retrieved sources are WixQA articles with relevant titles/content.
    - Answers are grounded in the provided context and not hallucinated.

- **Account / complaint flows**
  - Ask account-related and complaint-style messages (e.g. “I can’t log in”, “I am very unhappy with your service”).
  - Verify:
    - Intent classification is sensible (`account_issue`, `complaint`).
    - Escalation triggers when appropriate and a ticket is created.

- **Session history**
  - Start a conversation via UI, then fetch history via:

    ```powershell
    curl http://localhost:8000/sessions/<session_id>/history
    ```

  - Verify message ordering, roles, and timestamps.

- **Observability**
  - Hit `http://localhost:8000/metrics` and confirm the main counters and histograms update while chatting.
  - If Prometheus/Grafana are enabled, confirm dashboards show traffic, latency, cache hits, tool calls, escalations, and token counts.
  - With LangSmith enabled, verify new traces appear with `session_id`, `user_id`, and intent metadata.

## RAG quality QA (RAGAS)

When changing retrieval, chunking, embeddings, or LLM configuration:

1. **Build the evaluation testset:**

   ```powershell
   .\venv\Scripts\python -m evaluation.build_wixqa_testset
   ```

2. **Run the evaluation:**

   ```powershell
   .\venv\Scripts\python -m evaluation.ragas_eval --backend-url http://localhost:8000 --limit 50
   ```

3. **Assess results under `evaluation/results/`:**
   - Track `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
   - Block changes that significantly regress these metrics unless there is a documented trade-off.

## Release / merge checklist

Before merging or releasing:

- **Code-level**
  - Unit tests pass.
  - For changes that touch DB, retrieval, or caching: integration tests pass.
  - For changes to `/chat` or sessions: use-case tests pass.

- **Behaviour-level**
  - Manual end-to-end scenarios work in UI (at least one order, return, and policy question).
  - No critical errors in backend logs during smoke tests.

- **Quality-level**
  - For RAG-related changes: RAGAS metrics are within acceptable ranges or improved.
  - Observability is intact (metrics and traces still recorded).

If any of the above fail, log a defect in your tracking system with:

- Repro steps (including exact prompt and environment).
- Expected vs actual behaviour.
- Relevant logs, stack traces, or screenshots.

