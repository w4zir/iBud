## Use-Case Testing Guide

This document focuses on **use-case tests**: end-to-end scenarios that exercise the agent, API, and persistence through realistic workflows such as chatting via `/chat` and inspecting session history.

All examples assume a PowerShell shell on Windows from the project root.

## 1. Prerequisites

- Python 3.11+ and a virtual environment at `venv/`
- Backend dependencies installed:

  ```powershell
  .\venv\Scripts\python -m pip install -r backend\requirements.txt
  ```

- Docker services up (at least `postgres`, `redis`, `ollama`, `backend`)
- WixQA KB ingested into Postgres
- Mock orders seeded

Use `docs/how_to_run.md` for the exact commands to start services and perform ingestion/seeding.

## 2. Automated use-case tests (API-level)

Automated use-case tests are implemented in `tests/test_use_cases.py`. They drive the FastAPI app directly and validate:

- Session creation and message persistence via `POST /chat`
- Request-level Redis caching for identical queries in the same session
- Session history retrieval via `GET /sessions/{session_id}/history`

### Run all use-case tests

```powershell
.\venv\Scripts\python -m pytest tests/test_use_cases.py
```

These tests stub the underlying agent graph (`run_agent`) to avoid external LLM calls while still exercising the API surface and database interactions.

## 3. Manual end-to-end scenarios

In addition to automated checks, you can manually validate key user journeys once the backend and frontend are running.

### 3.1 Chat via Streamlit UI

1. Ensure Docker services are running (including `backend` and `frontend`).
2. Open the Streamlit UI at `http://localhost:8501`.
3. In the sidebar:
   - Set a `User ID` (e.g. `user-1`).
   - Optionally toggle “Show sources” on or off.
4. In the main chat area:
   - Ask an order-related question (e.g. “What is the status of my order?”).
   - Ask a return-related question (e.g. “Can I return my last order?”).
   - Ask a product or policy question (e.g. “What is your return policy?”).
5. Observe:
   - Assistant responses appear under your messages.
   - When applicable, a “Tools used” line indicates which tools were called.
   - When the agent escalates, a banner indicates a ticket was created and shows the ticket ID.

### 3.2 Chat via HTTP API

You can also call the API directly using `curl` or `Invoke-RestMethod`.

```powershell
curl -X POST http://localhost:8000/chat/ `
  -H "Content-Type: application/json" `
  -d '{ "session_id": null, "user_id": "manual-user", "message": "Hello" }'
```

The response includes:

- `session_id` — use this to continue the conversation.
- `response` — assistant answer text.
- `sources` — retrieved documents (when available).
- `tools_used` — list of tool names invoked.
- `escalated` / `ticket_id` — escalation status.

### 3.3 Inspect session history

Given a `session_id` from a chat response, fetch the full message history:

```powershell
curl http://localhost:8000/sessions/<session_id>/history
```

You should see a chronological list of `{role, content, created_at}` entries for the session.

## 4. Troubleshooting use-case tests

- **API not reachable (`connection refused`)**
  - Ensure `docker compose ps` shows `backend` as `Up`.
  - Confirm you are calling `http://localhost:8000` (or your configured host/port).

- **Use-case tests failing due to database errors**
  - Verify Postgres is running and accessible from the host.
  - Confirm ingestion and seeding have completed successfully.

- **Streamlit UI cannot reach backend**
  - Check that `BACKEND_BASE_URL` (if set) points to the correct backend URL.
  - Ensure CORS is configured via `CORS_ORIGINS` in `.env` (e.g. `http://localhost:8501`).

