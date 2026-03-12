## Foodpanda Testing Guide

This document describes how to set up and test **Foodpanda-specific** behaviour in the multi-company RAG system.

### 1. Prerequisites

- Backend services running (Postgres, Redis, backend API).
- Foodpanda policy markdown files present under `data/foodpanda/policy_docs/`.
- Python virtual environment with backend requirements installed:

```powershell
.\venv\Scripts\python -m pip install -r backend\requirements.txt
```

### 2. Ingest Foodpanda policy data

From the project root, run:

```powershell
docker compose exec -T backend python -m backend.rag.ingest_foodpanda
```

This reads markdown policies from `data/foodpanda/policy_docs`, chunks and embeds them, and writes `documents` rows with:

- `company_id="foodpanda"`
- `source="foodpanda"`
- `category="policy"`

### 3. Seed Foodpanda orders

To generate mixed demo and Foodpanda orders:

```powershell
docker compose exec -T backend python .\scripts\seed_mock_data.py
```

The seed script alternates `company_id` between `"default"` and `"foodpanda"` so order tools can be exercised per company.

### 4. Running automated Foodpanda tests

#### 4.1 API-level integration test

Run the Foodpanda RAG integration test:

```powershell
.\venv\Scripts\python -m pytest tests/test_foodpanda_rag_integration.py -m integration
```

This sends a Foodpanda query from `data/foodpanda/testing/RAG_Test_Queries.json` to `/chat` with:

- `dataset="foodpanda"`
- `company="foodpanda"`

and verifies the request succeeds end-to-end.

#### 4.2 Full integration suite

To run all integration tests (including order/seed checks):

```powershell
.\venv\Scripts\python -m pytest -m integration
```

### 5. Manual UI testing for Foodpanda

1. Start backend and frontend (see `docs/how_to_run.md`).
2. Open Streamlit UI at `http://localhost:8501`.
3. In the sidebar:
   - Set `User ID` to a known seed user (for example `user-1`).
   - Select **Company** = `Foodpanda`.
   - Select **Knowledge base dataset** = `Foodpanda policies`.
4. In the chat, use the queries from `data/foodpanda/testing/RAG_Test_Queries.json`, for example:
   - Electronics return outside the 15-day window.
   - Shipping delay with force majeure (blizzard).
   - Subscription refund within 48 hours.
   - Loyalty tier benefits around \$1,200 annual spend.
   - Damaged item on arrival.
5. Verify:
   - Answers reference Foodpanda policies, not WixQA/Bitext content.
   - Sources (when shown) include `source="foodpanda"`.
   - No cross-company leakage occurs when switching company between sessions.

### 6. RAG evaluation with a Foodpanda testset (optional)

You can run RAGAS evaluation over a Foodpanda-specific testset once you have created it (for example `evaluation/foodpanda_testset.json`).

```powershell
.\venv\Scripts\python -m evaluation.ragas_eval `
  --backend-url http://localhost:8000 `
  --dataset-key foodpanda `
  --testset-path evaluation/foodpanda_testset.json `
  --limit 50
```

This uses the Foodpanda dataset key so the backend routes retrieval to the Foodpanda policy corpus.

