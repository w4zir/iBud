## Agentic RAG — E‑Commerce Customer Support

This repository implements an agentic RAG system for e‑commerce customer support, following the phased plan in `plan.md`. It combines a FastAPI backend, a stateful workflow-style agent runtime, a pgvector/Redis data layer, and a Streamlit frontend, with observability and evaluation via Prometheus, Grafana, LangSmith, and RAGAS.

### Components

- **Backend (`backend/`)**: FastAPI API server plus workflow-style agent runtime (intent→plan→evaluate→execute→validate→respond), RAG pipeline, tools, and observability.
- **Frontend (`frontend/`)**: Streamlit chat UI for customer support workflows.
- **Infra (`infra/`)**: Postgres schema + migrations, Prometheus config, and Grafana dashboards.
- **Evaluation (`evaluation/`)**: WixQA and Bitext testset builders, the RAGAS evaluation runner, and intent classification evaluation for Bitext.
- **Scripts (`scripts/`)**: Data ingest and mock data seeding.

### Quick Start (Local)

From the project root:

1. **Set up Python environment**

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   .\venv\Scripts\python -m pip install -r backend\requirements.txt
   ```

2. **Configure environment**

   ```powershell
   copy .env.example .env
   ```

   Adjust values in `.env` as needed (LLM provider, Postgres/Redis, LangSmith, etc.).

3. **Start core services**

   ```powershell
   docker compose up -d postgres redis ollama backend frontend prometheus grafana
   ```

4. **Ingest data and seed mocks**

   ```powershell
   # Primary WixQA KB corpus
   docker compose exec -T backend python -m backend.rag.ingest_wixqa

   # Optional Bitext customer-support QA corpus
   docker compose exec -T backend python -m backend.rag.ingest_bitext

   # Seed mock orders for tools
   docker compose exec -T backend python /app/scripts/seed_mock_data.py
   ```

5. **Open the UI**

   - Backend API: `http://localhost:8000`
   - Streamlit frontend: `http://localhost:8501`

### Observability & Evaluation

- **Prometheus** scrapes backend metrics from `/metrics` on the `backend` service.
- **Grafana** can import the dashboard at `infra/grafana/dashboards/phase6-observability.json`.
- **LangSmith** tracing is enabled by setting `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` in `.env`.
- **RAGAS** evaluation:

  ```powershell
  # Build WixQA testset
  .\venv\Scripts\python -m evaluation.build_wixqa_testset

  # (Optional) Build Bitext full + sampled testsets
  .\venv\Scripts\python -m evaluation.build_bitext_testset --mode both --max-per-intent 50

  # Run evaluation against local backend (WixQA)
  .\venv\Scripts\python -m evaluation.ragas_eval --backend-url http://localhost:8000 --limit 50

  # Run Bitext sampled evaluation
  .\venv\Scripts\python -m evaluation.ragas_eval `
    --backend-url http://localhost:8000 `
    --dataset-key bitext `
    --testset-path evaluation/bitext_testset_sampled.json `
    --limit 100
  ```

- **Intent classification (Bitext) evaluation**:

  ```powershell
  .\venv\Scripts\python -m evaluation.intent_eval `
    --backend-url http://localhost:8000 `
    --dataset-key bitext `
    --testset-path evaluation/bitext_testset_sampled.json `
    --limit 200 `
    --experiment-name "bitext-intent-v1" `
    --intent-prompt-profile bitext
  ```

  You can configure a separate judge provider/model for RAGAS via `RAGAS_LLM_PROVIDER` and related env vars; by default it reuses the runtime provider (e.g. local Ollama when `LLM_PROVIDER=ollama`, Cerebras when `LLM_PROVIDER=cerebras`). See `docs/faq.md`, `docs/how_to_run.md`, and `docs/how_to_test.md` for details.

See `docs/how_to_run.md` and `docs/how_to_test.md` for detailed instructions and scenarios.

