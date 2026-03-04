# Observability TODO

This TODO defines the step-by-step work needed to move the current iBud implementation from basic Phase 6 observability to complete observability and metric tracking aligned with:

- `specs/observability.md` (full target model)
- `specs/plan.md` (Phase 6 baseline requirements)
- current implementation in `backend/`, `evaluation/`, `infra/`, and `docs/`

## Current Baseline (Already Implemented)

- Prometheus metrics and endpoint:
  - `backend/observability/prometheus_metrics.py`
  - `backend/api/routes/health.py` (`/metrics`)
- LangSmith tracing hooks:
  - `backend/observability/langsmith_tracer.py`
  - `backend/agent/graph.py`
- Dashboard and scrape config:
  - `infra/grafana/dashboards/phase6-observability.json`
  - `infra/prometheus/prometheus.yml`
- RAGAS evaluation scripts:
  - `evaluation/build_wixqa_testset.py`
  - `evaluation/ragas_eval.py`

## Completion Definition

Observability is considered complete when:

- Operational, product, business, and AI-quality metrics are captured and queryable.
- Traces provide end-to-end correlation from request -> agent nodes -> tool calls -> outcome.
- Evaluation is asynchronous and continuously produces quality scores.
- Dashboards and alerts support exec, product, and engineering use-cases.
- A warehouse/event model exists for historical analysis and feedback loops.

---

## Phase 1 - Stabilize Existing Metrics and Evaluation

- [ ] Verify and fix RAGAS score reliability in `evaluation/ragas_eval.py` so runs no longer produce `NaN`/empty breakdowns unless explicitly expected.
- [ ] Add defensive handling for missing contexts, malformed backend responses, and evaluator failures in `evaluation/ragas_eval.py`.
- [ ] Add/expand tests in `tests/test_ragas_eval.py` for non-NaN aggregation and per-split output.
- [ ] Ensure `evaluation/results/run_*.json` schema is stable and documented in `docs/how_to_test.md`.
- [ ] Add explicit pass/fail thresholds (minimum valid-row count + score sanity checks) to evaluation output.

Acceptance criteria:

- Evaluation output contains valid numeric metrics for normal runs.
- Failed rows are tracked with reasons; aggregate metrics are computed from valid rows only.

---

## Phase 2 - Expand Prometheus Metric Coverage

- [ ] Extend `backend/observability/prometheus_metrics.py` with missing metrics:
  - Tool success/failure counters by `tool_name`
  - Intent distribution counter by `intent`
  - Chat turns per session histogram
  - Task outcome counters (`completed`, `escalated`, `resolved_without_escalation`)
  - Error counters by `error_type` and `component`
  - Embedding latency histogram
  - Rerank latency histogram
  - Database and Redis operation latency histograms
- [ ] Instrument `backend/agent/nodes.py` for intent, tool outcome, escalation reason, and outcome metrics.
- [ ] Instrument `backend/rag/retriever.py` for embedding/rerank latency and retrieval-stage failure metrics.
- [ ] Instrument `backend/api/routes/chat.py` for turns/session and request outcome labeling.
- [ ] Instrument `backend/main.py` exception handlers with typed error labels.

Acceptance criteria:

- `/metrics` includes new metric families with meaningful labels.
- Core user journeys (order status, returns, product QA, complaint escalation) produce non-zero metric updates in expected series.

---

## Phase 3 - Introduce Structured Logging and Correlation IDs

- [ ] Replace ad-hoc debug logging with structured logging in:
  - `backend/config.py`
  - `backend/api/routes/chat.py`
  - `backend/agent/nodes.py`
  - `backend/rag/retriever.py`
  - `backend/db/redis_client.py`
- [ ] Define consistent log fields: `request_id`, `session_id`, `trace_id`, `user_id`, `intent`, `tool_name`, `status`, `latency_ms`, `error_type`.
- [ ] Add redaction rules for sensitive fields (user content, secrets, tokens).
- [ ] Update `docs/faq.md` and `docs/how_to_run.md` with logging mode guidance (`DEBUG` vs production).

Acceptance criteria:

- Each request produces correlated logs with stable keys.
- Log lines are machine-parsable and safe for ingestion.

---

## Phase 4 - Request Tracing Middleware and Propagation

- [ ] Add middleware in `backend/main.py` to create/propagate `request_id` from `X-Request-ID`.
- [ ] Attach `request_id` to `request.state` and include in logs, agent state, and response headers.
- [ ] Extend `backend/agent/state.py` to carry correlation fields (`request_id`, optional `trace_id`).
- [ ] Ensure `backend/observability/langsmith_tracer.py` includes correlation metadata in run config.

Acceptance criteria:

- A single request can be traced across API logs, agent steps, tool calls, and evaluation records using shared IDs.

---

## Phase 5 - Implement Event Warehouse Schema

- [x] Add migration(s) under `infra/postgres/` for analytics tables aligned with `specs/observability.md`:
  - `agent_spans`
  - `outcomes`
  - `evaluation_scores`
  - (extend) `sessions` analytics fields where needed
- [x] Add ORM models in `backend/db/models.py` and data access helpers.
- [x] Write event writers for span-level and outcome-level records from:
  - `backend/agent/nodes.py`
  - `backend/api/routes/chat.py`
  - `evaluation/ragas_eval.py` (or async evaluator service)
- [x] Define retention and indexing strategy for time-series querying.

Acceptance criteria:

- Every completed conversation yields warehouse records linking session, spans, outcomes, and evaluation scores.

---

## Phase 6 - Business and Product Metric Computation Layer

- [x] Add metric computation jobs/scripts in `evaluation/` or a new `backend/analytics/` module for:
  - Automation rate
  - Escalation rate
  - First contact resolution proxy
  - Tool success rate
  - Turns to resolution
  - Recovery rate
- [x] Add SQL definitions/views for repeatable metric queries over warehouse tables.
- [x] Document formulas and data assumptions in `docs/qa.md`.
- [x] Add optional placeholders for CSAT/NPS integration once data source exists.

Acceptance criteria:

- Business/product KPIs are reproducibly queryable from stored telemetry, not only live Prometheus streams.

---

## Phase 7 - Asynchronous Evaluation Pipeline

- [x] Create async evaluator pipeline (new module under `evaluation/` or `backend/evaluation/`) that:
  - Samples completed sessions
  - Reconstructs evaluation inputs
  - Runs groundedness/hallucination/helpfulness checks
  - Writes scores to `evaluation_scores`
- [x] Keep `evaluation/ragas_eval.py` as a benchmark/regression runner, but separate it from continuous scoring.
- [x] Add scheduling mechanism (cron/worker) and failure retries.
- [x] Add tests for evaluator idempotency and duplicate-run prevention.

Acceptance criteria:

- Evaluation runs continuously without blocking chat requests.
- Session-level quality scores are backfilled and queryable.

---

## Phase 8 - OpenTelemetry Alignment (Spec-Level Tracing)

- [x] Add OpenTelemetry SDK instrumentation in backend entrypoints and agent execution path.
- [x] Define root conversation trace and child spans for:
  - Intent detection
  - Retrieval
  - Tool calls
  - Response synthesis
  - Outcome
- [x] Add OTel Collector config under `infra/` and export pipeline (OTLP).
- [x] Bridge/align LangSmith metadata with OTel IDs where both are enabled.
- [x] Document local/dev setup in `docs/how_to_run.md`.

Acceptance criteria:

- Conversation traces are available in standard OTel format with required span attributes from the observability spec.

---

## Phase 9 - Dashboard Expansion (Executive, Product, AI Quality)

- [ ] Keep current operations dashboard and add:
  - `infra/grafana/dashboards/executive-observability.json`
  - `infra/grafana/dashboards/product-observability.json`
  - `infra/grafana/dashboards/ai-quality-observability.json`
- [ ] Include panels for:
  - Executive: automation rate, escalation, cost proxy, CSAT/NPS placeholders
  - Product: task completion, turns to resolution, recovery rate, tool success
  - AI quality: groundedness, hallucination rate, retrieval precision/recall, confidence calibration proxy
- [ ] Validate each panel query against warehouse or Prometheus source of truth.

Acceptance criteria:

- Stakeholders can inspect observability by audience without custom query writing.

---

## Phase 10 - Alerting and SLOs

- [ ] Add alert rules (Prometheus/Alertmanager or equivalent) for:
  - Elevated error rate
  - P95 latency regression
  - Retrieval failure spike
  - Escalation spike
  - Evaluation quality regression
- [ ] Add runbooks in `docs/` for each alert category.
- [ ] Define SLOs and error budgets for chat availability, latency, and quality.
- [ ] Add smoke-check automation to verify alerts trigger and resolve.

Acceptance criteria:

- Critical degradations produce actionable alerts with linked runbooks and owners.

---

## Phase 11 - Feedback Loops and Self-Healing (Maturity Level 3)

- [ ] Define automated remediation rules:
  - Retrieval quality drop -> trigger re-ingestion/re-index
  - Tool failure spike -> circuit breaker or fallback flow
  - Hallucination increase -> stricter grounding policy/prompt guard
- [ ] Add model/data drift detection jobs and retraining/reconfiguration triggers.
- [ ] Add governance controls for automatic actions (thresholds, cooldowns, manual override).
- [ ] Track interventions and outcomes as first-class events.

Acceptance criteria:

- The system can detect quality degradation and trigger safe, auditable corrective actions.

---

## Suggested Implementation Order (Execution Roadmap)

1. Phase 1 (stabilize current eval)  
2. Phase 2 (metrics coverage)  
3. Phase 3-4 (structured logs + request tracing)  
4. Phase 5-6 (warehouse + KPI computation)  
5. Phase 7 (async evaluation pipeline)  
6. Phase 8 (OpenTelemetry alignment)  
7. Phase 9-10 (dashboards + alerting)  
8. Phase 11 (self-healing loops)

## Notes

- Keep backward compatibility for existing dashboards and `/metrics`.
- Prefer additive schema/migration changes with clear versioning.
- Any metric name changes must include migration notes in docs and dashboards.
