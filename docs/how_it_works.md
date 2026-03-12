## Overview

This document explains **how the iBud agentic RAG system works** end to end:

- High-level **system architecture**.
- **Data flow** for a chat request.
- The main **components** and how they interact.
- Core **algorithms with pseudocode** for chunking, retrieval, and the agent runtime.

The system implements an e‑commerce customer support assistant backed by a **stateful workflow-style agent runtime**, a pgvector RAG store, Redis caching, and a Streamlit UI.

## System architecture

At a high level the system consists of:

- **Frontend (`frontend/`)**
  - Streamlit web app (`frontend/app.py`) that provides the chat UI.
  - Sends chat messages to the backend `/chat` API and renders responses, sources, and tool usage.

- **Backend API (`backend/`)**
  - FastAPI application (`backend/main.py`) exposing:
    - `POST /chat` — main chat endpoint.
    - `POST /chat/intent` — intent-only endpoint that persists the message and runs just the classifier node.
    - `GET /sessions/{session_id}/history` — session history.
    - `GET /health` — health and dependency checks.
    - `admin` routes for maintenance and debugging.
  - Wires up CORS, request-id middleware (`X-Request-ID` propagation), and global exception handlers.
  - Adds request correlation metadata to logs and response headers.

- **Agent runtime (`backend/agent/`)**
  - Orchestrator (`orchestrator.py`) implementing the runtime flow:
    - `intent_classifier` → `planner` → `evaluator` → `workflow_engine` → `executor` → `validator` → `response_generator`.
  - Agent state schema in `state.py` tracks conversation, workflow JSON state, plan, execution results, escalation, and response.
  - Planner cycle counter is stored in `AgentState["workflow_state"]["planner_cycle_count"]`.
  - Escalation performs dual-write: external human handoff (optional) + local ticket persistence.

- **RAG pipeline (`backend/rag/`)**
  - **Chunking** (`chunker.py`): section-aware splitting of WixQA help-center articles into parent documents and child chunks.
  - **Embeddings** (`embeddings.py`): wraps the embedding model configured via `get_embedding_model`.
  - **Retriever** (`retriever.py`): pgvector similarity search plus optional Redis caching and cross‑encoder reranking.
  - **Ingestion** (`ingest_wixqa.py`): loads `Wix/WixQA`, prepares articles, chunks them, embeds, and writes to `documents` table with parent/child relationships.

- **Tools (`backend/tools/`)**
  - `order_lookup.py` — queries `orders` table for a user’s order.
  - `return_initiate.py` — validates a return request and updates order/ticket state.
  - `faq_search.py` — higher-level search over the KB for FAQ-style answers.
  - `ticket_create.py` — creates `tickets` rows for human escalation.

- **Data & persistence (`backend/db/`)**
  - SQLAlchemy models in `models.py`:
    - `Document` — pgvector‑backed knowledge base entries (KB chunks and parents).
    - `Session` / `Message` — chat history storage.
    - `Order` — mock commerce orders for order-related flows.
    - `Ticket` — escalation tickets.
  - Async PostgreSQL engine and session factory in `postgres.py`, including Docker‑friendly host selection.
  - Redis client in `redis_client.py` for retrieval caching.

- **Observability (`backend/observability/`)**
  - `prometheus_metrics.py` — counters and histograms for chat requests, latency, retrieval latency, cache hits, tool calls, tool outcomes, intent distribution, task outcomes, typed errors, embedding/rerank latency, DB/Redis latency, escalations, and token usage.
  - `langsmith_tracer.py` — attaches LangSmith tracing metadata (including `request_id` / `trace_id` when available) to agent runs when enabled via environment variables.

- **Evaluation (`evaluation/`)**
  - Builds an evaluation testset based on WixQA.
  - Runs RAGAS (`ragas_eval.py`) against the live backend, writing JSON metrics (faithfulness, relevancy, context precision/recall).

Data stores and external services:

- **Postgres + pgvector** — persistent storage for documents, sessions, messages, orders, and tickets; pgvector column on `Document.embedding`.
- **Redis** — in‑memory cache for retrieval results keyed by query + filters.
- **Ollama / LLM provider** — chat and embedding models.
- **Prometheus & Grafana** — metrics collection and dashboards.
- **LangSmith** — tracing across LLM and tool calls.

## End-to-end data flow

### 1. Knowledge ingestion (offline)

1. The ingestion script (`backend.rag.ingest_wixqa`) loads articles from the `Wix/WixQA` dataset.
2. Each article is normalised into text + metadata and passed into the chunker.
3. The chunker:
   - Combines title/body.
   - Optionally splits HTML by headers.
   - Applies recursive character chunking (≈400 tokens, 50 token overlap).
4. For each article:
   - A **parent** `Document` row is created with the full article content.
   - Multiple **child** `Document` rows are created for chunks, with `parent_id` pointing to the parent.
5. Child chunks are embedded and stored in the `embedding` pgvector column.

This process is run manually or via scripts described in `docs/how_to_run.md`.

### 2. Chat request lifecycle (online)

When a user sends a message through the Streamlit UI:

1. **Frontend**
   - The Streamlit app sends `POST /chat` with:
     - `session_id` (optional for new session).
     - `user_id`.
     - `message` text.
2. **FastAPI route**
   - Validates the request body.
  - Generates or propagates `request_id` from `X-Request-ID`.
   - Creates or loads a `Session` row.
   - Appends a `Message` row with `role="user"`.
   - Builds an initial `AgentState` containing:
     - Messages, `session_id`, `user_id`, and any contextual metadata.
     - Correlation fields such as `request_id`.
3. **Agent runtime (`run_orchestrated_agent`)**
  - The orchestrator is invoked asynchronously with the current state.
  - The state flows through stages:
    - `intent_classifier` (small model; emits intent + sentiment_score + user_requested_human)
    - `planner` (large model; emits strict JSON plan schema and dependency graph)
    - `evaluator` (small model; validates plan, provides feedback to replan)
    - `workflow_engine` + `executor` (deterministic tools/APIs only; retries + failure recovery)
    - `validator` (small model; receives only Plan + Result)
    - `response_generator` (small model; final human-readable response)
4. **Response construction**
   - The final state includes:
     - `final_response` — answer text.
     - `retrieved_docs` — top KB snippets.
     - `tool_calls` and `tool_results`.
     - `should_escalate` and `ticket_id` (if escalation occurred).
   - The API serialises this into the HTTP response and also persists the assistant message in the `messages` table.
5. **Frontend rendering**
   - The UI displays the answer, optional sources/tools, and escalation banners when tickets are created.

### 3. Retrieval data flow

During `retrieve_context`:

1. The node derives the latest user text from conversation history.
2. It maps the classified intent (`order_status`, `return_request`, `product_qa`, etc.) into an optional KB `category`.
3. It calls `Retriever.search` with query, `category`, tier filter, and flags for caching and reranking.
4. The retriever:
   - Optionally returns **cached** results from Redis (if identical query parameters were seen before).
   - Otherwise:
     - Embeds the query.
     - Executes a pgvector similarity search over document embeddings.
     - Optionally expands highly relevant parent documents.
     - Optionally reranks candidates with a cross‑encoder.
   - Stores fresh results back into Redis.

The resulting `RetrievedDoc` structures are serialised into the agent state and later consumed by the responder node.

## Core algorithms (pseudocode)

### Chunking: preparing articles for storage

Based on `backend/rag/chunker.py`.

```pseudo
function prepare_article_for_chunking(title, body, url, category, source_id):
    text = join_non_empty_with_blank_line([title, body])
    metadata = {}
    if category: metadata["category"] = category
    if url:      metadata["url"] = url
    if source_id: metadata["source_id"] = source_id
    return (text, metadata)

function chunk_article(full_text, metadata, chunk_size=1600, chunk_overlap=200):
    if full_text is empty:
        return []

    sections = split_by_headers_if_html(full_text)
    all_chunks = []

    for (section_text, section_meta) in sections:
        section_metadata = copy(metadata)
        section_metadata.update(section_meta)   // add h1/h2/h3 info

        for chunk in recursive_split(section_text, chunk_size, chunk_overlap):
            if chunk not empty:
                all_chunks.append((chunk.trim(), section_metadata))

    return all_chunks
```

### Retrieval: pgvector + Redis + optional reranking

Based on `backend/rag/retriever.py`.

```pseudo
class Retriever:
    constructor(embedding_client):
        self.embedding_client = embedding_client or default_embedding_client()
        self.cross_encoder = null     // lazy loaded

    async function search(query, top_k=5, category=None, tier_filter=None,
                          use_cache=True, rerank=True) -> List[RetrievedDoc]:
        if query is empty:
            return []

        cache_key = null
        if use_cache:
            cache_key = build_cache_key(query, category, tier_filter, top_k, rerank)
            cached = await get_cached(cache_key)
            if cached exists:
                increment_prometheus_cache_hits()
                return deserialize_retrieved_docs(cached)

        start_time = now()

        // 1. Embed query
        query_vector = embedding_client.embed_query(query)

        // 2. Similarity search against pgvector
        async with async_session_factory() as session:
            docs = await similarity_search(session, query_vector,
                                           top_k, category, tier_filter)

        // 3. Optionally expand parent documents
        docs = await maybe_expand_parents(docs, score_threshold=0.4)

        // 4. Optionally rerank with cross-encoder
        if rerank:
            docs = await rerank_with_cross_encoder(query, docs)

        record_retrieval_latency(now() - start_time)

        // 5. Cache successful results
        if use_cache and cache_key and docs not empty:
            await set_cached(cache_key, serialize(docs))

        return docs

    async function similarity_search(session, query_vector, top_k, category, tier_filter):
        distance = cosine_distance(Document.embedding, query_vector)
        stmt = select(Document, distance as score)
                 .where(Document.embedding is not null)
        if category:
            stmt = stmt.where(Document.category == category)
        if tier_filter is not None:
            stmt = stmt.where(Document.doc_tier == tier_filter)
        stmt = stmt.order_by(distance).limit(top_k)

        rows = await session.execute(stmt)
        return [
            RetrievedDoc(
                content=row.Document.content,
                metadata=row.Document.metadata,
                score=float(row.score) or 0.0,
                source=row.Document.source,
                doc_tier=row.Document.doc_tier,
                document_id=row.Document.id,
                parent_id=row.Document.parent_id
            )
            for row in rows
        ]
```

Parent expansion and reranking are implemented as separate helpers:

- **Parent expansion**: when a high‑scoring child chunk has a parent, the parent article content is fetched and added as an additional `RetrievedDoc` directly following the child.
- **Reranking**: if the `sentence_transformers` cross‑encoder can be loaded, it scores `(query, doc.content)` pairs and sorts documents by descending cross‑encoder score.

### Agent runtime: orchestrated workflow

Based on `backend/agent/orchestrator.py`.

```pseudo
intent_classifier(user_message) -> intent, sentiment_score, user_requested_human
if sentiment_score < 0.3 or user_requested_human or angry: escalate_to_human()

planner(cycle_count, user_request, intent, feedback?) -> plan_json
evaluator(user_request, plan_json) -> plan_valid, feedback
if not plan_valid: loop planner (cycle_count++)

workflow_engine(plan_json) -> result_json (tasks executed with retries)
validator(plan_json, result_json) -> achieved, feedback, sentiment_score
if sentiment_score < 0.3: escalate_to_human()
if not achieved: loop planner (cycle_count++)

response_generator(plan_json, result_json) -> final_response
```

#### Node behaviours (pseudocode)

```pseudo
async function classify_intent(state):
    user_text = latest_user_message(state.messages)
    llm = get_llm()

    messages = [
      System(SYSTEM_INTENT_CLASSIFIER),
      Human(user_text)
    ]
    resp = await llm.ainvoke(messages)
    record_llm_tokens(resp)

    raw = normalise(resp.content)
    intent = map_to_supported_intent(raw, fallback_rules)

    return state.with(intent = intent)


async function retrieve_context(state):
    user_text = latest_user_message(state.messages)
    intent = state.intent
    category = intent_to_category(intent)   // e.g. "orders", "product", "account"

    retriever = Retriever()
    docs = await retriever.search(
        query = user_text,
        top_k = 5,
        category = category,
        tier_filter = 1,
        use_cache = True,
        rerank = True,
    )

    serialized_docs = [serialize_retrieved_doc(d) for d in docs]
    return state.with(retrieved_docs = serialized_docs)


async function plan_action(state):
    llm = get_llm()
    user_text = latest_user_message(state.messages)
    intent = state.intent
    snippets = top_k_snippets(state.retrieved_docs)

    planner_prompt = build_planner_prompt(intent, user_text, snippets)
    messages = [
      System(SYSTEM_PLANNER),
      Human(planner_prompt)
    ]
    resp = await llm.ainvoke(messages)
    record_llm_tokens(resp)

    tool_calls = safe_json_parse_tool_calls(resp.content)
    return state.with(tool_calls = tool_calls)


async function execute_tool(state):
    results = []

    for call in state.tool_calls:
        name = call.name
        args = call.arguments or {}
        increment_tool_calls_metric(name)

        try:
            if name == "order_lookup":
                result = await order_lookup_tool(args.order_number, args.user_id)
            elif name == "return_initiate":
                result = await return_initiate_tool(args.order_number, args.user_id)
            elif name == "faq_search":
                result = await faq_search_tool(
                    query = args.query or latest_user_message(state.messages),
                    category = args.category,
                    top_k = args.top_k or 5,
                )
            elif name == "ticket_create":
                result = await ticket_create_tool(
                    issue_type = args.issue_type or "escalation",
                    summary = args.summary or latest_user_message(state.messages),
                    session_id = state.session_id,
                )
            else:
                throw UnknownTool(name)

            results.append(success_tool_result(name, result))
        except Exception as e:
            results.append(failure_tool_result(name, error = str(e)))

    // Attach ticket_id if created
    next_state = state.with(tool_results = results)
    for r in results:
        if r.name == "ticket_create" and r.success:
            next_state.ticket_id = r.result.ticket_id

    return next_state


async function synthesize_response(state):
    llm = get_llm()
    user_text = latest_user_message(state.messages)
    kb_context = format_kb_context(state.retrieved_docs)
    tools_context = json_dump(state.tool_results)

    messages = [
      System(SYSTEM_RESPONDER),
      Human(build_responder_prompt(user_text, state.intent, kb_context, tools_context))
    ]
    resp = await llm.ainvoke(messages)
    record_llm_tokens(resp)

    answer = resp.content.trim()
    return state.with(final_response = answer)


async function check_escalation(state):
    llm = get_llm()
    user_text = latest_user_message(state.messages)
    tool_results = state.tool_results or []

    had_failures = any(result.success == False for result in tool_results)
    is_complaint = (state.intent == "complaint")

    if had_failures and is_complaint:
        should_escalate = True
    else:
        messages = [
          System(SYSTEM_ESCALATION),
          Human(build_escalation_prompt(user_text, state.intent, tool_results))
        ]
        resp = await llm.ainvoke(messages)
        record_llm_tokens(resp)
        should_escalate = "true" in resp.content.lower()

    return state.with(should_escalate = should_escalate)


async function create_ticket(state):
    if not state.should_escalate:
        return state
    if state.ticket_id already set:
        return state

    user_text = latest_user_message(state.messages)
    result = await ticket_create_tool(
        issue_type = "escalation",
        summary = user_text,
        session_id = state.session_id,
    )

    next_state = state.copy()
    next_state.ticket_id = result.ticket_id
    increment_escalations_metric()
    next_state.tool_results.append(success_tool_result("ticket_create", result))
    return next_state
```

## How components fit together

Putting it all together:

- **Frontend** collects user input and displays answers; it is stateless aside from session identifiers.
- **FastAPI backend** handles HTTP, validation, error handling, and persistence to Postgres (sessions, messages, tickets).
- **Agent runtime (orchestrator)** orchestrates model calls, tools, validation loops, and escalation decisions as a stateful workflow.
- **RAG pipeline** ensures answers are grounded in the WixQA corpus using chunking, embeddings, pgvector similarity, caching, and reranking.
- **Tools** connect the agent to **business data** (orders, returns) and **support workflows** (tickets).
- **Observability and evaluation** provide continuous feedback loops:
  - Prometheus + Grafana monitor latency, error rates, cache hits, and token usage.
  - LangSmith traces reveal step-by-step agent behaviour.
  - RAGAS scores measure answer quality and help prevent regressions.

## Event warehouse (Phase 5)

The runtime now writes structured events for historical analysis:

- `agent_spans` records stage-level spans (`intent_classifier`, `planner`, `evaluator`, `workflow_engine`, `validator`, `response_generator`, `human_escalation`) with JSON attributes and latency.
- `outcomes` records final task completion/escalation outcomes per conversation.
- `evaluation_scores` stores asynchronous quality scores (`groundedness`, `hallucination`, `helpfulness`).
- `sessions` includes analytics fields (`intent`, `escalated`, `resolved_at`, `csat_score`, `nps_score`).

These writes are done as non-blocking async side effects, so chat responses are not delayed by analytics persistence.

## Business/product KPIs (Phase 6)

The computation layer in `backend/analytics/metrics.py` calculates:

- Automation rate
- Escalation rate
- First-contact-resolution proxy
- Tool success rate
- Turns to resolution
- Recovery rate

SQL views under `infra/postgres/migrations/004_analytics_views.sql` provide repeatable query entry points (for example `v_automation_rate`, `v_escalation_rate`, `v_tool_success_rate`).

## Asynchronous evaluation pipeline (Phase 7)

Continuous scoring is separated from benchmark runs:

- Benchmark/regression: `evaluation/ragas_eval.py`
- Continuous async scoring: `backend/evaluation/pipeline.py`

Flow:

1. Sample completed sessions that do not yet have `evaluation_scores`.
2. Reconstruct conversation inputs from session messages.
3. Score groundedness/hallucination/helpfulness.
4. Write to `evaluation_scores` with idempotency guard (skip duplicates).

The pipeline is triggered via `POST /admin/eval/trigger` and is suitable for cron/worker scheduling.

## OpenTelemetry alignment (Phase 8)

OpenTelemetry instrumentation is available when `OTEL_ENABLED=true`:

- Root `conversation` span in chat request handling.
- Child spans in agent execution for intent, retrieval, tool calls, synthesis, and outcome.
- FastAPI auto-instrumentation via `backend/observability/otel.py`.
- Collector config at `infra/otel/otel-collector-config.yaml`.

LangSmith and OTel are linked by injecting `otel_trace_id`/`otel_span_id` into LangSmith run metadata, enabling cross-system trace correlation.

## Dashboard expansion (Phase 9)

Grafana dashboards are split by stakeholder audience:

- **Executive**: automation rate, escalation rate, cost proxy, request and error trend.
- **Product**: task completion, turns to resolution, recovery proxy, tool success/latency, intent mix.
- **AI quality**: quality proxies, retrieval/tool reliability, evaluation throughput hints, and warehouse quality guidance.

This split reduces custom query work and gives each audience a focused operational surface.

## Alerting and SLOs (Phase 10)

Prometheus alert rules encode key degradations:

- Elevated error rate
- P95 latency regression
- Retrieval failure spike
- Escalation spike
- Quality regression proxy

Alert annotations link directly to runbooks, and SLO/error-budget definitions in `docs/slos.md` define objective reliability targets for availability, latency, and quality.

## Feedback loops and self-healing (Phase 11)

The remediation engine adds controlled automation on top of observability:

1. Evaluate rules for retrieval quality drop, tool failure spikes, and hallucination increase.
2. Apply governance controls (global enable, manual override, cooldown, action budget).
3. Execute safe remediation actions when eligible.
4. Record interventions as first-class `agent_spans` events (`span_name="remediation"`).

Additional drift detection compares recent quality and intent patterns against baseline windows to support auditable corrective actions and future retraining/reconfiguration triggers.

