# FAQ

## Debugging and observability

### Q: How do I track the different steps a user message goes through to debug?

You can track each step of a user message in several ways:

1. **LangSmith tracing (recommended)**  
   When LangSmith is enabled, every chat run is sent as a trace so you can see each graph node and LLM call.

   - Set in your environment (e.g. `.env` or shell):
     - `LANGCHAIN_TRACING_V2=true`
     - `LANGCHAIN_API_KEY=<your LangSmith API key>`
     - `LANGCHAIN_PROJECT=<project name, e.g. ecom-support-rag>`
   - Restart the backend so it picks up these variables.
   - Send a message via the UI or API; then open the [LangSmith](https://smith.langchain.com) project. Each run appears as a trace with:
     - **Nodes**: `classify_intent` → `retrieve_context` → `plan_action` → (optionally `execute_tool`) → `synthesize_response` → `check_escalation` → (optionally `create_ticket`).
     - **Metadata**: `session_id`, `user_id`, and (after intent classification) `intent`.
   - You can inspect inputs/outputs per node, LLM requests/responses, tool calls, and token usage.

2. **Prometheus metrics**  
   Hit `http://localhost:8000/metrics` (with the backend running) to see counters and histograms for chat requests, latency, retrieval latency, cache hits, tool calls, escalations, and token usage. Useful for spotting slow steps or high error rates.

3. **Backend logs**  
   Run the backend in a terminal and watch stdout/stderr for errors and any log lines emitted by the API, agent, or RAG pipeline (e.g. Redis or DB warnings).

4. **Session and graph reference**  
   For the exact sequence of steps and data flow, see **Data flow** and **Agent graph** in `docs/how_it_works.md`.

### Q: plan_action says to call tool order_lookup — how do I check what call was made (e.g. SQL query) and what response was received?

- **What call was made (arguments)**  
  The agent invokes `order_lookup` from the `execute_tool` node with arguments from `plan_action` (`order_number`, `user_id`). To see the exact call:
  - **LangSmith**: In the trace, open the **execute_tool** node. Its input state includes `tool_calls` (name and `arguments`). The **plan_action** node's output also shows the same `tool_calls` (the JSON the LLM produced). That gives you the exact `order_number` and/or `user_id` passed to the tool.
  - **Backend**: There is no built-in log of tool arguments; add temporary logging in `backend/agent/nodes.py` inside `execute_tool` (e.g. log `name` and `args` for each call) if you need it in stdout.

- **What response was received**  
  The tool returns `{"orders": [...]}`. To see it:
  - **LangSmith**: In the same trace, the **execute_tool** node's *output* state contains `tool_results`. The entry for `order_lookup` has `result` with the `orders` list (and `success` / `error`).
  - **Backend**: Again, add logging in `execute_tool` after the tool call (e.g. log `result`) if you want it in your server logs.

- **The actual SQL query**  
  `order_lookup` uses SQLAlchemy to run a `SELECT` on the `orders` table (filtered by `order_number` and/or `user_id`). To see the emitted SQL:
  - **SQLAlchemy echo**: In `backend/db/postgres.py`, the engine is created with `echo=False`. Set `echo=True` (or drive it from an env var like `SQL_ECHO=1`) so that SQLAlchemy logs each statement to stdout. Then run the backend and trigger an order lookup; the SQL will appear in the terminal.
 - **PostgreSQL**: Alternatively, enable query logging on the Postgres server (e.g. `log_statement = 'all'`) and inspect the DB logs.

## Evaluation and LLM behavior

### Q: Where is the message `The output is incomplete due to a max_tokens length limit.` coming from, and where is the `max_tokens` limit defined?

This message is emitted by the Ragas / LLM stack (via the `instructor`-style JSON-enforcing wrapper around the chat model), not by our application code. When the underlying chat completion hits its configured `max_tokens` response limit and returns with `finish_reason="length"`, that wrapper raises an `InstructorRetryException` whose message includes `The output is incomplete due to a max_tokens length limit.`. The actual `max_tokens` value is defined inside the LLM client / provider configuration used by Ragas (or its dependencies) and is **not** set anywhere in this repository; to change it you would need to adjust the LLM configuration in the Ragas/LLM layer (for example, via that library’s settings or environment variables), not in our `evaluation/` code.

### Q: How do RAGAS evaluations work on the WixQA dataset, and what model is used for eval?

The `evaluation/ragas_eval.py` script runs RAGAS over the WixQA test set by sending each test question to the backend `/chat` endpoint, taking the **final response string** it returns (whatever the agent answered) and pairing it with the **ground-truth answer** from the WixQA JSON as the reference, plus the retrieved contexts (from `sources` or, as a fallback, the ground-truth article). These are assembled into a `Dataset` with `user_input`, `response`, `retrieved_contexts`, and `reference`, which is then scored by `ragas.evaluate` — RAGAS never inspects the agent internals, only the observable inputs/outputs. For the evaluation LLM, you can configure a **separate judge provider/model** from the runtime agent:

- By default, if no judge-specific env vars are set, RAGAS will:
  - Use a Cerebras-backed judge when `LLM_PROVIDER=cerebras` (via the OpenAI-compatible endpoint and `CEREBRAS_MODEL`, defaulting to `llama3.1-8b`), or
  - Fall back to RAGAS' own OpenAI-backed default evaluator.
- To override this and run evals on a different provider/model than the agent, set:
  - `RAGAS_LLM_PROVIDER=openai` or `cerebras`
  - Optionally, `RAGAS_OPENAI_MODEL` or `RAGAS_CEREBRAS_MODEL` and `RAGAS_CEREBRAS_API_KEY` / `RAGAS_CEREBRAS_BASE_URL`

These judge-specific env vars only affect RAGAS evaluation and do **not** change which model the agent uses at runtime.
