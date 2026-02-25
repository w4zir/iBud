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
