## P95 Latency Regression Runbook

## Alert

`P95LatencyRegression`

## Trigger Condition

P95 chat latency exceeds 10 seconds for 5 minutes.

## Likely Causes

- LLM provider slowness
- Retrieval latency increase (embedding/rerank/database)
- Thread starvation or connection pool pressure

## Diagnostics

- Compare `chat_latency_seconds_bucket` and `retrieval_latency_seconds_bucket`.
- Check `embedding_latency_seconds`, `rerank_latency_seconds`, DB/Redis operation histograms.
- Inspect recent model/provider or prompt changes.
- Verify resource usage on backend host/container.

## Immediate Remediation

- Disable reranking temporarily when safe.
- Reduce top_k retrieval and expensive prompt paths.
- Scale backend and tune DB/Redis pools.

## Escalation

- Owner: Engineering
- Escalate to model-platform owner when external provider latency is dominant.
