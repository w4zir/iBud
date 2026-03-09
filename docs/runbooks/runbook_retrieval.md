## Retrieval Failure Spike Runbook

## Alert

`RetrievalFailureSpike`

## Trigger Condition

Retriever error rate increases above baseline for 5 minutes.

## Likely Causes

- Embedding service or model unavailable
- pgvector query failures
- Redis connectivity failures masking retrieval behavior

## Diagnostics

- Inspect `errors_total{component="retriever"}` by `error_type`.
- Check embedding provider availability and credentials.
- Validate Postgres and pgvector health, query latency, and capacity.
- Confirm KB data is present and ingest status is healthy.

## Immediate Remediation

- Fail open with retrieval fallback if business-safe.
- Re-run ingestion/reindex job when data quality issues are confirmed.
- Temporarily disable rerank to reduce failure surface.

## Escalation

- Owner: Engineering
- Involve data platform owner if index/data corruption is suspected.
