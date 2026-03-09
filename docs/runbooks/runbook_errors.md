## Elevated Error Rate Runbook

## Alert

`HighErrorRate`

## Trigger Condition

Backend error rate is above 5% for 5 minutes.

## Likely Causes

- Backend dependency outage (Postgres, Redis, model provider)
- Regressed deployment in API or agent flow
- Increased invalid payload traffic

## Diagnostics

- Check `chat_requests_total` and `errors_total` series in Prometheus.
- Inspect backend logs filtered by `error_type`, `request_id`, and `component`.
- Check `/health` and dependency states.
- Review recent deploy and config changes.

## Immediate Remediation

- Roll back most recent deployment if regression is confirmed.
- Scale backend replicas if saturation is observed.
- Degrade gracefully by disabling expensive optional features if needed.

## Escalation

- Owner: Engineering
- Notify on-call backend engineer and SRE for sustained critical errors (>15 min).
