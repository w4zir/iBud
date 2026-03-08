## SLOs and Error Budgets

This document defines Phase 10 service-level objectives for chat availability, latency, and quality.

## SLO 1 - Chat Availability

- **Objective:** 99.5% successful request rate over 30-day rolling window.
- **SLI:** `1 - (errors_total / chat_requests_total)` using request-weighted ratio.
- **Target:** >= 0.995
- **Error Budget:** 0.5% of total requests in 30 days.

## SLO 2 - Chat Latency

- **Objective:** P95 chat latency under 10 seconds.
- **SLI:** `histogram_quantile(0.95, sum(rate(chat_latency_seconds_bucket[5m])) by (le))`
- **Target:** < 10 seconds
- **Error Budget:** Time above threshold should stay below 5% of rolling windows.

## SLO 3 - Quality

- **Objective:** Hallucination rate under 10%, groundedness above 0.6.
- **SLI (warehouse):**
  - `AVG(CASE WHEN hallucination THEN 1 ELSE 0 END) < 0.10`
  - `AVG(groundedness) > 0.60`
- **Target Window:** Daily + weekly trend tracking.

## Burn Rate Guidance

- **Fast burn alert:** 2h window consuming >10% of monthly error budget.
- **Slow burn alert:** 24h window consuming >25% of monthly error budget.

## Ownership

- Availability/latency: Engineering on-call
- Quality: QA + AI owner
- Escalation behavior: Product + Engineering
