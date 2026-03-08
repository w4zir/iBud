## Escalation Spike Runbook

## Alert

`EscalationSpike`

## Trigger Condition

Escalation rate above 30% for 10 minutes.

## Likely Causes

- Drop in tool success rate
- Intent classification drift
- Policy/prompt change increasing conservative escalation

## Diagnostics

- Review `task_outcome_total`, `tool_outcome_total`, and `intent_distribution_total`.
- Inspect recent sessions with escalation and ticket summaries.
- Compare baseline vs current quality metrics in warehouse.

## Immediate Remediation

- Enable remediation checks for tool failure spike and hallucination increase.
- Roll back recent prompt or routing changes if regressions are clear.
- Activate targeted fallback for unstable tools.

## Escalation

- Owner: Product + Engineering
- Notify support operations if customer impact is visible.
