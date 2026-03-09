## Evaluation Quality Regression Runbook

## Alert

`EvaluationQualityRegression`

## Trigger Condition

Quality regression proxy indicates sustained degradation.

## Likely Causes

- Retrieval grounding quality drop
- Prompt regressions increasing hallucination risk
- Model/provider output drift

## Diagnostics

- Query `evaluation_scores` for groundedness/hallucination/helpfulness trends.
- Review affected intents and tool paths from `agent_spans`.
- Compare recent changes in prompts, model version, and retriever config.

## Immediate Remediation

- Enable stricter grounding mode and reduce unsupported generation paths.
- Trigger re-ingestion/re-index if retrieval quality dropped.
- Increase manual review for impacted intents until stable.

## Escalation

- Owner: QA + Engineering
- Escalate to AI owner for model-level regressions.
