# Agentic Support Observability Specification

## Purpose

This document defines the observability, metrics, architecture, schema,
and evaluation pipeline for an **agentic customer support system** using
Retrieval-Augmented Generation (RAG). It is intended as a
production-ready specification for implementation alongside an agentic
RAG system.

------------------------------------------------------------------------

# 1. Observability Philosophy

Agentic systems must be observed as **decision engines**, not chatbots.

Traditional logging: User → Message → Response

Agentic logging: User Intent → Reasoning → Tool Actions → Outcome →
Business Impact

Each conversation is modeled as a distributed trace.

------------------------------------------------------------------------

# 2. Metrics Framework

## 2.1 Business Metrics

Measure real business value.

-   Automation Rate = AI resolved / total conversations
-   Cost per Resolution
-   Escalation Rate
-   First Contact Resolution (FCR)
-   CSAT / NPS
-   Revenue Impact (upsell, retention)

### Example

10,000 tickets/month 6,500 solved by AI → Automation = 65%

------------------------------------------------------------------------

## 2.2 Product Metrics

Measure agent experience quality.

-   Task Completion Rate
-   Tool Success Rate
-   Turns to Resolution
-   Containment Quality
-   User Effort Score (CES)
-   Recovery Rate

------------------------------------------------------------------------

## 2.3 Data & AI Metrics

Measure intelligence reliability.

-   Groundedness / Answer Accuracy
-   Hallucination Rate
-   Retrieval Precision@K
-   Context Recall
-   Intent Classification Accuracy
-   Tool Selection Accuracy
-   Latency
-   Confidence Calibration

------------------------------------------------------------------------

# 3. Architecture Overview

User Interaction ↓ Agent Runtime ↓ OpenTelemetry Traces ↓ OTel Collector
↓ Event Warehouse ↓ Evaluation Pipelines ↓ Metrics Tables ↓ Dashboards

------------------------------------------------------------------------

# 4. OpenTelemetry Model

## Core Concepts

-   Trace = entire conversation session
-   Span = single agent action

Example trace: - intent_detection - retrieval - reasoning - tool_call -
response_generation - task_outcome

------------------------------------------------------------------------

# 5. Instrumentation Specification

## Root Trace

Attributes: - session_id - channel - user_id

## Intent Span

Attributes: - intent - confidence

## Retrieval Span

Attributes: - docs_returned - doc_ids

## Tool Call Span

Attributes: - tool_name - success - latency_ms - retries

## Outcome Span

Attributes: - task - completed - escalated

------------------------------------------------------------------------

# 6. Event Schema (Warehouse)

## Table: sessions

  field        description
  ------------ -----------------
  session_id   conversation id
  intent       detected intent
  start_time   timestamp
  end_time     timestamp
  escalated    boolean
  csat         rating

## Table: agent_spans

  field        description
  ------------ ------------------
  trace_id     trace identifier
  span_name    action name
  attributes   json payload
  latency_ms   execution time
  timestamp    event time

## Table: outcomes

  field        description
  ------------ -------------
  session_id   session
  task         task name
  completed    boolean
  verified     boolean

## Table: evaluation_scores

  field           description
  --------------- -------------
  session_id      session
  groundedness    float
  hallucination   boolean
  helpfulness     float

------------------------------------------------------------------------

# 7. Evaluation Pipeline

Telemetry shows WHAT happened. Evaluation shows HOW GOOD it was.

## Flow

Completed sessions ↓ Sampling job ↓ LLM evaluator ↓ Scores written to
warehouse

## Evaluations

-   groundedness
-   hallucination detection
-   policy compliance
-   helpfulness scoring

Evaluations run asynchronously.

------------------------------------------------------------------------

# 8. Metrics Computation Examples

Automation Rate: SELECT COUNT(*) FILTER (WHERE completed=true AND
escalated=false) / COUNT(*) FROM outcomes;

Tool Success Rate: SELECT AVG(attributes-\>\>'success') FROM agent_spans
WHERE span_name='tool_call';

Hallucination Rate: SELECT AVG(hallucination::int) FROM
evaluation_scores;

------------------------------------------------------------------------

# 9. Dashboards

## Executive Dashboard

-   Automation rate
-   CSAT
-   Cost savings
-   Escalation rate

## Product Dashboard

-   Task completion
-   Turns per resolution
-   Tool latency
-   Recovery rate

## AI Quality Dashboard

-   Groundedness
-   Hallucination rate
-   Retrieval precision
-   Tool accuracy

------------------------------------------------------------------------

# 10. Recommended Stack

  Layer        Tool
  ------------ ----------------------
  Tracing      OpenTelemetry
  Collector    OTel Collector
  Streaming    Kafka
  Storage      BigQuery / Snowflake
  Evaluation   Python + LLM
  Dashboards   Grafana

------------------------------------------------------------------------

# 11. Design Principles

1.  Log outcomes, not messages.
2.  Evaluate asynchronously.
3.  Separate observation from evaluation.
4.  Treat agents like distributed systems.
5.  Link AI quality → product behavior → business impact.

------------------------------------------------------------------------

# 12. Target Maturity Model

Level 1: Chatbot Analytics - message logs only

Level 2: Agent Analytics - tool tracking - outcome logging

Level 3: Self-Healing Agent - automatic evaluation - feedback loops -
retraining triggers

------------------------------------------------------------------------

# End of Specification
