from __future__ import annotations

from prometheus_client import Counter, Histogram

request_count = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["status"],
)

request_latency = Histogram(
    "chat_latency_seconds",
    "Request latency in seconds",
)

retrieval_latency = Histogram(
    "retrieval_latency_seconds",
    "RAG retrieval latency in seconds",
)

cache_hits = Counter(
    "redis_cache_hits_total",
    "Redis cache hits",
)

tool_calls = Counter(
    "agent_tool_calls_total",
    "Tool calls",
    ["tool_name"],
)

escalations = Counter(
    "escalations_total",
    "Human escalations triggered",
)

llm_tokens = Counter(
    "llm_tokens_total",
    "LLM tokens used",
    ["provider"],
)

tool_outcome = Counter(
    "tool_outcome_total",
    "Tool outcomes by tool and status",
    ["tool_name", "outcome"],
)

intent_distribution = Counter(
    "intent_distribution_total",
    "Distribution of classified intents",
    ["intent"],
)

chat_turns = Histogram(
    "chat_turns_per_session",
    "Number of turns in a session when a chat request is handled",
)

task_outcome = Counter(
    "task_outcome_total",
    "Task outcomes across agent runs",
    ["outcome"],
)

error_count = Counter(
    "errors_total",
    "Errors by type and component",
    ["error_type", "component"],
)

embedding_latency = Histogram(
    "embedding_latency_seconds",
    "Embedding latency in seconds",
)

rerank_latency = Histogram(
    "rerank_latency_seconds",
    "Rerank latency in seconds",
)

db_latency = Histogram(
    "db_operation_latency_seconds",
    "Database operation latency in seconds",
    ["operation"],
)

redis_latency = Histogram(
    "redis_operation_latency_seconds",
    "Redis operation latency in seconds",
    ["operation"],
)

__all__ = [
    "request_count",
    "request_latency",
    "retrieval_latency",
    "cache_hits",
    "tool_calls",
    "escalations",
    "llm_tokens",
    "tool_outcome",
    "intent_distribution",
    "chat_turns",
    "task_outcome",
    "error_count",
    "embedding_latency",
    "rerank_latency",
    "db_latency",
    "redis_latency",
]

