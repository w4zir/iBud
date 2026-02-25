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

__all__ = [
    "request_count",
    "request_latency",
    "retrieval_latency",
    "cache_hits",
    "tool_calls",
    "escalations",
    "llm_tokens",
]

