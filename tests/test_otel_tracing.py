from __future__ import annotations

from backend.agent.state import AgentState
from backend.observability.langsmith_tracer import build_run_config
from backend.observability.otel import is_otel_enabled


def _make_state() -> AgentState:
    return {
        "session_id": "sess-1",
        "user_id": "user-1",
        "request_id": "req-1",
        "messages": [],
        "intent": "product_qa",
    }


def test_is_otel_enabled_from_env(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "true")
    assert is_otel_enabled() is True
    monkeypatch.setenv("OTEL_ENABLED", "false")
    assert is_otel_enabled() is False


def test_langsmith_config_includes_otel_ids(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(
        "backend.observability.langsmith_tracer.get_current_trace_ids",
        lambda: ("trace-otel-123", "span-otel-456"),
    )
    cfg = build_run_config(_make_state(), thread_id="t-1")
    meta = cfg.get("metadata", {})
    assert meta.get("otel_trace_id") == "trace-otel-123"
    assert meta.get("otel_span_id") == "span-otel-456"

