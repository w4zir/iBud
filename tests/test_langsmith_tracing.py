from __future__ import annotations

import os
from typing import Any, Dict

from backend.agent.state import AgentState
from backend.observability.langsmith_tracer import (
    build_run_config,
    is_tracing_enabled,
)


def _make_state() -> AgentState:
    return {
        "session_id": "sess-1",
        "user_id": "user-1",
        "request_id": "req-1",
        "trace_id": "trace-1",
        "messages": [],
        "intent": "product_qa",
        "retrieved_docs": [],
        "tool_calls": [],
        "tool_results": [],
        "final_response": None,
        "should_escalate": False,
        "ticket_id": None,
    }


def test_is_tracing_enabled_false_without_env(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    assert is_tracing_enabled() is False


def test_is_tracing_enabled_true_with_env(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    assert is_tracing_enabled() is True


def test_build_run_config_includes_metadata_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")

    state = _make_state()
    config: Dict[str, Any] = build_run_config(state, thread_id="t-1")

    assert "configurable" in config
    assert config["configurable"]["thread_id"] == "t-1"
    meta = config.get("metadata") or {}
    assert meta.get("session_id") == "sess-1"
    assert meta.get("user_id") == "user-1"
    assert meta.get("request_id") == "req-1"
    assert meta.get("trace_id") == "trace-1"
    assert meta.get("intent") == "product_qa"
    tags = config.get("tags") or []
    assert "phase-6" in tags


def test_build_run_config_minimal_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    state = _make_state()
    config: Dict[str, Any] = build_run_config(state, thread_id=None)
    # No metadata/tags when tracing is off.
    assert config == {}

