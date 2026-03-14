from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from backend.agent.state import AgentState
from backend.observability.langsmith_tracer import (
    build_run_config,
    build_stage_run_config,
    build_tool_run_config,
    chat_request_trace,
    get_parent_run_metadata,
    get_parent_run_tags,
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


def test_get_parent_run_metadata_includes_request_session_user_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(
        "backend.observability.langsmith_tracer.get_current_trace_ids",
        lambda: ("otel-trace-123", "otel-span-456"),
    )

    state = _make_state()
    meta = get_parent_run_metadata(state)

    assert meta.get("session_id") == "sess-1"
    assert meta.get("user_id") == "user-1"
    assert meta.get("request_id") == "req-1"
    assert meta.get("trace_id") == "trace-1"
    assert meta.get("otel_trace_id") == "otel-trace-123"
    assert meta.get("otel_span_id") == "otel-span-456"
    assert meta.get("phase") == "6"
    assert meta.get("component") == "chat_request"


def test_get_parent_run_metadata_includes_dataset_company_when_present(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(
        "backend.observability.langsmith_tracer.get_current_trace_ids",
        lambda: ("otel-trace-123", "otel-span-456"),
    )
    state = _make_state()
    state["dataset"] = "wixqa"
    state["company_id"] = "company-1"
    meta = get_parent_run_metadata(state)
    assert meta.get("dataset") == "wixqa"
    assert meta.get("company_id") == "company-1"


def test_get_parent_run_metadata_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    state = _make_state()
    assert get_parent_run_metadata(state) == {}


def test_get_parent_run_tags():
    tags = get_parent_run_tags()
    assert "agentic-rag" in tags
    assert "phase-6" in tags
    assert "langgraph-agent" in tags


@pytest.mark.asyncio
async def test_chat_request_trace_no_op_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    state = _make_state()
    entered = []

    async with chat_request_trace(state):
        entered.append(True)

    assert entered == [True]


@pytest.mark.asyncio
async def test_chat_request_trace_enters_context_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    fake_entered = []

    class FakeTraceCM:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeTraceCM":
            fake_entered.append(True)
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    def fake_get_trace() -> Any:
        return FakeTraceCM

    monkeypatch.setattr(
        "backend.observability.langsmith_tracer._get_trace_context_manager",
        fake_get_trace,
    )

    state = _make_state()
    async with chat_request_trace(state):
        pass

    assert fake_entered == [True]


def test_build_stage_run_config_includes_run_name_and_metadata_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(
        "backend.observability.langsmith_tracer.get_current_trace_ids",
        lambda: ("otel-trace-123", "otel-span-456"),
    )
    state = _make_state()
    config = build_stage_run_config(state, "agent.planner")
    assert config.get("run_name") == "agent.planner"
    meta = config.get("metadata") or {}
    assert meta.get("session_id") == "sess-1"
    assert meta.get("request_id") == "req-1"
    assert meta.get("stage") == "agent.planner"
    assert meta.get("component") == "agent_stage"
    tags = config.get("tags") or []
    assert "stage:agent.planner" in tags


def test_build_stage_run_config_includes_cycle_count(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(
        "backend.observability.langsmith_tracer.get_current_trace_ids",
        lambda: ("otel-trace-123", "otel-span-456"),
    )
    state = _make_state()
    config = build_stage_run_config(state, "agent.evaluator", cycle_count=2)
    assert config.get("metadata", {}).get("cycle_count") == 2


def test_build_stage_run_config_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    state = _make_state()
    config = build_stage_run_config(state, "agent.planner", cycle_count=1)
    assert config == {}


def test_build_tool_run_config_includes_run_name_and_metadata_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(
        "backend.observability.langsmith_tracer.get_current_trace_ids",
        lambda: ("otel-trace-123", "otel-span-456"),
    )
    state = _make_state()
    config = build_tool_run_config(state, "agent.tool.faq_search")
    assert config.get("run_name") == "agent.tool.faq_search"
    meta = config.get("metadata") or {}
    assert meta.get("component") == "agent_tool"
    assert meta.get("tool") == "agent.tool.faq_search"
    tags = config.get("tags") or []
    assert "tool:agent.tool.faq_search" in tags


def test_build_tool_run_config_includes_task_metadata(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(
        "backend.observability.langsmith_tracer.get_current_trace_ids",
        lambda: ("otel-trace-123", "otel-span-456"),
    )
    state = _make_state()
    config = build_tool_run_config(
        state,
        "agent.tool.order_lookup",
        task_id="t1",
        action="order_lookup",
        attempt=2,
        depends_on=["t0"],
    )
    meta = config.get("metadata") or {}
    assert meta.get("task_id") == "t1"
    assert meta.get("action") == "order_lookup"
    assert meta.get("attempt") == 2
    assert meta.get("depends_on") == ["t0"]


def test_build_tool_run_config_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    state = _make_state()
    config = build_tool_run_config(state, "agent.tool.faq_search", task_id="t1")
    assert config == {}

