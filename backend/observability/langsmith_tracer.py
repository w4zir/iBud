from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from ..agent.state import AgentState
from .otel import get_current_trace_ids


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def is_tracing_enabled() -> bool:
    """
    Return True when LangSmith tracing should be enabled.

    This follows the Phase 6 contract: tracing is opt-in via
    LANGCHAIN_TRACING_V2 plus LangSmith credentials in the environment.
    """
    if not _env_flag("LANGCHAIN_TRACING_V2"):
        return False
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")
    return bool(api_key and project)


def build_run_config(state: AgentState, *, thread_id: str | None = None) -> Dict[str, Any]:
    """
    Build LangGraph run config with LangSmith-friendly metadata.

    LangGraph and LangChain will automatically emit traces to LangSmith
    when LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT
    are set. This helper centralises config construction so that every
    agent run is tagged consistently.
    """
    config: Dict[str, Any] = {}

    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    if not is_tracing_enabled():
        return config

    otel_trace_id, otel_span_id = get_current_trace_ids()
    metadata: Dict[str, Any] = {
        "session_id": state.get("session_id"),
        "user_id": state.get("user_id"),
        "request_id": state.get("request_id"),
        "trace_id": state.get("trace_id") or otel_trace_id,
        "otel_trace_id": otel_trace_id,
        "otel_span_id": otel_span_id,
        "intent": state.get("intent"),
        "phase": "6",
        "component": "agent_graph",
    }

    # Drop unset values to keep traces clean.
    config["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    config["tags"] = ["agentic-rag", "phase-6", "langgraph-agent"]

    return config


def get_parent_run_metadata(state: AgentState) -> Dict[str, Any]:
    """
    Build metadata for a single parent LangSmith run (e.g. chat_request).
    Used so all nested LLM/tool calls appear under one trace window.
    Returns only set values; safe to pass to trace(metadata=...).
    Includes route-level hints (dataset, company_id) for filtering in the UI.
    """
    if not is_tracing_enabled():
        return {}
    otel_trace_id, otel_span_id = get_current_trace_ids()
    metadata: Dict[str, Any] = {
        "session_id": state.get("session_id"),
        "user_id": state.get("user_id"),
        "request_id": state.get("request_id"),
        "trace_id": state.get("trace_id") or otel_trace_id,
        "otel_trace_id": otel_trace_id,
        "otel_span_id": otel_span_id,
        "phase": "6",
        "component": "chat_request",
        "dataset": state.get("dataset"),
        "company_id": state.get("company_id"),
    }
    return {k: v for k, v in metadata.items() if v is not None}


def get_parent_run_tags() -> List[str]:
    """Tags for the chat_request parent run."""
    return ["agentic-rag", "phase-6", "langgraph-agent"]


def build_stage_run_config(
    state: AgentState,
    run_name: str,
    *,
    cycle_count: int | None = None,
    **extra_metadata: Any,
) -> Dict[str, Any]:
    """
    Build LangSmith run config for a named agent stage (LLM call).
    Use as config= for ainvoke so the run appears with a readable name
    and inherits request metadata under the chat_request parent.
    """
    config: Dict[str, Any] = {}
    if not is_tracing_enabled():
        return config
    otel_trace_id, otel_span_id = get_current_trace_ids()
    metadata: Dict[str, Any] = {
        "session_id": state.get("session_id"),
        "user_id": state.get("user_id"),
        "request_id": state.get("request_id"),
        "trace_id": state.get("trace_id") or otel_trace_id,
        "otel_trace_id": otel_trace_id,
        "otel_span_id": otel_span_id,
        "intent": state.get("intent"),
        "phase": "6",
        "component": "agent_stage",
        "stage": run_name,
        **extra_metadata,
    }
    if cycle_count is not None:
        metadata["cycle_count"] = cycle_count
    config["run_name"] = run_name
    config["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    config["tags"] = ["agentic-rag", "phase-6", "langgraph-agent", f"stage:{run_name}"]
    return config


def build_tool_run_config(
    state: AgentState,
    run_name: str,
    *,
    task_id: str | None = None,
    action: str | None = None,
    attempt: int | None = None,
    depends_on: List[str] | None = None,
    **extra_metadata: Any,
) -> Dict[str, Any]:
    """
    Build LangSmith run config for a named tool/action run.
    Use when opening a trace span around deterministic tool execution
    so tool calls appear as named children in the tree.
    """
    config: Dict[str, Any] = {}
    if not is_tracing_enabled():
        return config
    otel_trace_id, otel_span_id = get_current_trace_ids()
    metadata: Dict[str, Any] = {
        "session_id": state.get("session_id"),
        "request_id": state.get("request_id"),
        "trace_id": state.get("trace_id") or otel_trace_id,
        "otel_trace_id": otel_trace_id,
        "otel_span_id": otel_span_id,
        "phase": "6",
        "component": "agent_tool",
        "tool": run_name,
        **extra_metadata,
    }
    if task_id is not None:
        metadata["task_id"] = task_id
    if action is not None:
        metadata["action"] = action
    if attempt is not None:
        metadata["attempt"] = attempt
    if depends_on is not None:
        metadata["depends_on"] = depends_on
    config["run_name"] = run_name
    config["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    config["tags"] = ["agentic-rag", "phase-6", "langgraph-agent", f"tool:{run_name}"]
    return config


def _get_trace_context_manager():  # type: ignore[no-untyped-def]
    """Return the trace context manager from langsmith if available."""
    try:
        from langsmith.run_helpers import trace
        return trace
    except Exception:
        pass
    try:
        import langsmith as ls  # type: ignore[import-not-found]
        return getattr(ls, "trace", None)
    except Exception:
        return None


@asynccontextmanager
async def chat_request_trace(state: AgentState) -> AsyncIterator[Any]:
    """
    Async context manager that wraps the enclosed code in a single LangSmith
    parent run when tracing is enabled. All nested LangChain/LangSmith calls
    (e.g. classifier, planner, evaluator, validator, response LLM) will appear
    as children under one trace window. No-op when tracing is disabled or
    langsmith is unavailable.

    Uses async trace context when available; preserves a single parent run
    named "chat_request" per request for tree visibility in the UI.
    """
    if not is_tracing_enabled():
        yield None
        return
    trace_cm = _get_trace_context_manager()
    if trace_cm is None:
        yield None
        return
    metadata = get_parent_run_metadata(state)
    tags = get_parent_run_tags()
    # LangSmith run_helpers.trace supports async context manager protocol.
    # Use async with so nested runs (LLM/tool) attach to this parent.
    async with trace_cm(
        "chat_request",
        run_type="chain",
        metadata=metadata,
        tags=tags,
    ):
        yield None


@asynccontextmanager
async def tool_trace(
    state: AgentState,
    run_name: str,
    *,
    task_id: str | None = None,
    action: str | None = None,
    attempt: int | None = None,
    depends_on: List[str] | None = None,
    **extra_metadata: Any,
) -> AsyncIterator[Any]:
    """
    Async context manager for a named tool run under the current trace.
    Use around deterministic tool execution so tool calls appear as named children.
    """
    if not is_tracing_enabled():
        yield None
        return
    trace_cm = _get_trace_context_manager()
    if trace_cm is None:
        yield None
        return
    config = build_tool_run_config(
        state,
        run_name,
        task_id=task_id,
        action=action,
        attempt=attempt,
        depends_on=depends_on,
        **extra_metadata,
    )
    if not config:
        yield None
        return
    async with trace_cm(
        config["run_name"],
        run_type="tool",
        metadata=config.get("metadata") or {},
        tags=config.get("tags") or [],
    ):
        yield None


__all__ = [
    "is_tracing_enabled",
    "build_run_config",
    "build_stage_run_config",
    "build_tool_run_config",
    "get_parent_run_metadata",
    "get_parent_run_tags",
    "chat_request_trace",
    "tool_trace",
]

