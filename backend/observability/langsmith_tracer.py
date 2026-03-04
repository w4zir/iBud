from __future__ import annotations

import os
from typing import Any, Dict

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


__all__ = ["is_tracing_enabled", "build_run_config"]

