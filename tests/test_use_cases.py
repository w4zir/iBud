import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


@pytest.mark.integration
def test_chat_creates_single_langsmith_parent_for_orchestrator(monkeypatch):
    """Chat flow runs run_orchestrated_agent inside one LangSmith parent trace context."""
    trace_entered = []
    orchestrator_called_inside_trace = []

    @asynccontextmanager
    async def fake_chat_request_trace(state):  # type: ignore[no-untyped-def]
        trace_entered.append(True)
        try:
            yield None
        finally:
            pass

    async def fake_run(state):  # type: ignore[arg-type]
        orchestrator_called_inside_trace.append(trace_entered and trace_entered[-1])
        new_state = dict(state)
        new_state["final_response"] = "Stubbed response"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.api.routes.chat as chat_routes
    import backend.agent.orchestrator as orch_module

    monkeypatch.setattr(chat_routes, "chat_request_trace", fake_chat_request_trace)
    monkeypatch.setattr(orch_module, "run_orchestrated_agent", fake_run)

    resp = client.post(
        "/chat/",
        json={
            "session_id": None,
            "user_id": "test-user",
            "message": "Hello",
            "dataset": "wixqa",
            "company": "default",
        },
    )
    assert resp.status_code == 200
    assert trace_entered == [True], "chat_request_trace context should be entered"
    assert orchestrator_called_inside_trace == [True], (
        "run_orchestrated_agent should be called inside chat_request_trace"
    )


@pytest.mark.integration
def test_chat_trace_collector_sees_chat_request_name_and_run_type(monkeypatch):
    """When tracing is enabled, parent run name is 'chat_request' and run_type is 'chain'."""
    collected = []

    class FakeTraceCM:
        def __init__(self, name: str, run_type: str = "chain", **kwargs: object) -> None:
            self.name = name
            self.run_type = run_type
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeTraceCM":
            collected.append({"name": self.name, "run_type": self.run_type})
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    def fake_get_trace() -> type:
        return FakeTraceCM

    import backend.observability.langsmith_tracer as tracer_module
    import backend.agent.orchestrator as orch_module

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "ecom-support-rag")
    monkeypatch.setattr(tracer_module, "_get_trace_context_manager", fake_get_trace)

    async def fake_run(state):  # type: ignore[arg-type]
        new_state = dict(state)
        new_state["final_response"] = "Stubbed"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    monkeypatch.setattr(orch_module, "run_orchestrated_agent", fake_run)

    resp = client.post(
        "/chat/",
        json={
            "session_id": None,
            "user_id": "test-user",
            "message": "Hi",
            "dataset": "wixqa",
            "company": "default",
        },
    )
    assert resp.status_code == 200
    assert len(collected) == 1, "exactly one parent trace should be created"
    assert collected[0]["name"] == "chat_request"
    assert collected[0]["run_type"] == "chain"


@pytest.mark.integration
def test_chat_stream_creates_single_langsmith_parent_for_orchestrator(monkeypatch):
    """Stream chat runs run_orchestrated_agent inside one LangSmith parent trace context."""
    trace_entered = []
    orchestrator_called_inside_trace = []

    @asynccontextmanager
    async def fake_chat_request_trace(state):  # type: ignore[no-untyped-def]
        trace_entered.append(True)
        try:
            yield None
        finally:
            pass

    async def fake_run(state):  # type: ignore[arg-type]
        orchestrator_called_inside_trace.append(trace_entered and trace_entered[-1])
        new_state = dict(state)
        new_state["final_response"] = "Streamed response"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.api.routes.chat as chat_routes
    import backend.agent.orchestrator as orch_module

    monkeypatch.setattr(chat_routes, "chat_request_trace", fake_chat_request_trace)
    monkeypatch.setattr(orch_module, "run_orchestrated_agent", fake_run)

    resp = client.post(
        "/chat/stream",
        json={
            "session_id": None,
            "user_id": "test-user",
            "message": "Hello",
            "dataset": "wixqa",
            "company": "default",
        },
    )
    assert resp.status_code == 200
    assert trace_entered == [True], "chat_request_trace context should be entered for stream"
    assert orchestrator_called_inside_trace == [True], (
        "run_orchestrated_agent should be called inside chat_request_trace for stream"
    )


@pytest.mark.integration
def test_chat_creates_session_and_persists_messages(monkeypatch):
    async def fake_run(state):  # type: ignore[arg-type]
        new_state = dict(state)
        new_state["final_response"] = "Stubbed response"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "run_orchestrated_agent", fake_run)

    resp = client.post(
        "/chat/",
        json={
            "session_id": None,
            "user_id": "test-user",
            "message": "Hello",
            "dataset": "wixqa",
            "company": "default",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]
    assert data["response"] == "Stubbed response"


@pytest.mark.integration
def test_chat_cache_same_query(monkeypatch):
    calls = {"count": 0}

    async def fake_run(state):  # type: ignore[arg-type]
        calls["count"] += 1
        new_state = dict(state)
        new_state["final_response"] = f"Response {calls['count']}"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "run_orchestrated_agent", fake_run)

    first = client.post(
        "/chat/",
        json={
            "session_id": None,
            "user_id": "test-user",
            "message": "Hi there",
            "dataset": "wixqa",
            "company": "default",
        },
    )
    assert first.status_code == 200
    first_data = first.json()
    session_id = first_data["session_id"]
    assert first_data["response"] == "Response 1"

    second = client.post(
        "/chat/",
        json={
            "session_id": session_id,
            "user_id": "test-user",
            "message": "Hi there",
            "dataset": "wixqa",
            "company": "default",
        },
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["session_id"] == session_id
    assert second_data["response"] == "Response 1"


@pytest.mark.integration
def test_sessions_history_endpoint(monkeypatch):
    async def fake_run(state):  # type: ignore[arg-type]
        new_state = dict(state)
        new_state["final_response"] = "Session test response"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "run_orchestrated_agent", fake_run)

    resp = client.post(
        "/chat/",
        json={
            "session_id": None,
            "user_id": "history-user",
            "message": "Check history",
            "dataset": "wixqa",
            "company": "default",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    session_id = data["session_id"]

    hist = client.get(f"/sessions/{session_id}/history")
    assert hist.status_code == 200
    hist_data = hist.json()
    assert len(hist_data["messages"]) >= 2
    roles = [m["role"] for m in hist_data["messages"]]
    assert "user" in roles and "assistant" in roles


@pytest.mark.integration
def test_chat_intent_endpoint_persists_and_classifies(monkeypatch):
    async def fake_classify_intent_only(state):  # type: ignore[arg-type]
        new_state = dict(state)
        new_state["intent"] = "cancel_order"
        new_state["intent_prompt_profile"] = state.get("intent_prompt_profile") or "default"
        return new_state

    import backend.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "classify_intent_only", fake_classify_intent_only)

    # Ensure LangSmith is explicitly disabled for intent-only flow even if a
    # langsmith module exists at runtime.
    import contextlib
    import sys

    calls = {"enabled": None, "entered": 0}

    @contextlib.contextmanager
    def fake_tracing_context(*, enabled=None, **kwargs):  # type: ignore[no-untyped-def]
        calls["enabled"] = enabled
        calls["entered"] += 1
        yield

    sys.modules["langsmith"] = type(
        "FakeLangsmith",
        (),
        {"tracing_context": staticmethod(fake_tracing_context)},
    )()

    resp = client.post(
        "/chat/intent",
        json={
            "session_id": None,
            "user_id": "intent-user",
            "message": "please cancel my order",
            "dataset": "bitext",
            "intent_prompt_profile": "bitext",
            "company": "default",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]
    assert data["intent"] == "cancel_order"
    assert data["intent_prompt_profile"] == "bitext"
    assert calls["entered"] == 1
    assert calls["enabled"] is False

