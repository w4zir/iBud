import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


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

