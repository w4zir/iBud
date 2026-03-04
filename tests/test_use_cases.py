import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


@pytest.mark.integration
def test_chat_creates_session_and_persists_messages(monkeypatch):
    def fake_run_agent(state, thread_id=None):  # type: ignore[arg-type]
        new_state = dict(state)
        new_state["final_response"] = "Stubbed response"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "run_agent", fake_run_agent)

    resp = client.post(
        "/chat/",
        json={"session_id": None, "user_id": "test-user", "message": "Hello"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]
    assert data["response"] == "Stubbed response"


@pytest.mark.integration
def test_chat_cache_same_query(monkeypatch):
    calls = {"count": 0}

    def fake_run_agent(state, thread_id=None):  # type: ignore[arg-type]
        calls["count"] += 1
        new_state = dict(state)
        new_state["final_response"] = f"Response {calls['count']}"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "run_agent", fake_run_agent)

    first = client.post(
        "/chat/",
        json={"session_id": None, "user_id": "test-user", "message": "Hi there"},
    )
    assert first.status_code == 200
    first_data = first.json()
    session_id = first_data["session_id"]
    assert first_data["response"] == "Response 1"

    second = client.post(
        "/chat/",
        json={"session_id": session_id, "user_id": "test-user", "message": "Hi there"},
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["session_id"] == session_id
    assert second_data["response"] == "Response 1"


@pytest.mark.integration
def test_sessions_history_endpoint(monkeypatch):
    def fake_run_agent(state, thread_id=None):  # type: ignore[arg-type]
        new_state = dict(state)
        new_state["final_response"] = "Session test response"
        new_state["retrieved_docs"] = []
        new_state["tool_results"] = []
        new_state["should_escalate"] = False
        new_state["ticket_id"] = None
        return new_state

    import backend.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "run_agent", fake_run_agent)

    resp = client.post(
        "/chat/",
        json={"session_id": None, "user_id": "history-user", "message": "Check history"},
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

