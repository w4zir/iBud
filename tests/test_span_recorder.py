from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.observability.span_recorder import (
    finalize_session,
    persist_trace,
    record_evaluation,
    record_outcome,
    record_span,
)
from backend.observability.tracing import Trace, create_trace


def _mock_session_factory():
    """Return a mock async context manager that yields a mock session."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, mock_session


@pytest.mark.asyncio
async def test_record_span_persists(monkeypatch):
    ctx, mock_session = _mock_session_factory()
    monkeypatch.setattr(
        "backend.observability.span_recorder.async_session_factory", lambda: ctx
    )

    await record_span(
        trace_id="t-1",
        span_name="intent_detection",
        attributes={"intent": "order_status"},
        latency_ms=42.5,
    )

    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()

    row = mock_session.add.call_args[0][0]
    assert row.trace_id == "t-1"
    assert row.span_name == "intent_detection"
    assert row.attributes == {"intent": "order_status"}
    assert row.latency_ms == 42.5


@pytest.mark.asyncio
async def test_record_span_swallows_db_errors(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "backend.observability.span_recorder.async_session_factory", boom
    )

    # Must not raise
    await record_span(trace_id="t-x", span_name="boom", attributes={})


@pytest.mark.asyncio
async def test_record_outcome_persists(monkeypatch):
    ctx, mock_session = _mock_session_factory()
    monkeypatch.setattr(
        "backend.observability.span_recorder.async_session_factory", lambda: ctx
    )

    await record_outcome(
        session_id="sess-1",
        task="order_status",
        completed=True,
        verified=False,
    )

    row = mock_session.add.call_args[0][0]
    assert row.session_id == "sess-1"
    assert row.task == "order_status"
    assert row.completed is True
    assert row.verified is False


@pytest.mark.asyncio
async def test_record_evaluation_persists(monkeypatch):
    ctx, mock_session = _mock_session_factory()
    monkeypatch.setattr(
        "backend.observability.span_recorder.async_session_factory", lambda: ctx
    )

    await record_evaluation(
        session_id="sess-2",
        groundedness=0.85,
        hallucination=False,
        helpfulness=0.9,
        policy_compliance=1.0,
    )

    row = mock_session.add.call_args[0][0]
    assert row.session_id == "sess-2"
    assert row.groundedness == 0.85
    assert row.hallucination is False
    assert row.helpfulness == 0.9
    assert row.policy_compliance == 1.0


@pytest.mark.asyncio
async def test_finalize_session_updates_row(monkeypatch):
    mock_row = MagicMock()
    mock_row.end_time = None
    mock_row.intent = None
    mock_row.escalated = None

    ctx, mock_session = _mock_session_factory()
    mock_session.get = AsyncMock(return_value=mock_row)
    monkeypatch.setattr(
        "backend.observability.span_recorder.async_session_factory", lambda: ctx
    )

    await finalize_session("sess-5", intent="product_qa", escalated=False)

    assert mock_row.intent == "product_qa"
    assert mock_row.escalated is False
    assert mock_row.end_time is not None
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_session_missing_row(monkeypatch):
    ctx, mock_session = _mock_session_factory()
    mock_session.get = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "backend.observability.span_recorder.async_session_factory", lambda: ctx
    )

    # Should not raise
    await finalize_session("nonexistent")
    mock_session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_trace_records_all_spans(monkeypatch):
    recorded = []

    async def fake_record_span(trace_id, span_name, attributes, latency_ms=None):
        recorded.append((trace_id, span_name))

    monkeypatch.setattr(
        "backend.observability.span_recorder.record_span", fake_record_span
    )

    trace = create_trace("sess-10")
    s1 = trace.start_span("intent_detection")
    s1.finish()
    s2 = trace.start_span("retrieval")
    s2.finish()

    await persist_trace(trace)

    assert len(recorded) == 2
    assert recorded[0] == ("sess-10", "intent_detection")
    assert recorded[1] == ("sess-10", "retrieval")
