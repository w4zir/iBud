from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.observability.metrics_queries import (
    automation_rate,
    avg_groundedness,
    avg_helpfulness,
    avg_resolution_latency_ms,
    escalation_rate,
    hallucination_rate,
    tool_success_rate,
)


def _mock_session(scalars: list):
    """
    Build a mock AsyncSession whose `.execute()` returns scalar values
    in the order they are called.
    """
    session = AsyncMock()
    results = []
    for val in scalars:
        result_mock = MagicMock()
        result_mock.scalar.return_value = val
        results.append(result_mock)
    session.execute = AsyncMock(side_effect=results)
    return session


@pytest.mark.asyncio
async def test_automation_rate_basic():
    # total=10, resolved=6 → 0.6
    session = _mock_session([10, 6])
    rate = await automation_rate(session)
    assert rate == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_automation_rate_no_data():
    session = _mock_session([0, 0])
    rate = await automation_rate(session)
    assert rate is None


@pytest.mark.asyncio
async def test_escalation_rate_basic():
    # total=20, escalated=4 → 0.2
    session = _mock_session([20, 4])
    rate = await escalation_rate(session)
    assert rate == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_escalation_rate_no_data():
    session = _mock_session([0, 0])
    rate = await escalation_rate(session)
    assert rate is None


@pytest.mark.asyncio
async def test_tool_success_rate_returns_value():
    session = _mock_session([0.85])
    rate = await tool_success_rate(session)
    assert rate == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_tool_success_rate_returns_none():
    session = _mock_session([None])
    rate = await tool_success_rate(session)
    assert rate is None


@pytest.mark.asyncio
async def test_hallucination_rate_basic():
    # total=10, hallucinated=3 → 0.3
    session = _mock_session([10, 3])
    rate = await hallucination_rate(session)
    assert rate == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_hallucination_rate_no_data():
    session = _mock_session([0, 0])
    rate = await hallucination_rate(session)
    assert rate is None


@pytest.mark.asyncio
async def test_avg_resolution_latency():
    session = _mock_session([150.5])
    latency = await avg_resolution_latency_ms(session)
    assert latency == pytest.approx(150.5)


@pytest.mark.asyncio
async def test_avg_resolution_latency_no_data():
    session = _mock_session([None])
    latency = await avg_resolution_latency_ms(session)
    assert latency is None


@pytest.mark.asyncio
async def test_avg_groundedness():
    session = _mock_session([0.78])
    score = await avg_groundedness(session)
    assert score == pytest.approx(0.78)


@pytest.mark.asyncio
async def test_avg_helpfulness():
    session = _mock_session([0.91])
    score = await avg_helpfulness(session)
    assert score == pytest.approx(0.91)
