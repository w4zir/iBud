from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.observability.evaluation import (
    EvaluationResult,
    evaluate_groundedness,
    evaluate_hallucination,
    evaluate_helpfulness,
    evaluate_policy_compliance,
    evaluate_session,
)


def _make_llm(content: str) -> AsyncMock:
    """Create a mock LLM that returns the given content string."""
    resp = MagicMock()
    resp.content = content
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=resp)
    return llm


@pytest.mark.asyncio
async def test_evaluate_groundedness_parses_score():
    llm = _make_llm('{"score": 0.85}')
    score = await evaluate_groundedness("answer", "context", llm=llm)
    assert score == 0.85
    llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_groundedness_returns_none_on_bad_json():
    llm = _make_llm("not json at all")
    score = await evaluate_groundedness("answer", "context", llm=llm)
    assert score is None


@pytest.mark.asyncio
async def test_evaluate_groundedness_returns_none_on_exception():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
    score = await evaluate_groundedness("answer", "context", llm=llm)
    assert score is None


@pytest.mark.asyncio
async def test_evaluate_hallucination_true():
    llm = _make_llm('{"hallucinated": true}')
    result = await evaluate_hallucination("answer", "context", llm=llm)
    assert result is True


@pytest.mark.asyncio
async def test_evaluate_hallucination_false():
    llm = _make_llm('{"hallucinated": false}')
    result = await evaluate_hallucination("answer", "context", llm=llm)
    assert result is False


@pytest.mark.asyncio
async def test_evaluate_hallucination_returns_none_on_bad_json():
    llm = _make_llm("maybe?")
    result = await evaluate_hallucination("answer", "context", llm=llm)
    assert result is None


@pytest.mark.asyncio
async def test_evaluate_helpfulness_parses_score():
    llm = _make_llm('{"score": 0.72}')
    score = await evaluate_helpfulness("answer", "query", llm=llm)
    assert score == 0.72


@pytest.mark.asyncio
async def test_evaluate_policy_compliance_parses_score():
    llm = _make_llm('{"score": 1.0}')
    score = await evaluate_policy_compliance("answer", llm=llm)
    assert score == 1.0


@pytest.mark.asyncio
async def test_evaluate_session_full_pipeline():
    llm = _make_llm('{"score": 0.9}')

    hallucination_llm = _make_llm('{"hallucinated": false}')

    call_count = 0
    original_ainvoke = llm.ainvoke

    async def routing_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return MagicMock(content='{"hallucinated": false}')
        return await original_ainvoke(messages)

    llm.ainvoke = AsyncMock(side_effect=routing_ainvoke)

    result = await evaluate_session(
        session_id="sess-1",
        query="Where is my order?",
        response="Your order is being shipped.",
        context="Order 123 is in transit.",
        llm=llm,
    )

    assert isinstance(result, EvaluationResult)
    assert result.session_id == "sess-1"
    assert result.groundedness == 0.9
    assert result.hallucination is False
    assert result.helpfulness == 0.9
    assert result.policy_compliance == 0.9


@pytest.mark.asyncio
async def test_evaluate_session_partial_failures():
    """Individual evaluator failures should not block the others."""
    call_count = 0

    async def mixed_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("LLM timeout")
        resp = MagicMock()
        resp.content = '{"score": 0.5}' if call_count != 2 else '{"hallucinated": true}'
        return resp

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=mixed_ainvoke)

    result = await evaluate_session(
        session_id="sess-2",
        query="q",
        response="r",
        context="c",
        llm=llm,
    )

    assert result.session_id == "sess-2"
    assert result.groundedness is None  # first call failed
    assert result.hallucination is True
