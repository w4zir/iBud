from __future__ import annotations

from typing import Any, List

import pytest

from backend.evaluation.pipeline import AsyncEvaluator, EvaluationInput


class _FakeScalarResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class _FakeRowsResult:
    def __init__(self, rows: List[Any]) -> None:
        self._rows = rows

    def all(self) -> List[Any]:
        return self._rows


class _FakeDB:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._result


class _FakeSessionFactory:
    def __init__(self, db: _FakeDB) -> None:
        self._db = db

    async def __aenter__(self) -> _FakeDB:
        return self._db

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.mark.asyncio
async def test_sample_sessions_returns_only_ids(monkeypatch):
    db = _FakeDB(_FakeRowsResult([("s1",), ("s2",)]))
    monkeypatch.setattr(
        "backend.evaluation.pipeline.async_session_factory",
        lambda: _FakeSessionFactory(db),
    )
    evaluator = AsyncEvaluator()
    rows = await evaluator.sample_sessions(limit=10, min_age_minutes=1)
    assert rows == ["s1", "s2"]


@pytest.mark.asyncio
async def test_evaluate_session_skips_duplicate(monkeypatch):
    db = _FakeDB(_FakeScalarResult("existing-eval-id"))
    monkeypatch.setattr(
        "backend.evaluation.pipeline.async_session_factory",
        lambda: _FakeSessionFactory(db),
    )
    evaluator = AsyncEvaluator()
    ok = await evaluator.evaluate_session("sess-1")
    assert ok is False


@pytest.mark.asyncio
async def test_evaluate_session_writes_score(monkeypatch):
    db = _FakeDB(_FakeScalarResult(None))
    monkeypatch.setattr(
        "backend.evaluation.pipeline.async_session_factory",
        lambda: _FakeSessionFactory(db),
    )

    evaluator = AsyncEvaluator()
    async def _reconstruct(session_id: str) -> EvaluationInput:
        return EvaluationInput(
            session_id=session_id,
            user_message="hello",
            assistant_message="hi there",
            contexts=[],
        )

    async def _score(payload: EvaluationInput) -> dict[str, Any]:
        return {
            "groundedness": 0.8,
            "helpfulness": 0.9,
            "hallucination": False,
        }

    monkeypatch.setattr(evaluator, "reconstruct_inputs", _reconstruct)
    monkeypatch.setattr(evaluator, "_score_payload", _score)

    written: dict[str, Any] = {}

    async def _fake_record(**kwargs: Any) -> None:
        written.update(kwargs)

    monkeypatch.setattr("backend.evaluation.pipeline.record_evaluation_score", _fake_record)

    ok = await evaluator.evaluate_session("sess-1")
    assert ok is True
    assert written["session_id"] == "sess-1"
    assert written["groundedness"] == 0.8
    assert written["helpfulness"] == 0.9
    assert written["hallucination"] is False


@pytest.mark.asyncio
async def test_run_batch_is_idempotent(monkeypatch):
    evaluator = AsyncEvaluator()
    async def _sample(**kwargs: Any) -> list[str]:
        return ["s1", "s2"]

    monkeypatch.setattr(evaluator, "sample_sessions", _sample)

    seen: set[str] = set()

    async def _eval(session_id: str) -> bool:
        if session_id in seen:
            return False
        seen.add(session_id)
        return True

    monkeypatch.setattr(evaluator, "evaluate_session", _eval)

    first = await evaluator.run_batch(limit=10, min_age_minutes=1)
    second = await evaluator.run_batch(limit=10, min_age_minutes=1)

    assert first["processed"] == 2
    assert first["skipped"] == 0
    assert second["processed"] == 0
    assert second["skipped"] == 2

