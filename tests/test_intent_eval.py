from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from evaluation import intent_eval


def _make_example(
    test_id: str,
    expected: str,
    predicted: str,
    split: str = "bitext_sampled",
) -> intent_eval.ExampleResult:
    return intent_eval.ExampleResult(
        test_id=test_id,
        split=split,
        question="q",
        expected_intent=expected,
        predicted_intent=predicted,
        is_correct=expected == predicted,
        session_id=None,
        error=None,
    )


def test_compute_metrics_accuracy_and_macro_f1() -> None:
    results = [
        _make_example("1", "order_status", "order_status"),
        _make_example("2", "order_status", "return_request"),
        _make_example("3", "return_request", "return_request"),
        _make_example("4", "product_qa", "product_qa"),
    ]

    metrics, confusion = intent_eval._compute_metrics(results)

    assert metrics["total_examples"] == 4
    assert metrics["correct_examples"] == 3
    assert metrics["failed_examples"] == 0
    assert pytest.approx(metrics["accuracy"]) == 0.75

    # Basic confusion sanity checks.
    assert confusion["order_status"]["order_status"] == 1
    assert confusion["order_status"]["return_request"] == 1
    assert confusion["return_request"]["return_request"] == 1


@pytest.mark.asyncio
async def test_persist_run_and_predictions_uses_warehouse_metadata(monkeypatch, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}

    async def fake_record_evaluation_score(
        *,
        session_id,
        groundedness,
        hallucination,
        helpfulness,
        metadata,
    ):
        captured["session_id"] = session_id
        captured["groundedness"] = groundedness
        captured["hallucination"] = hallucination
        captured["helpfulness"] = helpfulness
        captured["metadata"] = metadata

    # Avoid touching a real database by monkeypatching async_session_factory
    class DummySession:
        def __init__(self):
            self.added: List[Any] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def add(self, obj: Any) -> None:
            self.added.append(obj)

        async def flush(self) -> None:
            # Simulate server-side id assignment for the run.
            for obj in self.added:
                if isinstance(obj, intent_eval.IntentEvalRun) and not getattr(obj, "id", None):
                    obj.id = "run-1"

        async def commit(self) -> None:
            return None

    async def fake_async_session_factory():
        return DummySession()

    monkeypatch.setattr(intent_eval, "record_evaluation_score", fake_record_evaluation_score)
    monkeypatch.setattr(intent_eval, "async_session_factory", fake_async_session_factory)

    run_meta = {
        "experiment_name": "exp-1",
        "dataset_key": "bitext",
        "model_provider": "ollama",
        "model_name": "llama3.2",
        "prompt_version": "v1",
        "metadata": {"tag": "test"},
    }
    results = [
        _make_example("1", "order_status", "order_status"),
        _make_example("2", "order_status", "return_request"),
    ]
    metrics, confusion = intent_eval._compute_metrics(results)

    run_id = await intent_eval._persist_run_and_predictions(
        run_meta=run_meta,
        results=results,
        metrics=metrics,
        confusion=confusion,
    )

    assert run_id == "run-1"
    assert captured["session_id"] is None
    assert captured["groundedness"] is None
    assert captured["hallucination"] is None
    assert captured["helpfulness"] is None
    meta = captured["metadata"]
    assert meta["pipeline"] == "intent_eval"
    assert meta["run_id"] == "run-1"
    assert meta["metrics"]["total_examples"] == metrics["total_examples"]


def test_load_testset_supports_indices_and_limit(tmp_path: Path, monkeypatch) -> None:
    rows: List[Dict[str, Any]] = []
    for i in range(10):
        rows.append(
            {
                "id": f"row-{i}",
                "split": "bitext_sampled",
                "question": f"Q{i}",
                "answer": f"A{i}",
                "supporting_article": f"S{i}",
                "intent": "order_status",
            }
        )

    test_path = tmp_path / "bitext_testset.json"
    test_path.write_text(json.dumps(rows), encoding="utf-8")

    # Index-based selection.
    selected = intent_eval._load_testset(path=test_path, limit=None, indices=[3, 1], randomize=False, random_seed=None)
    assert [r["id"] for r in selected] == ["row-3", "row-1"]

    # Limit-based selection.
    limited = intent_eval._load_testset(path=test_path, limit=3, indices=None, randomize=False, random_seed=None)
    assert len(limited) == 3
    assert [r["id"] for r in limited] == ["row-0", "row-1", "row-2"]

