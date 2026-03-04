from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
from openai import AsyncOpenAI

from evaluation import ragas_eval


class DummyResult:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def _write_testset(tmp_path: Path, rows: List[Dict[str, Any]]) -> Path:
    testset_path = tmp_path / "wixqa_testset.json"
    testset_path.write_text(json.dumps(rows), encoding="utf-8")
    return testset_path


def test_run_ragas_eval_writes_summary(tmp_path, monkeypatch):
    rows = [
        {
            "id": "expertwritten-0",
            "split": "expertwritten",
            "question": "Q1",
            "answer": "A1",
            "supporting_article": "S1",
        }
    ]

    testset_path = _write_testset(tmp_path, rows)

    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)

    def fake_backend_calls(backend_url: str, loaded_rows: List[Dict[str, Any]], use_local: bool = False):
        assert backend_url == "http://localhost:8000"
        assert loaded_rows == rows
        return {
            "user_input": [rows[0]["question"]],
            "response": ["model answer"],
            "retrieved_contexts": [["ctx"]],
            "reference": [rows[0]["answer"]],
            "split": [rows[0]["split"]],
        }

    def fake_evaluate(dataset, show_progress=True):
        df = pd.DataFrame(
            [
                {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                    "context_precision": 0.85,
                    "context_recall": 0.88,
                    "split": "expertwritten",
                }
            ]
        )
        return DummyResult(df)

    monkeypatch.setattr(ragas_eval, "_run_backend_calls", fake_backend_calls)
    monkeypatch.setattr(ragas_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    summary = ragas_eval.run_ragas_eval("http://localhost:8000", limit=None)
    assert "metrics" in summary and "by_split" in summary
    assert summary["metrics"]["faithfulness"] == 0.9
    assert "expertwritten" in summary["by_split"]


def test_run_ragas_eval_handles_missing_metric_columns(tmp_path, monkeypatch):
    rows = [
        {
            "id": "expertwritten-0",
            "split": "expertwritten",
            "question": "Q1",
            "answer": "A1",
            "supporting_article": "S1",
        }
    ]

    testset_path = _write_testset(tmp_path, rows)

    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)

    def fake_backend_calls(backend_url: str, loaded_rows: List[Dict[str, Any]], use_local: bool = False):
        return {
            "user_input": [rows[0]["question"]],
            "response": ["model answer"],
            "retrieved_contexts": [["ctx"]],
            "reference": [rows[0]["answer"]],
            "split": [rows[0]["split"]],
        }

    def fake_evaluate(dataset, show_progress=True):
        # Deliberately omit some metric columns to exercise the conditional mean logic.
        df = pd.DataFrame(
            [
                {
                    "faithfulness": 0.7,
                    "answer_relevancy": 0.6,
                }
            ]
        )
        return DummyResult(df)

    monkeypatch.setattr(ragas_eval, "_run_backend_calls", fake_backend_calls)
    monkeypatch.setattr(ragas_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    summary = ragas_eval.run_ragas_eval("http://localhost:8000", limit=None)
    assert "metrics" in summary
    assert summary["metrics"]["faithfulness"] == 0.7
    assert summary["metrics"]["answer_relevancy"] == 0.6
    assert summary["metrics"]["context_precision"] is None
    assert summary["metrics"]["context_recall"] is None


def test_load_testset_supports_index_selection(tmp_path, monkeypatch):
    rows = [
        {
            "id": f"row-{i}",
            "split": "expertwritten",
            "question": f"Q{i}",
            "answer": f"A{i}",
            "supporting_article": f"S{i}",
        }
        for i in range(5)
    ]

    testset_path = _write_testset(tmp_path, rows)
    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)

    # Select a subset of rows by 0-based indices, in non-sequential order.
    selected = ragas_eval._load_testset(indices=[3, 1])
    assert [r["id"] for r in selected] == ["row-3", "row-1"]


def test_load_testset_randomize_respects_seed(tmp_path, monkeypatch):
    rows = [
        {
            "id": f"row-{i}",
            "split": "expertwritten",
            "question": f"Q{i}",
            "answer": f"A{i}",
            "supporting_article": f"S{i}",
        }
        for i in range(10)
    ]

    testset_path = _write_testset(tmp_path, rows)
    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)

    first = ragas_eval._load_testset(limit=5, randomize=True, random_seed=123)
    second = ragas_eval._load_testset(limit=5, randomize=True, random_seed=123)
    third = ragas_eval._load_testset(limit=5, randomize=True, random_seed=456)

    # Same seed should give identical selection/order.
    assert [r["id"] for r in first] == [r["id"] for r in second]
    # Different seed should very likely give a different order or subset.
    assert [r["id"] for r in first] != [r["id"] for r in third]


def test_ragas_llm_falls_back_to_runtime_cerebras_when_no_judge_env(monkeypatch):
    # No RAGAS_* overrides; use the app-wide Cerebras settings.
    monkeypatch.delenv("RAGAS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RAGAS_CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("RAGAS_CEREBRAS_MODEL", raising=False)
    monkeypatch.delenv("RAGAS_CEREBRAS_BASE_URL", raising=False)

    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "app-key")
    monkeypatch.setenv("CEREBRAS_MODEL", "llama3.1-8b-app")
    monkeypatch.setenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1-app")

    captured: Dict[str, Any] = {}

    def fake_llm_factory(model: str, client: AsyncOpenAI | None = None):
        captured["model"] = model
        captured["client"] = client
        return "dummy-llm"

    monkeypatch.setattr(ragas_eval, "llm_factory", fake_llm_factory)

    llm = ragas_eval._build_ragas_llm_from_env()

    assert llm == "dummy-llm"
    assert captured["model"] == "llama3.1-8b-app"
    assert isinstance(captured["client"], AsyncOpenAI)
    assert captured["client"].api_key == "app-key"
    # AsyncOpenAI normalises the base URL and exposes it as a URL object.
    assert str(captured["client"].base_url).rstrip("/") == "https://api.cerebras.ai/v1-app"


def test_ragas_llm_uses_judge_specific_cerebras_over_runtime(monkeypatch):
    # Judge-specific Cerebras settings should override app-wide ones.
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "app-key")
    monkeypatch.setenv("CEREBRAS_MODEL", "llama3.1-8b-app")
    monkeypatch.setenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1-app")

    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("RAGAS_CEREBRAS_API_KEY", "judge-key")
    monkeypatch.setenv("RAGAS_CEREBRAS_MODEL", "llama3.1-8b-judge")
    monkeypatch.setenv("RAGAS_CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1-judge")

    captured: Dict[str, Any] = {}

    def fake_llm_factory(model: str, client: AsyncOpenAI | None = None):
        captured["model"] = model
        captured["client"] = client
        return "dummy-llm"

    monkeypatch.setattr(ragas_eval, "llm_factory", fake_llm_factory)

    llm = ragas_eval._build_ragas_llm_from_env()

    assert llm == "dummy-llm"
    assert captured["model"] == "llama3.1-8b-judge"
    assert isinstance(captured["client"], AsyncOpenAI)
    assert captured["client"].api_key == "judge-key"
    # AsyncOpenAI normalises the base URL and exposes it as a URL object.
    assert str(captured["client"].base_url).rstrip("/") == "https://api.cerebras.ai/v1-judge"


def test_ragas_llm_can_use_openai_as_judge_independent_of_runtime(monkeypatch):
    # Even if the runtime provider is Cerebras (or something else), we can
    # force RAGAS to use OpenAI as its judge via RAGAS_LLM_PROVIDER.
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "app-key")

    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_OPENAI_MODEL", "gpt-4.1-mini-judge")

    captured: Dict[str, Any] = {}

    def fake_llm_factory(model: str, client: Any | None = None):
        captured["model"] = model
        captured["client"] = client
        return "dummy-openai-llm"

    monkeypatch.setattr(ragas_eval, "llm_factory", fake_llm_factory)

    llm = ragas_eval._build_ragas_llm_from_env()

    assert llm == "dummy-openai-llm"
    assert captured["model"] == "gpt-4.1-mini-judge"
    # For OpenAI we expect an AsyncOpenAI client to be constructed, with
    # credentials read from the environment.
    assert isinstance(captured["client"], AsyncOpenAI)


def test_ragas_llm_falls_back_to_runtime_ollama_when_no_judge_env(monkeypatch):
    # No RAGAS_* overrides; use the app-wide Ollama settings.
    monkeypatch.delenv("RAGAS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RAGAS_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("RAGAS_OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_OLLAMA_API_KEY", raising=False)

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-app")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-app:11434")

    captured: Dict[str, Any] = {}

    def fake_llm_factory(model: str, client: AsyncOpenAI | None = None):
        captured["model"] = model
        captured["client"] = client
        return "dummy-ollama-llm"

    monkeypatch.setattr(ragas_eval, "llm_factory", fake_llm_factory)

    llm = ragas_eval._build_ragas_llm_from_env()

    assert llm == "dummy-ollama-llm"
    assert captured["model"] == "llama3.2-app"
    assert isinstance(captured["client"], AsyncOpenAI)
    # AsyncOpenAI normalises the base URL and exposes it as a URL object.
    assert (
        str(captured["client"].base_url).rstrip("/")
        == "http://ollama-app:11434/v1"
    )


def test_ragas_llm_uses_judge_specific_ollama_over_runtime(monkeypatch):
    # Judge-specific Ollama settings should override app-wide ones.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2-app")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-app:11434")

    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("RAGAS_OLLAMA_MODEL", "llama3.2-judge")
    monkeypatch.setenv("RAGAS_OLLAMA_BASE_URL", "http://ollama-judge:11434")

    captured: Dict[str, Any] = {}

    def fake_llm_factory(model: str, client: AsyncOpenAI | None = None):
        captured["model"] = model
        captured["client"] = client
        return "dummy-ollama-judge-llm"

    monkeypatch.setattr(ragas_eval, "llm_factory", fake_llm_factory)

    llm = ragas_eval._build_ragas_llm_from_env()

    assert llm == "dummy-ollama-judge-llm"
    assert captured["model"] == "llama3.2-judge"
    assert isinstance(captured["client"], AsyncOpenAI)
    assert (
        str(captured["client"].base_url).rstrip("/")
        == "http://ollama-judge:11434/v1"
    )


def test_use_local_cache_hit_does_not_call_backend(tmp_path, monkeypatch):
    """With use_local=True and a pre-filled cache, backend is not called."""
    rows = [
        {
            "id": "expertwritten-0",
            "split": "expertwritten",
            "question": "Q1",
            "answer": "A1",
            "supporting_article": "S1",
        }
    ]
    _write_testset(tmp_path, rows)
    cache_path = tmp_path / "agent_responses.json"
    cache_path.write_text(
        json.dumps({"Q1": {"response": "cached answer", "retrieved_contexts": ["ctx1"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", tmp_path / "wixqa_testset.json")
    monkeypatch.setattr(ragas_eval, "AGENT_RESPONSES_PATH", cache_path)
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    class ClientThatMustNotBeCalled:
        def post(self, *args, **kwargs):
            raise AssertionError("backend must not be called on cache hit")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(ragas_eval.httpx, "Client", lambda **kw: ClientThatMustNotBeCalled())

    def fake_evaluate(dataset, show_progress=True):
        return DummyResult(pd.DataFrame([{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.85, "context_recall": 0.88, "split": "expertwritten"}]))

    monkeypatch.setattr(ragas_eval, "evaluate", fake_evaluate)

    summary = ragas_eval.run_ragas_eval("http://localhost:8000", limit=None, use_local=True)
    assert summary["metrics"]["faithfulness"] == 0.9
    assert summary["num_rows"] == 1


def test_use_local_cache_miss_calls_backend_and_writes_cache(tmp_path, monkeypatch):
    """With use_local=True and empty cache, backend is called and cache file is written."""
    rows = [
        {
            "id": "expertwritten-0",
            "split": "expertwritten",
            "question": "Q1",
            "answer": "A1",
            "supporting_article": "S1",
        }
    ]
    _write_testset(tmp_path, rows)
    cache_path = tmp_path / "agent_responses.json"
    assert not cache_path.exists()

    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", tmp_path / "wixqa_testset.json")
    monkeypatch.setattr(ragas_eval, "AGENT_RESPONSES_PATH", cache_path)
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    def fake_post(*args, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"response": "model answer", "sources": [{"content": "ctx"}]}
        return resp

    class FakeClient:
        def post(self, *args, **kwargs):
            return fake_post(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(ragas_eval.httpx, "Client", lambda **kw: FakeClient())

    def fake_evaluate(dataset, show_progress=True):
        return DummyResult(pd.DataFrame([{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.85, "context_recall": 0.88, "split": "expertwritten"}]))

    monkeypatch.setattr(ragas_eval, "evaluate", fake_evaluate)

    summary = ragas_eval.run_ragas_eval("http://localhost:8000", limit=None, use_local=True)

    assert summary["num_rows"] == 1
    assert cache_path.exists()
    cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "Q1" in cache_data
    assert cache_data["Q1"]["response"] == "model answer"
    assert cache_data["Q1"]["retrieved_contexts"] == ["ctx"]


def test_nan_scores_excluded_from_aggregation(tmp_path, monkeypatch):
    rows = [
        {
            "id": "expertwritten-0",
            "split": "expertwritten",
            "question": "Q1",
            "answer": "A1",
            "supporting_article": "S1",
        },
        {
            "id": "expertwritten-1",
            "split": "expertwritten",
            "question": "Q2",
            "answer": "A2",
            "supporting_article": "S2",
        },
    ]
    testset_path = _write_testset(tmp_path, rows)
    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    def fake_backend_calls(backend_url: str, loaded_rows: List[Dict[str, Any]], use_local: bool = False):
        return {
            "user_input": [r["question"] for r in loaded_rows],
            "response": ["r1", "r2"],
            "retrieved_contexts": [["ctx1"], ["ctx2"]],
            "reference": [r["answer"] for r in loaded_rows],
            "split": [r["split"] for r in loaded_rows],
        }

    def fake_evaluate(dataset, show_progress=True):
        df = pd.DataFrame(
            [
                {
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.7,
                    "context_precision": float("nan"),
                    "context_recall": 0.6,
                    "split": "expertwritten",
                },
                {
                    "faithfulness": float("nan"),
                    "answer_relevancy": 0.9,
                    "context_precision": 0.5,
                    "context_recall": float("nan"),
                    "split": "expertwritten",
                },
            ]
        )
        return DummyResult(df)

    monkeypatch.setattr(ragas_eval, "_run_backend_calls", fake_backend_calls)
    monkeypatch.setattr(ragas_eval, "evaluate", fake_evaluate)

    summary = ragas_eval.run_ragas_eval("http://localhost:8000", limit=None)
    assert summary["metrics"]["faithfulness"] == 0.8
    assert summary["metrics"]["answer_relevancy"] == 0.8
    assert summary["metrics"]["context_precision"] == 0.5
    assert summary["metrics"]["context_recall"] == 0.6
    assert summary["valid_counts"]["faithfulness"] == 1
    assert summary["valid_counts"]["answer_relevancy"] == 2


def test_all_nan_scores_produce_none_metric(tmp_path, monkeypatch):
    rows = [
        {
            "id": "expertwritten-0",
            "split": "expertwritten",
            "question": "Q1",
            "answer": "A1",
            "supporting_article": "S1",
        }
    ]
    testset_path = _write_testset(tmp_path, rows)
    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    def fake_backend_calls(backend_url: str, loaded_rows: List[Dict[str, Any]], use_local: bool = False):
        return {
            "user_input": [rows[0]["question"]],
            "response": ["model answer"],
            "retrieved_contexts": [["ctx"]],
            "reference": [rows[0]["answer"]],
            "split": [rows[0]["split"]],
        }

    def fake_evaluate(dataset, show_progress=True):
        return DummyResult(
            pd.DataFrame(
                [
                    {
                        "faithfulness": float("nan"),
                        "answer_relevancy": float("nan"),
                        "context_precision": float("nan"),
                        "context_recall": float("nan"),
                        "split": "expertwritten",
                    }
                ]
            )
        )

    monkeypatch.setattr(ragas_eval, "_run_backend_calls", fake_backend_calls)
    monkeypatch.setattr(ragas_eval, "evaluate", fake_evaluate)

    summary = ragas_eval.run_ragas_eval("http://localhost:8000", limit=None)
    assert summary["metrics"]["faithfulness"] is None
    assert summary["metrics"]["answer_relevancy"] is None
    assert summary["valid_counts"]["faithfulness"] == 0
    assert summary["valid_counts"]["answer_relevancy"] == 0


def test_pass_fail_thresholds_in_summary(tmp_path, monkeypatch):
    rows = [
        {
            "id": "expertwritten-0",
            "split": "expertwritten",
            "question": "Q1",
            "answer": "A1",
            "supporting_article": "S1",
        }
    ]
    testset_path = _write_testset(tmp_path, rows)
    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    def fake_backend_calls(backend_url: str, loaded_rows: List[Dict[str, Any]], use_local: bool = False):
        return {
            "user_input": [rows[0]["question"]],
            "response": ["model answer"],
            "retrieved_contexts": [["ctx"]],
            "reference": [rows[0]["answer"]],
            "split": [rows[0]["split"]],
        }

    def fake_evaluate(dataset, show_progress=True):
        return DummyResult(
            pd.DataFrame(
                [
                    {
                        "faithfulness": 0.2,
                        "answer_relevancy": 0.9,
                        "context_precision": 0.8,
                        "context_recall": 0.7,
                        "split": "expertwritten",
                    }
                ]
            )
        )

    monkeypatch.setattr(ragas_eval, "_run_backend_calls", fake_backend_calls)
    monkeypatch.setattr(ragas_eval, "evaluate", fake_evaluate)

    summary = ragas_eval.run_ragas_eval("http://localhost:8000", limit=None)
    assert summary["passed"] is False
    assert "thresholds" in summary
    assert summary["threshold_details"]["min_valid_rows"]["passed"] is True
    assert summary["threshold_details"]["min_faithfulness"]["passed"] is False
