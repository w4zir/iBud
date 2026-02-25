from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation import ragas_eval


class DummyResult:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


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

    testset_path = tmp_path / "wixqa_testset.json"
    testset_path.write_text(json.dumps(rows), encoding="utf-8")

    monkeypatch.setattr(ragas_eval, "TESTSET_PATH", testset_path)

    def fake_backend_calls(backend_url: str, loaded_rows: List[Dict[str, Any]]):
        assert backend_url == "http://localhost:8000"
        assert loaded_rows == rows
        return {
            "question": [rows[0]["question"]],
            "answer": ["model answer"],
            "contexts": [["ctx"]],
            "ground_truths": [[rows[0]["answer"]]],
            "split": [rows[0]["split"]],
        }

    def fake_evaluate(dataset, metrics, show_progress=True):
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

