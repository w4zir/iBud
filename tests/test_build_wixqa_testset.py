from __future__ import annotations

from typing import Any, Dict, List

from backend import __init__  # ensure package import works
from evaluation import build_wixqa_testset


class DummyDataset:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def test_build_wixqa_testset_shapes_rows(monkeypatch):
    expert_rows = [
        {"question": "Q1", "answer": "A1", "supporting_article": "S1"},
        {"question": "Q2", "answer": "A2", "supporting_article": "S2"},
    ]
    simulated_rows = [
        {"question": "Q3", "answer": "A3", "supporting_article": "S3"},
    ]

    def fake_load_dataset(name: str, split_name: str, split: str):
        assert name == build_wixqa_testset.DATASET_NAME
        if split_name == build_wixqa_testset.EXPERT_SPLIT:
            return DummyDataset(expert_rows)
        if split_name == build_wixqa_testset.SIMULATED_SPLIT:
            return DummyDataset(simulated_rows)
        raise AssertionError(f"unexpected split {split_name}")

    monkeypatch.setattr(
        build_wixqa_testset, "load_dataset", fake_load_dataset  # type: ignore[arg-type]
    )

    rows = build_wixqa_testset.build_wixqa_testset()
    assert len(rows) == 3
    splits = {r["id"].split("-")[0] for r in rows}
    assert splits == {"expertwritten", "simulated"}
    for row in rows:
        assert "question" in row and "answer" in row and "supporting_article" in row

