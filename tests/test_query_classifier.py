from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from backend.rag.query_classifier import (
    ClassificationResult,
    QueryClassifier,
    get_query_classifier,
)


class FakePipeline:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:
        self.calls.append({"text": text, "candidate_labels": candidate_labels})
        return self._payload


def test_classifier_issue_above_threshold():
    pipe = FakePipeline(
        {
            "labels": [
                "customer issue or complaint",
                "general information request",
            ],
            "scores": [0.9, 0.1],
        }
    )
    clf = QueryClassifier(
        threshold=0.7,
        hf_pipeline=pipe,
        candidate_labels=[
            "customer issue or complaint",
            "general information request",
        ],
    )
    res = clf.classify("My package is missing")
    assert res.is_issue is True
    assert res.confidence == 0.9
    assert res.label == "customer issue or complaint"


def test_classifier_non_issue_when_below_threshold():
    pipe = FakePipeline(
        {
            "labels": [
                "customer issue or complaint",
                "general information request",
            ],
            "scores": [0.6, 0.4],
        }
    )
    clf = QueryClassifier(
        threshold=0.7,
        hf_pipeline=pipe,
        candidate_labels=[
            "customer issue or complaint",
            "general information request",
        ],
    )
    res = clf.classify("I have a problem with my order")
    assert res.is_issue is False
    assert res.confidence == 0.6


def test_classifier_empty_text_returns_non_issue_default():
    pipe = FakePipeline(
        {
            "labels": [
                "customer issue or complaint",
                "general information request",
            ],
            "scores": [0.2, 0.8],
        }
    )
    clf = QueryClassifier(
        threshold=0.7,
        hf_pipeline=pipe,
        candidate_labels=[
            "customer issue or complaint",
            "general information request",
        ],
    )
    res = clf.classify("  ")
    assert res.is_issue is False
    assert res.confidence == 0.0


def test_get_query_classifier_is_singleton(monkeypatch: pytest.MonkeyPatch):
    import backend.rag.query_classifier as mod

    mod._QUERY_CLASSIFIER = None

    class FakeQC:
        pass

    monkeypatch.setattr(mod, "QueryClassifier", FakeQC)

    qc1 = mod.get_query_classifier()
    qc2 = mod.get_query_classifier()
    assert qc1 is qc2

