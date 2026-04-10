from __future__ import annotations

import pytest

from backend.rag.query_classifier import (
    QueryClassifier,
    get_query_classifier,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_classifier_empty_text_returns_non_issue_default():
    clf = QueryClassifier(endpoint="http://classifier:3000/classify")
    res = clf.classify("  ")
    assert res.is_issue is False
    assert res.confidence == 0.0
    assert res.label == "no_issue"


def test_classifier_uses_bentoml_response(monkeypatch: pytest.MonkeyPatch):
    called: dict = {}

    def _fake_post(url: str, json: dict, timeout: float):
        called["url"] = url
        called["json"] = json
        called["timeout"] = timeout
        return FakeResponse(
            {
                "is_issue": True,
                "confidence": 0.92,
                "label": "issue",
            }
        )

    monkeypatch.setattr("backend.rag.query_classifier.requests.post", _fake_post)
    clf = QueryClassifier(
        endpoint="http://classifier:3000/classify",
        timeout_seconds=9,
    )
    res = clf.classify("Order never arrived")
    assert called == {
        "url": "http://classifier:3000/classify",
        "json": {"text": "Order never arrived"},
        "timeout": 9.0,
    }
    assert res.is_issue is True
    assert res.confidence == 0.92
    assert res.label == "issue"


def test_get_query_classifier_is_singleton(monkeypatch: pytest.MonkeyPatch):
    import backend.rag.query_classifier as mod

    mod._QUERY_CLASSIFIER = None

    class FakeQC(QueryClassifier):
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(mod, "QueryClassifier", FakeQC)
    qc1 = mod.get_query_classifier()
    qc2 = mod.get_query_classifier()
    assert qc1 is qc2

