from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..config import get_classifier_model, get_classifier_threshold


@dataclass(frozen=True)
class ClassificationResult:
    is_issue: bool
    confidence: float
    label: str


class QueryClassifier:
    """
    ModernBERT zero-shot classifier for coarse intent routing:
    - "issue" (customer complaint/problem) vs
    - "non-issue" (general information request)
    """

    DEFAULT_LABELS: Sequence[str] = (
        "customer issue or complaint",
        "general information request",
    )

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
        candidate_labels: Optional[Sequence[str]] = None,
        hf_pipeline: Any = None,
    ) -> None:
        self._model_name = model_name or get_classifier_model()
        self._threshold = (
            float(threshold) if threshold is not None else get_classifier_threshold()
        )
        self._candidate_labels = list(candidate_labels or self.DEFAULT_LABELS)
        self._issue_label = self._candidate_labels[0]

        # Allow dependency injection for tests (avoid loading large models).
        self._pipeline = hf_pipeline
        if self._pipeline is None:
            from transformers import pipeline  # type: ignore[import]

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self._model_name,
            )

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                is_issue=False, confidence=0.0, label=self._candidate_labels[1]
            )

        raw = self._pipeline(
            text,
            candidate_labels=self._candidate_labels,
        )

        # Transformers output is typically either:
        # - dict with keys {"labels": [...], "scores": [...]}
        # - or list[dict] where the first element contains those keys.
        payload: Dict[str, Any]
        if isinstance(raw, list):
            payload = raw[0] if raw else {}
        else:
            payload = raw or {}

        labels: List[str] = list(payload.get("labels") or [])
        scores: List[float] = [float(s) for s in payload.get("scores") or []]

        if not labels or not scores:
            return ClassificationResult(
                is_issue=False, confidence=0.0, label=self._candidate_labels[1]
            )

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_label = labels[best_idx]
        best_score = scores[best_idx]

        if best_score < self._threshold:
            return ClassificationResult(
                is_issue=False,
                confidence=float(best_score),
                label=best_label,
            )

        return ClassificationResult(
            is_issue=best_label == self._issue_label,
            confidence=float(best_score),
            label=best_label,
        )


_QUERY_CLASSIFIER: Optional[QueryClassifier] = None


def get_query_classifier() -> QueryClassifier:
    global _QUERY_CLASSIFIER
    if _QUERY_CLASSIFIER is None:
        _QUERY_CLASSIFIER = QueryClassifier()
    return _QUERY_CLASSIFIER


__all__ = ["ClassificationResult", "QueryClassifier", "get_query_classifier"]

