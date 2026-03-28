from __future__ import annotations

from typing import Any, Dict, List

from backend import __init__  # ensure package import works
from scripts.datasets import build_bitext_testset


class DummyDataset:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def test_build_bitext_full_shapes_rows(monkeypatch):
    raw_rows = [
        {
            "flags": "B",
            "instruction": "How do I cancel my order?",
            "category": "ORDER",
            "intent": "cancel_order",
            "response": "You can cancel from your account page.",
        },
        {
            "flags": "BI",
            "instruction": "Can I change my shipping address?",
            "category": "SHIPPING_ADDRESS",
            "intent": "change_shipping_address",
            "response": "Yes, before the order ships you can update it.",
        },
    ]

    def fake_load_dataset(name: str, split: str):
        assert name == build_bitext_testset.DATASET_NAME
        assert split == "train"
        return DummyDataset(raw_rows)

    monkeypatch.setattr(
        build_bitext_testset, "load_dataset", fake_load_dataset  # type: ignore[arg-type]
    )

    rows = build_bitext_testset.build_bitext_full()
    assert len(rows) == 2
    for idx, row in enumerate(rows):
        assert row["id"].startswith("bitext_full-")
        assert row["question"] == raw_rows[idx]["instruction"]
        assert row["answer"] == raw_rows[idx]["response"]
        assert row["supporting_article"] == raw_rows[idx]["response"]
        assert row["category"] == raw_rows[idx]["category"]
        assert row["intent"] == raw_rows[idx]["intent"]


def test_build_bitext_sampled_respects_max_per_intent(monkeypatch):
    # Create multiple rows across two (category, intent) buckets.
    rows: List[Dict[str, Any]] = []
    for i in range(10):
        rows.append(
            {
                "flags": "B",
                "instruction": f"Q-orders-{i}",
                "category": "ORDER",
                "intent": "cancel_order",
                "response": f"R-orders-{i}",
            }
        )
    for i in range(10):
        rows.append(
            {
                "flags": "B",
                "instruction": f"Q-account-{i}",
                "category": "ACCOUNT",
                "intent": "delete_account",
                "response": f"R-account-{i}",
            }
        )

    def fake_load_dataset(name: str, split: str):
        assert name == build_bitext_testset.DATASET_NAME
        assert split == "train"
        return DummyDataset(rows)

    monkeypatch.setattr(
        build_bitext_testset, "load_dataset", fake_load_dataset  # type: ignore[arg-type]
    )

    sampled = build_bitext_testset.build_bitext_sampled(max_per_intent=3)
    # We expect at most 3 rows from each of the two buckets.
    assert len(sampled) <= 6
    intents = {r["intent"] for r in sampled}
    assert intents == {"cancel_order", "delete_account"}

