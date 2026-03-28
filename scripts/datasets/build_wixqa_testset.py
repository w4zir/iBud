from __future__ import annotations

"""
Build a consolidated WixQA evaluation set for RAGAS.

This script loads the ExpertWritten and Simulated splits from the
Wix/WixQA dataset and writes a unified JSON file:

    evaluation/wixqa_testset.json

Each row has:
    - id: stable identifier within the combined set
    - split: "expertwritten" | "simulated"
    - question: user question text
    - answer: expert answer text
    - supporting_article: ground-truth context/article text (if available)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    # Optional dependency; tests monkeypatch this symbol.
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]


DATASET_NAME = "Wix/WixQA"
EXPERT_SPLIT = "wixqa_expertwritten"
SIMULATED_SPLIT = "wixqa_simulated"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "evaluation" / "wixqa_testset.json"


def _extract_row(raw: Dict[str, Any], *, split: str, idx: int) -> Dict[str, Any]:
    question = (raw.get("question") or "").strip()
    answer = (raw.get("answer") or "").strip()

    # The dataset schema may expose supporting context under different keys;
    # prefer an explicit supporting_article field when present.
    supporting_article = (
        raw.get("supporting_article")
        or raw.get("article")
        or raw.get("contents")
        or ""
    )
    if isinstance(supporting_article, dict):
        # Some variants wrap the text in a nested structure.
        supporting_article = supporting_article.get("text") or ""

    return {
        "id": f"{split}-{idx}",
        "split": split,
        "question": question,
        "answer": answer,
        "supporting_article": supporting_article,
    }


def build_wixqa_testset() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if load_dataset is None:  # pragma: no cover
        raise ImportError("build_wixqa_testset requires the 'datasets' package")

    expert = load_dataset(DATASET_NAME, EXPERT_SPLIT, split="train")
    for idx, raw in enumerate(expert):
        rows.append(_extract_row(dict(raw), split="expertwritten", idx=idx))

    simulated = load_dataset(DATASET_NAME, SIMULATED_SPLIT, split="train")
    for idx, raw in enumerate(simulated):
        rows.append(_extract_row(dict(raw), split="simulated", idx=idx))

    return rows


def main() -> None:
    rows = build_wixqa_testset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

