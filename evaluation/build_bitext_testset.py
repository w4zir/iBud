from __future__ import annotations

"""
Build Bitext evaluation sets for RAGAS.

This script loads the Bitext customer-support QA dataset and writes one or
two JSON files:

    evaluation/bitext_testset_full.json
    evaluation/bitext_testset_sampled.json

Each row has:
    - id: stable identifier within the combined set
    - split: "bitext_full" | "bitext_sampled"
    - question: user request text (instruction)
    - answer: assistant response text
    - supporting_article: ground-truth context text (we use the response)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
FULL_OUTPUT_PATH = Path(__file__).with_name("bitext_testset_full.json")
SAMPLED_OUTPUT_PATH = Path(__file__).with_name("bitext_testset_sampled.json")


def _extract_row(raw: Dict[str, Any], *, split: str, idx: int) -> Dict[str, Any]:
    instruction = (raw.get("instruction") or "").strip()
    response = (raw.get("response") or "").strip()

    return {
        "id": f"{split}-{idx}",
        "split": split,
        "question": instruction,
        "answer": response,
        # There is no separate article field; we treat the response as the
        # ground-truth context for evaluation purposes.
        "supporting_article": response,
        "category": raw.get("category"),
        "intent": raw.get("intent"),
        "flags": raw.get("flags"),
    }


def build_bitext_full() -> List[Dict[str, Any]]:
    ds = load_dataset(DATASET_NAME, split="train")
    rows: List[Dict[str, Any]] = []
    for idx, raw in enumerate(ds):
        rows.append(_extract_row(dict(raw), split="bitext_full", idx=idx))
    return rows


def build_bitext_sampled(max_per_intent: int = 50) -> List[Dict[str, Any]]:
    """
    Build a sampled evaluation set by taking up to `max_per_intent` examples
    from each (category, intent) bucket, to keep runs fast but representative.
    """
    ds = load_dataset(DATASET_NAME, split="train")
    buckets: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for raw in ds:
        row = dict(raw)
        category = (row.get("category") or "").strip() or "unknown"
        intent = (row.get("intent") or "").strip() or "unknown"
        buckets[(category, intent)].append(row)

    rows: List[Dict[str, Any]] = []
    idx = 0
    for (category, intent), bucket in buckets.items():
        # Deterministic order: take the first N examples per bucket.
        for raw in bucket[:max_per_intent]:
            record = dict(raw)
            record.setdefault("category", category)
            record.setdefault("intent", intent)
            rows.append(_extract_row(record, split="bitext_sampled", idx=idx))
            idx += 1

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Bitext evaluation testsets (full and/or sampled)."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "sampled", "both"],
        default="both",
        help="Which testset(s) to build. Defaults to both.",
    )
    parser.add_argument(
        "--max-per-intent",
        type=int,
        default=50,
        help="Maximum rows per (category, intent) bucket for the sampled set.",
    )
    args = parser.parse_args()

    if args.mode in ("full", "both"):
        full_rows = build_bitext_full()
        FULL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FULL_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(full_rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(full_rows)} rows to {FULL_OUTPUT_PATH}")

    if args.mode in ("sampled", "both"):
        sampled_rows = build_bitext_sampled(max_per_intent=args.max_per_intent)
        SAMPLED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SAMPLED_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(sampled_rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(sampled_rows)} rows to {SAMPLED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

