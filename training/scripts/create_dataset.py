#!/usr/bin/env python3
"""
Combine issue/no-issue JSON exports into train/eval/test JSONL for ModernBERT finetuning.

Each input file is expected to be a JSON object with a top-level "samples" array.
Each sample should have "user_message" (str) and "is_issue" (bool).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(r"d:\ai_ws\data\ai_bot\issue_no_issue")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SEED = 42


def normalize_text_key(text: str) -> str:
    """Normalize for deduplication: strip, lowercase, collapse whitespace."""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_samples_from_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load samples from one JSON file. Returns (samples, counters)."""
    stats = {"malformed_root": 0, "missing_samples": 0}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to read JSON: {path}: {e}") from e

    if not isinstance(raw, dict):
        stats["malformed_root"] = 1
        return [], stats

    samples = raw.get("samples")
    if not isinstance(samples, list):
        stats["missing_samples"] = 1
        return [], stats

    out: list[dict[str, Any]] = []
    for obj in samples:
        if not isinstance(obj, dict):
            continue
        out.append(
            {
                **obj,
                "_source_file": path.name,
            }
        )
    return out, stats


def row_from_sample(sample: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """
    Build a normalized row. Returns (row, skip_reason).
    Row keys: text, label, id, source_file, non_issue_category (optional).
    """
    msg = sample.get("user_message")
    if msg is None or not isinstance(msg, str) or not msg.strip():
        return None, "missing_or_empty_user_message"

    flag = sample.get("is_issue")
    if not isinstance(flag, bool):
        return None, "invalid_is_issue"

    label = 1 if flag else 0
    row: dict[str, Any] = {
        "text": msg.strip(),
        "label": label,
        "id": sample.get("id"),
        "source_file": sample.get("_source_file"),
        "non_issue_category": sample.get("non_issue_category"),
    }
    return row, None


def dedupe_by_text(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep first occurrence per normalized text key."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dup = 0
    for r in rows:
        key = normalize_text_key(r["text"])
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept, dup


def stratified_train_eval_test_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified 3-way split by integer label (train / eval / test). Test gets the remainder after floor counts."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    if train_ratio + eval_ratio >= 1.0:
        raise ValueError("train_ratio + eval_ratio must be < 1 so test has a positive share")

    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(r)

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for _label in sorted(by_label.keys()):
        items = by_label[_label][:]
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            continue
        # Single sample: training only (no eval/test for this stratum)
        if n == 1:
            train.extend(items)
            continue

        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)
        n_test = n - n_train - n_eval

        # Ensure at least one training example when n >= 2 (match prior 2-way behavior)
        if n_train <= 0:
            n_train = 1
            n_test = n - n_train - n_eval
            if n_test < 0:
                n_eval = max(0, n_eval + n_test)
                n_test = 0
        elif n_train >= n:
            n_train = max(1, n - 1)
            n_eval = min(n_eval, max(0, n - n_train - 1))
            n_test = n - n_train - n_eval

        if n_test < 0:
            n_eval = max(0, n_eval + n_test)
            n_test = 0

        train.extend(items[:n_train])
        eval_rows.extend(items[n_train : n_train + n_eval])
        test.extend(items[n_train + n_eval :])

    rng.shuffle(train)
    rng.shuffle(eval_rows)
    rng.shuffle(test)
    return train, eval_rows, test


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(int(r["label"]) for r in rows)
    return {str(k): c[k] for k in sorted(c.keys())}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps({"text": r["text"], "label": int(r["label"])}, ensure_ascii=False)
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ModernBERT issue/no-issue train/eval/test JSONL.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing source *.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for train.jsonl, eval.jsonl, test.jsonl, dataset_stats.json",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Target fraction of each class for training (stratified; floor per class)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="Target fraction of each class for eval (stratified; floor per class; test gets remainder)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for shuffling and split",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"Error: no *.json files in {input_dir}", file=sys.stderr)
        return 1

    all_samples: list[dict[str, Any]] = []
    root_issues = 0
    for jf in json_files:
        samples, st = load_samples_from_file(jf)
        root_issues += st["malformed_root"] + st["missing_samples"]
        all_samples.extend(samples)

    rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    for s in all_samples:
        row, reason = row_from_sample(s)
        if row is None:
            skip_reasons[reason or "unknown"] += 1
            continue
        rows.append(row)

    rows_before_dedup = len(rows)
    rows, n_dup = dedupe_by_text(rows)

    train_rows, eval_rows, test_rows = stratified_train_eval_test_split(
        rows, args.train_ratio, args.eval_ratio, args.seed
    )
    test_ratio_effective = 1.0 - args.train_ratio - args.eval_ratio

    stats: dict[str, Any] = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "json_files": [p.name for p in json_files],
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "test_ratio_target": test_ratio_effective,
        "seed": args.seed,
        "raw_sample_objects": len(all_samples),
        "rows_valid": rows_before_dedup,
        "rows_after_dedup": len(rows),
        "duplicates_removed": n_dup,
        "skipped_rows": dict(skip_reasons),
        "skipped_total": int(sum(skip_reasons.values())),
        "malformed_or_missing_samples_key_files": root_issues,
        "class_distribution_all": label_distribution(rows),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "test_size": len(test_rows),
        "class_distribution_train": label_distribution(train_rows),
        "class_distribution_eval": label_distribution(eval_rows),
        "class_distribution_test": label_distribution(test_rows),
    }

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    test_path = output_dir / "test.jsonl"
    stats_path = output_dir / "dataset_stats.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(test_path, test_rows)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total = len(rows)
    tr, ev, te = len(train_rows), len(eval_rows), len(test_rows)
    denom = tr + ev + te
    ratio_str = (
        f"{tr / denom:.1%} / {ev / denom:.1%} / {te / denom:.1%}" if denom else "n/a"
    )

    print("Dataset build complete")
    print(f"  Input:  {input_dir} ({len(json_files)} JSON files)")
    print(f"  Rows:   {total} (after dedup; {rows_before_dedup} before, {n_dup} dupes removed)")
    print(f"  Skipped: {sum(skip_reasons.values())} {dict(skip_reasons) if skip_reasons else ''}")
    print(f"  Split:  train={tr}, eval={ev}, test={te} ({ratio_str})")
    print(f"  Train labels: {stats['class_distribution_train']}")
    print(f"  Eval labels:  {stats['class_distribution_eval']}")
    print(f"  Test labels:  {stats['class_distribution_test']}")
    print(f"  Wrote: {train_path}")
    print(f"  Wrote: {eval_path}")
    print(f"  Wrote: {test_path}")
    print(f"  Wrote: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
