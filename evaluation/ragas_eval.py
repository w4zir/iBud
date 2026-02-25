from __future__ import annotations

"""
Run RAGAS evaluation over the WixQA test set.

This script expects:
  - A running backend FastAPI service exposing POST /chat
  - A pre-built evaluation/wixqa_testset.json file

For each question, it:
  - Calls POST {backend_url}/chat with the question
  - Collects the model answer and retrieved contexts
  - Scores with RAGAS metrics (faithfulness, answer_relevancy,
    context_precision, context_recall)
  - Writes a JSON result file under evaluation/results/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


ROOT = Path(__file__).resolve().parent
TESTSET_PATH = ROOT / "wixqa_testset.json"
RESULTS_DIR = ROOT / "results"


def _load_testset(limit: int | None = None) -> List[Dict[str, Any]]:
    with TESTSET_PATH.open("r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)
    if limit is not None and limit > 0:
        return rows[:limit]
    return rows


def _run_backend_calls(
    backend_url: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    backend_url = backend_url.rstrip("/")

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[List[str]] = []
    splits: List[str] = []

    with httpx.Client(timeout=30.0) as client:
        for row in rows:
            q = row.get("question") or ""
            if not q.strip():
                continue

            resp = client.post(
                f"{backend_url}/chat/",
                json={
                    "session_id": None,
                    "user_id": "ragas-eval",
                    "message": q,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            questions.append(q)
            answers.append(str(data.get("response") or ""))

            srcs = data.get("sources") or []
            ctx = [str(s.get("content") or "") for s in srcs if s.get("content")]
            if not ctx:
                # Fallback to ground-truth article if no retrieval context is present.
                article = row.get("supporting_article") or ""
                if article:
                    ctx = [str(article)]
            contexts.append(ctx)

            gt_answer = (row.get("answer") or "").strip()
            ground_truths.append([gt_answer] if gt_answer else [])
            splits.append(str(row.get("split") or "unknown"))

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "split": splits,
    }


def run_ragas_eval(backend_url: str, limit: int | None = None) -> Dict[str, Any]:
    rows = _load_testset(limit=limit)
    if not rows:
        raise RuntimeError(f"No rows loaded from {TESTSET_PATH}")

    payload = _run_backend_calls(backend_url, rows)
    dataset = Dataset.from_dict(payload)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        show_progress=True,
    )

    # EvaluationResult exposes to_pandas/save_json; use both for flexibility.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    summary_path = RESULTS_DIR / f"run_{ts}.json"

    try:
        df = result.to_pandas()
        # Overall metric means.
        metric_means: Dict[str, float] = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in df.columns:
                metric_means[metric] = float(df[metric].mean())

        # Per-split breakdown.
        split_breakdown: Dict[str, Dict[str, float]] = {}
        if "split" in df.columns:
            for split in df["split"].unique():
                sub = df[df["split"] == split]
                split_breakdown[str(split)] = {
                    metric: float(sub[metric].mean())
                    for metric in metric_means.keys()
                    if metric in sub.columns
                }

        summary: Dict[str, Any] = {
            "metrics": metric_means,
            "by_split": split_breakdown,
            "num_rows": len(df),
            "backend_url": backend_url,
            "timestamp": ts,
        }
    except Exception:
        # Fallback: best-effort serialisation of the result object.
        summary = {
            "raw_result": str(result),
            "backend_url": backend_url,
            "timestamp": ts,
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on WixQA testset.")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Base URL of the backend FastAPI service (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of test rows to evaluate.",
    )
    args = parser.parse_args()

    summary = run_ragas_eval(args.backend_url, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

