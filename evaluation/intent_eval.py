from __future__ import annotations

"""
Intent classification evaluation against Bitext testsets.

This script:
  - Loads a Bitext-based JSON testset (built via evaluation/build_bitext_testset.py)
  - Sends each question to the /chat API with dataset="bitext"
  - Reads the model's classified intent from the chat response
  - Compares predicted intent against the ground-truth intent in the testset
  - Computes accuracy and macro-averaged precision/recall/F1
  - Persists run + per-example results into Postgres
  - Writes a JSON summary artifact under evaluation/results/
  - Can regenerate summaries from Postgres using --from-db mode
"""

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.db.models import IntentEvalPrediction, IntentEvalRun
from backend.db.postgres import async_session_factory
from backend.observability.warehouse import record_evaluation_score


ROOT = Path(__file__).resolve().parent
DEFAULT_TESTSET_PATH = ROOT / "bitext_testset_sampled.json"
RESULTS_DIR = ROOT / "results"


@dataclass
class ExampleResult:
    test_id: str
    split: str
    question: str
    expected_intent: Optional[str]
    predicted_intent: Optional[str]
    is_correct: Optional[bool]
    session_id: Optional[str]
    error: Optional[str]


def _load_testset(
    path: Path,
    limit: Optional[int] = None,
    indices: Optional[List[int]] = None,
    randomize: bool = False,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    import random

    with path.open("r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)

    if indices:
        max_idx = len(rows) - 1
        for idx in indices:
            if idx < 0 or idx > max_idx:
                raise IndexError(
                    f"Requested test row index {idx} is out of range "
                    f"for testset of size {len(rows)}."
                )
        rows = [rows[idx] for idx in indices]
    elif randomize:
        rng = random.Random(random_seed)
        rng.shuffle(rows)

    if limit is not None and limit > 0:
        rows = rows[:limit]

    return rows


def _normalise_intent(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.strip().lower()
    return value or None


def _compute_metrics(results: List[ExampleResult]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """Compute accuracy and macro-averaged precision/recall/F1."""
    valid = [r for r in results if r.error is None and r.predicted_intent is not None]
    total_examples = len(valid)
    correct_examples = sum(1 for r in valid if r.is_correct)
    failed_examples = len(results) - total_examples

    accuracy = (correct_examples / total_examples) if total_examples > 0 else None

    labels: set[str] = set()
    for r in valid:
        if r.expected_intent is not None:
            labels.add(r.expected_intent)
        if r.predicted_intent is not None:
            labels.add(r.predicted_intent)

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in valid:
        exp = r.expected_intent or "unknown"
        pred = r.predicted_intent or "unknown"
        confusion[exp][pred] += 1
        if exp == pred:
            tp[exp] += 1
        else:
            fn[exp] += 1
            fp[pred] += 1

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in sorted(labels):
        tp_l = float(tp[label])
        fp_l = float(fp[label])
        fn_l = float(fn[label])
        prec = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else None
        rec = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else None
        f1 = (
            2 * prec * rec / (prec + rec)
            if prec is not None and rec is not None and (prec + rec) > 0
            else None
        )
        if prec is not None:
            precisions.append(prec)
        if rec is not None:
            recalls.append(rec)
        if f1 is not None:
            f1s.append(f1)

    macro_precision = (sum(precisions) / len(precisions)) if precisions else None
    macro_recall = (sum(recalls) / len(recalls)) if recalls else None
    macro_f1 = (sum(f1s) / len(f1s)) if f1s else None

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "total_examples": total_examples,
        "correct_examples": correct_examples,
        "failed_examples": failed_examples,
    }

    # Convert confusion to plain dict of dicts.
    confusion_dict: Dict[str, Dict[str, int]] = {}
    for exp, preds in confusion.items():
        confusion_dict[exp] = dict(preds)

    return metrics, confusion_dict


async def _persist_run_and_predictions(
    *,
    run_meta: Dict[str, Any],
    results: List[ExampleResult],
    metrics: Dict[str, Any],
    confusion: Dict[str, Dict[str, int]],
) -> str:
    """Insert IntentEvalRun + IntentEvalPrediction rows and an evaluation_scores metadata row."""
    created_at = datetime.now(timezone.utc)

    async with async_session_factory() as db:
        run = IntentEvalRun(
            experiment_name=run_meta.get("experiment_name"),
            dataset_key=run_meta.get("dataset_key"),
            model_provider=run_meta.get("model_provider"),
            model_name=run_meta.get("model_name"),
            prompt_version=run_meta.get("prompt_version"),
            metadata_=run_meta.get("metadata"),
            accuracy=metrics.get("accuracy"),
            macro_precision=metrics.get("macro_precision"),
            macro_recall=metrics.get("macro_recall"),
            macro_f1=metrics.get("macro_f1"),
            total_examples=int(metrics.get("total_examples") or 0),
            correct_examples=int(metrics.get("correct_examples") or 0),
            failed_examples=int(metrics.get("failed_examples") or 0),
            created_at=created_at,
        )
        db.add(run)
        await db.flush()

        for r in results:
            db.add(
                IntentEvalPrediction(
                    run_id=run.id,
                    test_id=r.test_id,
                    split=r.split,
                    question=r.question,
                    expected_intent=r.expected_intent,
                    predicted_intent=r.predicted_intent,
                    is_correct=r.is_correct,
                    session_id=r.session_id,
                    error=r.error,
                )
            )

        await db.commit()

    # Also write a summary row into evaluation_scores.metadata via warehouse helper.
    summary_metadata = {
        "pipeline": "intent_eval",
        "run_id": run.id,
        "experiment_name": run_meta.get("experiment_name"),
        "dataset_key": run_meta.get("dataset_key"),
        "model_provider": run_meta.get("model_provider"),
        "model_name": run_meta.get("model_name"),
        "prompt_version": run_meta.get("prompt_version"),
        "metrics": metrics,
        "confusion": confusion,
        "created_at": created_at.isoformat(),
    }
    await record_evaluation_score(
        session_id=None,
        groundedness=None,
        hallucination=None,
        helpfulness=None,
        metadata=summary_metadata,
    )

    return str(run.id)


async def _regenerate_from_db(
    *,
    run_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Rebuild a summary from existing DB rows."""
    from sqlalchemy import and_, select

    async with async_session_factory() as db:
        if run_id:
            run_stmt = select(IntentEvalRun).where(IntentEvalRun.id == run_id)
        elif experiment_name:
            run_stmt = (
                select(IntentEvalRun)
                .where(IntentEvalRun.experiment_name == experiment_name)
                .order_by(IntentEvalRun.created_at.desc())
                .limit(1)
            )
        else:
            run_stmt = (
                select(IntentEvalRun)
                .order_by(IntentEvalRun.created_at.desc())
                .limit(1)
            )

        run_res = await db.execute(run_stmt)
        run = run_res.scalar_one_or_none()
        if run is None:
            raise RuntimeError("No matching intent_eval_runs row found for regeneration.")

        preds_stmt = select(IntentEvalPrediction).where(IntentEvalPrediction.run_id == run.id)
        preds_res = await db.execute(preds_stmt)
        rows = [
            ExampleResult(
                test_id=p.test_id or "",
                split=p.split or "",
                question=p.question or "",
                expected_intent=_normalise_intent(p.expected_intent),
                predicted_intent=_normalise_intent(p.predicted_intent),
                is_correct=p.is_correct,
                session_id=p.session_id,
                error=p.error,
            )
            for p in preds_res.scalars().all()
        ]

    metrics, confusion = _compute_metrics(rows)
    summary = {
        "run_id": str(run.id),
        "experiment_name": run.experiment_name,
        "dataset_key": run.dataset_key,
        "model_provider": run.model_provider,
        "model_name": run.model_name,
        "prompt_version": run.prompt_version,
        "metrics": metrics,
        "confusion": confusion,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }
    return summary


async def _run_intent_eval(
    *,
    backend_url: str,
    dataset_key: str,
    testset_path: Path,
    limit: Optional[int],
    indices: Optional[List[int]],
    randomize: bool,
    random_seed: Optional[int],
    experiment_name: Optional[str],
    model_provider: Optional[str],
    model_name: Optional[str],
    prompt_version: Optional[str],
    extra_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = _load_testset(
        path=testset_path,
        limit=limit,
        indices=indices,
        randomize=randomize,
        random_seed=random_seed,
    )
    if not rows:
        raise RuntimeError(f"No rows loaded from {testset_path}")

    backend_url = backend_url.rstrip("/")

    results: List[ExampleResult] = []

    with httpx.Client(timeout=180.0) as client:
        for row in rows:
            q = (row.get("question") or "").strip()
            if not q:
                continue

            expected_intent = _normalise_intent(row.get("intent"))
            test_id = str(row.get("id") or "")
            split = str(row.get("split") or "bitext")

            try:
                resp = client.post(
                    f"{backend_url}/chat/",
                    json={
                        "session_id": None,
                        "user_id": "intent-eval",
                        "message": q,
                        "dataset": dataset_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError("Backend response is not a JSON object.")
                session_id = str(data.get("session_id") or "")
                predicted_intent = _normalise_intent(data.get("intent"))
                error: Optional[str] = None
                is_correct: Optional[bool] = None
                if predicted_intent is None:
                    error = "missing_intent"
                else:
                    is_correct = predicted_intent == expected_intent
            except Exception as exc:
                session_id = None
                predicted_intent = None
                is_correct = None
                error = str(exc)

            results.append(
                ExampleResult(
                    test_id=test_id,
                    split=split,
                    question=q,
                    expected_intent=expected_intent,
                    predicted_intent=predicted_intent,
                    is_correct=is_correct,
                    session_id=session_id,
                    error=error,
                )
            )

    metrics, confusion = _compute_metrics(results)

    run_meta: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "dataset_key": dataset_key,
        "model_provider": model_provider,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "metadata": extra_metadata or {},
    }

    run_id = await _persist_run_and_predictions(
        run_meta=run_meta,
        results=results,
        metrics=metrics,
        confusion=confusion,
    )

    ts = int(time.time())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"intent_run_{ts}.json"

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "dataset_key": dataset_key,
        "model_provider": model_provider,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "metrics": metrics,
        "confusion": confusion,
        "backend_url": backend_url,
        "timestamp": ts,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run intent classification evaluation against Bitext testsets."
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Base URL of the backend FastAPI service (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default="bitext",
        help="Dataset key to send to the backend for retrieval (default: bitext).",
    )
    parser.add_argument(
        "--testset-path",
        type=str,
        default=str(DEFAULT_TESTSET_PATH),
        help="Path to the JSON testset file to load (default: Bitext sampled testset).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of test rows to evaluate.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of 0-based test row indices to "
            "evaluate (e.g. '0,3,5'). When provided, this overrides --limit "
            "and --randomize selection order."
        ),
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the order of test rows before applying --limit.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed to use when --randomize is set.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Logical experiment name to tag the run with.",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        default=None,
        help="LLM provider name for metadata (e.g. ollama, openai, cerebras).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name or identifier used during the run.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Optional prompt version string for experiment tracking.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Additional experiment metadata as a JSON object string.",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Regenerate a summary from existing DB rows instead of calling the backend.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific intent_eval_runs.id to regenerate from when using --from-db.",
    )
    parser.add_argument(
        "--from-experiment",
        type=str,
        default=None,
        help="Experiment name to regenerate the most recent run for when using --from-db.",
    )

    args = parser.parse_args()

    if args.metadata_json:
        try:
            extra_metadata = json.loads(args.metadata_json)
            if not isinstance(extra_metadata, dict):
                raise ValueError("metadata_json must be a JSON object.")
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise SystemExit(f"Failed to parse --metadata-json: {exc}") from exc
    else:
        extra_metadata = None

    if args.indices:
        indices: Optional[List[int]] = [
            int(part) for part in args.indices.split(",") if part.strip()
        ]
    else:
        indices = None

    if args.from_db:
        summary = asyncio.run(
            _regenerate_from_db(
                run_id=args.run_id,
                experiment_name=args.from_experiment,
            )
        )
    else:
        summary = asyncio.run(
            _run_intent_eval(
                backend_url=args.backend_url,
                dataset_key=args.dataset_key,
                testset_path=Path(args.testset_path),
                limit=args.limit,
                indices=indices,
                randomize=args.randomize,
                random_seed=args.random_seed,
                experiment_name=args.experiment_name,
                model_provider=args.model_provider,
                model_name=args.model_name,
                prompt_version=args.prompt_version,
                extra_metadata=extra_metadata,
            )
        )

    # Print human-readable summary to the terminal.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

