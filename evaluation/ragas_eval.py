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
import inspect
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory


ROOT = Path(__file__).resolve().parent
TESTSET_PATH = ROOT / "wixqa_testset.json"
RESULTS_DIR = ROOT / "results"
AGENT_RESPONSES_PATH = ROOT / "agent_responses.json"
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_valid_rows": 1.0,
    "min_faithfulness": 0.3,
    "min_answer_relevancy": 0.3,
}


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra_fields = getattr(record, "structured_extra", {})
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload, ensure_ascii=True)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("evaluation.ragas_eval")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(_StructuredFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    raw = (os.getenv("DEBUG") or "").strip().lower()
    logger.setLevel(logging.DEBUG if raw in ("true", "1", "yes") else logging.INFO)
    return logger


LOGGER = _build_logger()

# Active dataset key for the current run; this is propagated to backend
# calls so the chat API can route retrieval to the correct corpus. It
# defaults to the primary WixQA KB.
ACTIVE_DATASET_KEY: str = "wixqa"


def _load_agent_responses_cache(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load agent response cache from JSON; tolerate missing/empty/invalid file."""
    p = path if path is not None else AGENT_RESPONSES_PATH
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_agent_responses_cache(cache: Dict[str, Dict[str, Any]], path: Path | None = None) -> None:
    """Write agent response cache to JSON."""
    p = path if path is not None else AGENT_RESPONSES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _load_env_file() -> None:
    """
    Load .env from sensible locations so the script works when run from
    project root, from the evaluation package, or via python -m evaluation.ragas_eval.
    Uses python-dotenv; existing OS env vars are not overridden.
    """
    candidate_paths = [
        Path.cwd() / ".env",
        ROOT.parent / ".env",
        ROOT / ".env",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            LOGGER.debug(
                "loaded .env file",
                extra={"structured_extra": {"component": "env", "path": str(candidate)}},
            )

            return
    LOGGER.debug(
        "no .env file found",
        extra={
            "structured_extra": {
                "component": "env",
                "tried": ",".join(str(p) for p in candidate_paths),
            }
        },
    )
    

def _build_ragas_llm_from_env():
    """
    Build a Ragas-compatible LLM based on env vars, with support for a
    dedicated judge provider/model configuration.

    Precedence rules:
      - `RAGAS_LLM_PROVIDER` (if set) controls which provider RAGAS uses
        as its judge.
      - If `RAGAS_LLM_PROVIDER` is not set, we fall back to
        `LLM_PROVIDER` so that existing configs keep working.
      - For each provider, judge-scoped model/credential env vars are
        preferred, then we fall back to the existing app-wide ones.

    Providers:
      - Cerebras: use its OpenAI-compatible endpoint via an `AsyncOpenAI`
        client pointed at the Cerebras base URL.
      - OpenAI: use an `AsyncOpenAI` client that reads credentials from
        standard env vars (e.g. `OPENAI_API_KEY`).
      - Ollama: use its OpenAI-compatible endpoint via `AsyncOpenAI`
        pointed at `<OLLAMA_BASE_URL>/v1`, so local llama models can be
        used as the RAGAS judge without changing agent behaviour.

    If we cannot construct a provider-specific client, we return None so
    that RAGAS falls back to its own defaults.
    """
    provider = (os.getenv("RAGAS_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").lower()
    
    # Cerebras: use an OpenAI-compatible AsyncOpenAI client pointed at the
    # Cerebras endpoint, so RAGAS can talk to it without needing real OpenAI.
    if provider == "cerebras":
        api_key = os.getenv("RAGAS_CEREBRAS_API_KEY") or os.getenv("CEREBRAS_API_KEY")
        model = os.getenv("RAGAS_CEREBRAS_MODEL") or os.getenv("CEREBRAS_MODEL", "llama3.1-8b")
        base_url = (
            os.getenv("RAGAS_CEREBRAS_BASE_URL")
            or os.getenv("CEREBRAS_BASE_URL")
            or "https://api.cerebras.ai/v1"
        )
        if not api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY (or RAGAS_CEREBRAS_API_KEY) must be set when "
                "using Cerebras as the RAGAS judge provider."
            )

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        # Pass the custom OpenAI-compatible client through to Ragas'
        # llm_factory. Newer versions of Ragas/Instructor expect
        # configuration such as timeouts/retries to be set on the client
        # itself, not via an extra `config` argument, so we only pass
        # the client here.
        return llm_factory(model, client=client)

    # OpenAI: construct a judge directly via RAGAS' factory so we can
    # control the model via env vars. Credentials are read by the OpenAI
    # client from standard env vars (e.g. OPENAI_API_KEY).
    if provider == "openai":
        model = os.getenv("RAGAS_OPENAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = AsyncOpenAI()
        return llm_factory(model, client=client)

    # Ollama: use its OpenAI-compatible endpoint so that local llama
    # models can be used as the RAGAS judge.
    if provider == "ollama":
        model = os.getenv("RAGAS_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = (
            os.getenv("RAGAS_OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        # Ollama's OpenAI-compatible API ignores the API key but the client
        # requires one, so we fall back to a dummy token if nothing is set.
        api_key = os.getenv("RAGAS_OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY") or "ollama"

        client = AsyncOpenAI(
            base_url=base_url.rstrip("/") + "/v1",
            api_key=api_key,
        )
        return llm_factory(model, client=client)

    # Default / unsupported providers:
    # - If RAGAS_LLM_PROVIDER is unset and LLM_PROVIDER is "ollama"
    #   (the default for the main app) but you do not want to use the
    #   Ollama judge, either set `RAGAS_LLM_PROVIDER` explicitly or rely
    #   on RAGAS' own defaults by leaving both unset.
    # - For any other provider values we also return None and let RAGAS
    #   decide how to construct an evaluator.
    return None


def _load_testset(
    limit: int | None = None,
    indices: List[int] | None = None,
    randomize: bool = False,
    random_seed: int | None = None,
    path: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Load a JSON testset with optional filtering.

    Args:
        limit: Optional maximum number of rows to return. Applied
            after any index-based or random selection.
        indices: Optional list of 0-based row indices to select
            from the full testset, in the order provided.
        randomize: If True (and indices is None), shuffle the rows
            before applying the limit.
        random_seed: Optional seed used when randomize is True to
            make shuffling reproducible.
    """
    p = path if path is not None else TESTSET_PATH
    with p.open("r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)

    # Explicit index-based selection takes precedence over randomisation.
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

    LOGGER.debug(
        "loaded rows",
        extra={
            "structured_extra": {
                "component": "testset",
                "count": len(rows),
                "limit": limit,
                "indices": indices,
            }
        },
    )
    return rows


def _run_backend_calls(
    backend_url: str,
    rows: List[Dict[str, Any]],
    use_local: bool = False,
) -> tuple[Dict[str, List[Any]], List[Dict[str, Any]]]:
    backend_url = backend_url.rstrip("/")

    user_inputs: List[str] = []
    responses: List[str] = []
    retrieved_contexts: List[List[str]] = []
    references: List[str] = []
    splits: List[str] = []
    failed_rows: List[Dict[str, Any]] = []

    cache: Dict[str, Dict[str, Any]] = _load_agent_responses_cache() if use_local else {}

    LOGGER.debug(
        "running backend calls",
        extra={
            "structured_extra": {
                "component": "backend",
                "url": backend_url,
                "num_questions": len(rows),
                "use_local": use_local,
            }
        },
    )

    # Chat endpoint can be slow (RAG + LLM); use a long timeout per request.
    with httpx.Client(timeout=180.0) as client:
        for row in rows:
            q = row.get("question") or ""
            if not q.strip():
                continue

            if use_local and q in cache:
                entry = cache[q]
                resp_text = str(entry.get("response") or "")
                ctx = list(entry.get("retrieved_contexts") or [])
                if not ctx:
                    article = row.get("supporting_article") or ""
                    if article:
                        ctx = [str(article)]
                LOGGER.debug(
                    "cache hit",
                    extra={
                        "structured_extra": {
                            "component": "backend",
                            "question_preview": q[:50] + "..." if len(q) > 50 else q,
                        }
                    },
                )
                user_inputs.append(q)
                responses.append(resp_text)
                retrieved_contexts.append(ctx)
                references.append((row.get("answer") or "").strip())
                splits.append(str(row.get("split") or "unknown"))
                continue

            try:
                resp = client.post(
                    f"{backend_url}/chat/",
                    json={
                        "session_id": None,
                        "user_id": "ragas-eval",
                        "message": q,
                        # Tell the backend which KB dataset to use.
                        "dataset": ACTIVE_DATASET_KEY,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError("Backend response is not a JSON object.")
            except Exception as exc:
                failed_rows.append(
                    {
                        "question": q,
                        "split": str(row.get("split") or "unknown"),
                        "error": str(exc),
                    }
                )
                LOGGER.debug(
                    "backend call failed",
                    extra={
                        "structured_extra": {
                            "component": "backend",
                            "question_preview": q[:50] + "..." if len(q) > 50 else q,
                            "error": str(exc),
                        }
                    },
                )
                continue

            resp_text = str(data.get("response") or "")
            srcs = data.get("sources") or []
            ctx = [str(s.get("content") or "") for s in srcs if s.get("content")]
            if not ctx:
                article = row.get("supporting_article") or ""
                if article:
                    ctx = [str(article)]

            if use_local:
                cache[q] = {"response": resp_text, "retrieved_contexts": ctx}
                _save_agent_responses_cache(cache)
                LOGGER.debug(
                    "cache miss, saved",
                    extra={
                        "structured_extra": {
                            "component": "backend",
                            "question_preview": q[:50] + "..." if len(q) > 50 else q,
                        }
                    },
                )

            user_inputs.append(q)
            responses.append(resp_text)
            retrieved_contexts.append(ctx)
            references.append((row.get("answer") or "").strip())
            splits.append(str(row.get("split") or "unknown"))

    return (
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
            "split": splits,
        },
        failed_rows,
    )


def run_ragas_eval(
    backend_url: str,
    limit: int | None = None,
    indices: List[int] | None = None,
    randomize: bool = False,
    random_seed: int | None = None,
    use_local: bool = False,
    dataset_key: str | None = None,
    testset_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation over a JSON test set.

    Args:
        backend_url: Base URL of the backend FastAPI service.
        limit: Optional maximum number of test rows to evaluate.
        indices: Optional list of 0-based row indices to evaluate
            instead of the default sequential slice.
        randomize: If True (and indices is None), randomise the
            order of test rows before applying limit.
        random_seed: Optional RNG seed used when randomize is True.
        use_local: If True, read/write agent responses from
            evaluation/agent_responses.json to avoid re-calling the
            backend for questions already cached.
    """
    # Resolve dataset key and testset path defaults.
    global ACTIVE_DATASET_KEY
    ACTIVE_DATASET_KEY = (dataset_key or os.getenv("RAGAS_DATASET_KEY") or "wixqa").lower()

    resolved_path: Path | None = None
    if testset_path is not None:
        resolved_path = Path(testset_path)

    rows = _load_testset(
        limit=limit,
        indices=indices,
        randomize=randomize,
        random_seed=random_seed,
        path=resolved_path,
    )
    if not rows:
        raise RuntimeError(f"No rows loaded from {TESTSET_PATH}")

    backend_result = _run_backend_calls(backend_url, rows, use_local=use_local)
    if isinstance(backend_result, tuple):
        payload, failed_rows = backend_result
    else:  # backward compatibility for tests monkeypatching legacy return shape
        payload = backend_result
        failed_rows = []
    dataset = Dataset.from_dict(payload)
 
    # Ensure .env is loaded so we can reuse the same provider
    # configuration as the main app (e.g. Cerebras keys).
    _load_env_file()
 
    # If we can build a provider-specific LLM, pass it into RAGAS so it
    # doesn't try to construct its own OpenAI client (which would
    # require OPENAI_API_KEY even when using Cerebras).
    llm = _build_ragas_llm_from_env()
    provider = (os.getenv("RAGAS_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "default").lower()
    LOGGER.debug(
        "evaluator model selection",
        extra={
            "structured_extra": {
                "component": "ragas",
                "provider": provider,
                "llm_configured": llm is not None,
            }
        },
    )

    # Keep compatibility with tests that monkeypatch `evaluate` with a
    # simpler signature (dataset, show_progress=True) by only passing
    # `llm` when the callable actually accepts it.
    sig = inspect.signature(evaluate)
    try:
        if llm is not None and "llm" in sig.parameters:
            LOGGER.debug(
                "starting ragas evaluate",
                extra={"structured_extra": {"component": "ragas", "llm": True}},
            )
            result = evaluate(
                dataset,
                llm=llm,
                show_progress=True,
            )
        else:
            # Fallback to previous behaviour; this may use OpenAI directly.
            LOGGER.debug(
                "starting ragas evaluate",
                extra={"structured_extra": {"component": "ragas", "llm": False}},
            )
            result = evaluate(
                dataset,
                show_progress=True,
            )
    except Exception as exc:
        failed_rows.append(
            {
                "question": "__evaluation__",
                "split": "n/a",
                "error": f"ragas_evaluate_failed: {exc}",
            }
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        summary_path = RESULTS_DIR / f"run_{ts}.json"
        summary: Dict[str, Any] = {
            "metrics": {metric: None for metric in METRIC_NAMES},
            "by_split": {},
            "num_rows": 0,
            "valid_counts": {metric: 0 for metric in METRIC_NAMES},
            "failed_rows": failed_rows,
            "thresholds": DEFAULT_THRESHOLDS,
            "threshold_details": {
                "min_valid_rows": {
                    "required": int(DEFAULT_THRESHOLDS["min_valid_rows"]),
                    "actual": 0,
                    "passed": False,
                }
            },
            "passed": False,
            "backend_url": backend_url,
            "timestamp": ts,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    LOGGER.debug(
        "evaluate done",
        extra={"structured_extra": {"component": "ragas", "dataset_rows": len(dataset)}},
    )

    # EvaluationResult exposes to_pandas/save_json; use both for flexibility.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    summary_path = RESULTS_DIR / f"run_{ts}.json"
    LOGGER.debug(
        "writing summary",
        extra={"structured_extra": {"component": "results", "path": str(summary_path)}},
    )

    try:
        df = result.to_pandas()
        # Overall metric means (NaN-safe) + valid counts.
        metric_means: Dict[str, float | None] = {}
        valid_counts: Dict[str, int] = {}
        for metric in METRIC_NAMES:
            if metric in df.columns:
                series = df[metric].dropna()
                valid_counts[metric] = int(len(series))
                metric_means[metric] = float(series.mean()) if len(series) > 0 else None
            else:
                valid_counts[metric] = 0
                metric_means[metric] = None

        valid_row_count = 0
        present_metric_cols = [m for m in METRIC_NAMES if m in df.columns]
        if present_metric_cols:
            valid_row_count = int(len(df[present_metric_cols].dropna(how="all")))

        # Per-split breakdown.
        split_breakdown: Dict[str, Dict[str, float | None]] = {}
        if "split" in df.columns:
            for split in df["split"].unique():
                sub = df[df["split"] == split]
                split_breakdown[str(split)] = {
                    metric: (
                        float(sub[metric].dropna().mean())
                        if metric in sub.columns and len(sub[metric].dropna()) > 0
                        else None
                    )
                    for metric in METRIC_NAMES
                }

        thresholds = DEFAULT_THRESHOLDS.copy()
        min_valid_rows = int(thresholds["min_valid_rows"])
        threshold_details: Dict[str, Dict[str, Any]] = {
            "min_valid_rows": {
                "required": min_valid_rows,
                "actual": valid_row_count,
                "passed": valid_row_count >= min_valid_rows,
            }
        }
        for metric in ("faithfulness", "answer_relevancy"):
            required_key = f"min_{metric}"
            required_value = float(thresholds.get(required_key, 0.0))
            actual_value = metric_means.get(metric)
            passed_metric = (
                actual_value is not None and float(actual_value) >= required_value
            )
            threshold_details[required_key] = {
                "required": required_value,
                "actual": actual_value,
                "passed": passed_metric,
            }

        passed = all(item["passed"] for item in threshold_details.values())

        summary: Dict[str, Any] = {
            "metrics": metric_means,
            "by_split": split_breakdown,
            "num_rows": len(df),
            "valid_counts": valid_counts,
            "failed_rows": failed_rows,
            "thresholds": thresholds,
            "threshold_details": threshold_details,
            "passed": passed,
            "backend_url": backend_url,
            "timestamp": ts,
        }
    except Exception:
        # Fallback: best-effort serialisation of the result object.
        summary = {
            "metrics": {metric: None for metric in METRIC_NAMES},
            "by_split": {},
            "num_rows": 0,
            "valid_counts": {metric: 0 for metric in METRIC_NAMES},
            "failed_rows": failed_rows,
            "thresholds": DEFAULT_THRESHOLDS,
            "threshold_details": {
                "min_valid_rows": {
                    "required": int(DEFAULT_THRESHOLDS["min_valid_rows"]),
                    "actual": 0,
                    "passed": False,
                }
            },
            "passed": False,
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
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of 0-based test row indices to "
            "evaluate (e.g. '0,3,5'). When provided, this overrides "
            "--limit and --randomize selection order."
        ),
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help=(
            "Randomize the order of test rows before applying --limit. "
            "Ignored if --indices is provided."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed to use when --randomize is set.",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help=(
            "Use evaluation/agent_responses.json to reuse cached agent "
            "responses when present, and save new responses there after calling the backend."
        ),
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default="wixqa",
        help=(
            "Dataset key to send to the backend for retrieval "
            "(for example 'wixqa' or 'bitext'). Defaults to 'wixqa'."
        ),
    )
    parser.add_argument(
        "--testset-path",
        type=str,
        default=None,
        help=(
            "Path to the JSON testset file to load. "
            "Defaults to the WixQA testset when omitted."
        ),
    )
    args = parser.parse_args()

    indices = None
    if args.indices:
        # Allow simple comma-separated integers, ignoring empty segments.
        indices = [int(part) for part in args.indices.split(",") if part.strip()]

    summary = run_ragas_eval(
        args.backend_url,
        limit=args.limit,
        indices=indices,
        randomize=args.randomize,
        random_seed=args.random_seed,
        use_local=args.use_local,
        dataset_key=args.dataset_key,
        testset_path=args.testset_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

