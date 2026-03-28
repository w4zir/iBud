#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Evaluate a fine-tuned ModernBERT classifier on JSONL and print the same metrics as train_modernbert.py.

Uses the same schema and ``compute_metrics`` as ``training/scripts/train_modernbert.py`` (accuracy,
precision/recall/F1 binary/macro/weighted, MCC, confusion counts, ROC-AUC, PR-AUC).

Example (from repo root)::

    python training/scripts/eval_modernbert.py \\
        --checkpoint modernbert-large-is-issue \\
        --data-file training/data/train.jsonl

    python training/scripts/eval_modernbert.py \\
        --checkpoint modernbert-large-is-issue \\
        --data-file training/data/test.jsonl \\
        --metrics-json training/data/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Same directory as train_modernbert.py — import shared metric + data helpers.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_modernbert import (  # noqa: E402
    _load_jsonl_as_dataset,
    _trainer_tokenizer_kwargs,
    build_compute_metrics_fn,
)

_LOG = logging.getLogger(__name__)

_REPO_TRAINING = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO_TRAINING / "data" / "train.jsonl"
_DEFAULT_CHECKPOINT = Path("modernbert-large-is-issue")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ModernBERT issue classifier on JSONL (same metrics as training)."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help="Directory with saved model + tokenizer (default: modernbert-large-is-issue).",
    )
    p.add_argument(
        "--data-file",
        type=Path,
        default=_DEFAULT_DATA,
        help="JSONL with text/label rows (default: training/data/train.jsonl).",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Use fp16 (CUDA).")
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 on CUDA when available (default: on). Use --no-bf16 to disable.",
    )
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="If set, write metrics (and metadata) to this JSON file.",
    )
    return p.parse_args()


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, (float, int)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()

    ckpt = args.checkpoint.resolve()
    data_path = args.data_file.resolve()

    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    _LOG.info("Loading data: %s", data_path)
    ds = _load_jsonl_as_dataset(data_path)
    _LOG.info("Samples: %s", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt),
        trust_remote_code=True,
    )

    def tokenize_fn(batch: dict[str, list]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    eval_tok = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available())

    tmpdir = tempfile.mkdtemp(prefix="modernbert-eval-")
    try:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            data_seed=args.seed,
            report_to="none",
            fp16=use_fp16,
            bf16=use_bf16,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_tok,
            compute_metrics=build_compute_metrics_fn(),
            **_trainer_tokenizer_kwargs(tokenizer),
        )

        _LOG.info("Evaluating checkpoint=%s", ckpt)
        metrics = trainer.evaluate()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    _LOG.info("Metrics: %s", metrics)

    payload = {
        "checkpoint": str(ckpt),
        "data_file": str(data_path),
        "num_samples": len(ds),
        "config_name_or_path": getattr(model.config, "name_or_path", None),
        "metrics": _json_safe(dict(metrics)),
    }

    if args.metrics_json is not None:
        out = args.metrics_json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _LOG.info("Wrote %s", out)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
