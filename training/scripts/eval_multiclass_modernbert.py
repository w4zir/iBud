#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Evaluate a fine-tuned multiclass ModernBERT classifier on bitext JSONL.

Uses the same schema and ``compute_metrics`` as ``training/scripts/train_multiclass_modernbert.py``
(accuracy, macro/weighted precision/recall/F1, MCC, ROC-AUC OVR macro, PR-AUC macro).

With ``--compare-with-base``, uses or downloads ModernBERT zeroshot v2 under ``training/models``,
initializes a multiclass head from ``label2id.json``, evaluates on the same data, and reports
finetuned vs base metrics plus deltas.

Example (from repo root)::

    python training/scripts/eval_multiclass_modernbert.py \\
        --checkpoint training/models/bitext \\
        --data-file training/data/bitext/test.jsonl

    python training/scripts/eval_multiclass_modernbert.py \\
        --checkpoint training/models/bitext \\
        --metrics-json training/data/bitext/test_metrics.json

    python training/scripts/eval_multiclass_modernbert.py \\
        --checkpoint training/models/bitext \\
        --compare-with-base
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_multiclass_modernbert import (  # noqa: E402
    _load_jsonl_as_dataset,
    _trainer_tokenizer_kwargs,
    build_compute_metrics_fn,
    load_label_maps,
)

_LOG = logging.getLogger(__name__)

_REPO_TRAINING = Path(__file__).resolve().parent.parent
_BITEXT_DATA = _REPO_TRAINING / "data" / "bitext"
_DEFAULT_DATA = _BITEXT_DATA / "test.jsonl"
_DEFAULT_LABEL2ID = _BITEXT_DATA / "label2id.json"
_DEFAULT_CHECKPOINT = _REPO_TRAINING / "models" / "bitext"
_DEFAULT_BASE_MODEL_ID = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
_DEFAULT_BASE_MODEL_DIR = _REPO_TRAINING / "models" / "modernbert-base-zeroshot-v2.0"

_DELTA_SKIP_KEYS = frozenset(
    {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "epoch"}
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multiclass ModernBERT on bitext JSONL (same metrics as multiclass training)."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help=f"Directory with saved model + tokenizer (default: {_DEFAULT_CHECKPOINT}).",
    )
    p.add_argument(
        "--data-file",
        type=Path,
        default=_DEFAULT_DATA,
        help=f"JSONL with text/label rows (default: {_DEFAULT_DATA}).",
    )
    p.add_argument(
        "--label2id-file",
        type=Path,
        default=_DEFAULT_LABEL2ID,
        help=f"Path to label2id.json (default: {_DEFAULT_LABEL2ID}).",
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
    p.add_argument(
        "--compare-with-base",
        action="store_true",
        help="Also evaluate base zeroshot v2 with multiclass head and print deltas vs finetuned.",
    )
    p.add_argument(
        "--base-model-id",
        type=str,
        default=_DEFAULT_BASE_MODEL_ID,
        help=f"Hugging Face model id for base zeroshot (default: {_DEFAULT_BASE_MODEL_ID}).",
    )
    p.add_argument(
        "--base-model-dir",
        type=Path,
        default=_DEFAULT_BASE_MODEL_DIR,
        help=f"Local directory for base model (default: {_DEFAULT_BASE_MODEL_DIR}).",
    )
    return p.parse_args()


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, np.generic):
        v = x.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    try:
        fv = float(x)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    except (TypeError, ValueError):
        return str(x)


def _load_tokenizer_with_fallbacks(primary: str, fallbacks: list[str]) -> Any:
    """Load tokenizer; checkpoints may reference hub-only classes (e.g. TokenizersBackend)."""
    errors: list[tuple[str, str]] = []
    for path in [primary, *fallbacks]:
        if not path:
            continue
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except (ValueError, OSError) as e:
            errors.append((path, str(e)))
            _LOG.debug("Tokenizer load failed for %s: %s", path, e)
            continue
    msg = "; ".join(f"{p}: {err}" for p, err in errors)
    raise ValueError(f"Could not load tokenizer from {primary} or fallbacks. {msg}")


def _ensure_base_model_local(
    base_model_id: str,
    base_model_dir: Path,
    *,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> Path:
    """Use cached base under base_model_dir if present; otherwise download and save there."""
    base_model_dir = base_model_dir.resolve()
    config_path = base_model_dir / "config.json"
    if config_path.is_file():
        _LOG.info("Using cached base model at %s", base_model_dir)
        return base_model_dir

    base_model_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("Downloading base model %s -> %s", base_model_id, base_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.save_pretrained(str(base_model_dir))
    tokenizer.save_pretrained(str(base_model_dir))
    return base_model_dir


def _run_eval(
    model_source: str,
    ds: Any,
    *,
    num_labels: int,
    max_length: int,
    eval_batch_size: int,
    seed: int,
    use_fp16: bool,
    use_bf16: bool,
    base_init: bool,
    id2label: dict[int, str],
    label2id: dict[str, int],
    eval_desc: str,
    tokenizer_fallbacks: list[str],
) -> tuple[dict[str, Any], str | None]:
    """Tokenize ``ds``, run Trainer.evaluate, return (metrics dict, config name_or_path)."""
    tokenizer = _load_tokenizer_with_fallbacks(model_source, tokenizer_fallbacks)
    if base_init:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            trust_remote_code=True,
        )

    def tokenize_fn(batch: dict[str, list]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    eval_tok = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc=eval_desc,
    )

    tmpdir = tempfile.mkdtemp(prefix="modernbert-mc-eval-")
    try:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_eval_batch_size=eval_batch_size,
            seed=seed,
            data_seed=seed,
            report_to="none",
            fp16=use_fp16,
            bf16=use_bf16,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_tok,
            compute_metrics=build_compute_metrics_fn(num_labels),
            **_trainer_tokenizer_kwargs(tokenizer),
        )
        metrics = trainer.evaluate()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    name_or_path = getattr(model.config, "name_or_path", None)
    return dict(metrics), name_or_path


def _metric_deltas(
    finetuned_metrics: dict[str, Any],
    base_metrics: dict[str, Any],
) -> dict[str, float]:
    """Per-key difference finetuned - base for numeric task metrics shared by both."""
    out: dict[str, float] = {}
    for k in finetuned_metrics:
        if k in _DELTA_SKIP_KEYS:
            continue
        if k not in base_metrics:
            continue
        fv, bv = finetuned_metrics[k], base_metrics[k]
        if isinstance(fv, bool) or isinstance(bv, bool):
            continue
        if isinstance(fv, (int, float, np.floating, np.integer)) and isinstance(
            bv, (int, float, np.floating, np.integer)
        ):
            try:
                fvf, bvf = float(fv), float(bv)
                if math.isnan(fvf) or math.isnan(bvf):
                    continue
                out[str(k)] = fvf - bvf
            except (TypeError, ValueError):
                continue
    return out


def _metadata_payload(
    *,
    ckpt: Path,
    data_path: Path,
    label2id_path: Path,
    num_samples: int,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    ft_name_or_path: str | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint": str(ckpt),
        "data_file": str(data_path),
        "label2id_file": str(label2id_path),
        "num_samples": num_samples,
        "num_labels": num_labels,
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": dict(label2id),
        "config_name_or_path": ft_name_or_path,
        "metrics": _json_safe(dict(metrics)),
    }


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()

    ckpt = args.checkpoint.resolve()
    data_path = args.data_file.resolve()
    label2id_path = args.label2id_file.resolve()

    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    num_labels, id2label, label2id = load_label_maps(label2id_path)
    valid_ids = set(label2id.values())

    _LOG.info("Loading data: %s", data_path)
    ds = _load_jsonl_as_dataset(data_path, valid_ids)
    _LOG.info("Samples: %s num_labels=%s", len(ds), num_labels)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available())

    tok_fallbacks = [str(args.base_model_dir.resolve()), args.base_model_id]

    _LOG.info("Evaluating finetuned checkpoint=%s", ckpt)
    metrics, ft_name_or_path = _run_eval(
        str(ckpt),
        ds,
        num_labels=num_labels,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        base_init=False,
        id2label=id2label,
        label2id=label2id,
        eval_desc="Tokenizing finetuned",
        tokenizer_fallbacks=tok_fallbacks,
    )
    _LOG.info("Metrics: %s", metrics)

    if not args.compare_with_base:
        payload = _metadata_payload(
            ckpt=ckpt,
            data_path=data_path,
            label2id_path=label2id_path,
            num_samples=len(ds),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ft_name_or_path=ft_name_or_path,
            metrics=dict(metrics),
        )
        if args.metrics_json is not None:
            out = args.metrics_json.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _LOG.info("Wrote %s", out)
        print(json.dumps(_json_safe(dict(metrics)), indent=2))
        return 0

    base_dir = _ensure_base_model_local(
        args.base_model_id,
        args.base_model_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    _LOG.info("Evaluating base model at %s", base_dir)
    base_metrics, base_name_or_path = _run_eval(
        str(base_dir),
        ds,
        num_labels=num_labels,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        base_init=True,
        id2label=id2label,
        label2id=label2id,
        eval_desc="Tokenizing base",
        tokenizer_fallbacks=tok_fallbacks,
    )
    _LOG.info("Base metrics: %s", base_metrics)

    deltas = _metric_deltas(metrics, base_metrics)
    compare_payload = {
        "finetuned": _json_safe(dict(metrics)),
        "base": _json_safe(dict(base_metrics)),
        "deltas_finetuned_minus_base": _json_safe(deltas),
        "base_model_id": args.base_model_id,
        "base_model_dir": str(base_dir),
    }

    payload = {
        **_metadata_payload(
            ckpt=ckpt,
            data_path=data_path,
            label2id_path=label2id_path,
            num_samples=len(ds),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ft_name_or_path=ft_name_or_path,
            metrics=dict(metrics),
        ),
        "base_model_id": args.base_model_id,
        "base_model_dir": str(base_dir),
        "base_config_name_or_path": base_name_or_path,
        "base_metrics": _json_safe(dict(base_metrics)),
        "metric_deltas_finetuned_minus_base": _json_safe(deltas),
        "compare": compare_payload,
    }

    if args.metrics_json is not None:
        out = args.metrics_json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _LOG.info("Wrote %s", out)

    print(json.dumps(compare_payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
