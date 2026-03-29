#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Fine-tune a ModernBERT checkpoint for multiclass text classification on bitext JSONL.

Expected JSONL schema (one object per line):
  {"text": "<user utterance>", "label": <int>}
  Labels are integer ids matching ``training/data/bitext/label2id.json``.

Local (from repo root)::

    pip install -r training/requirements-train.txt
    python training/scripts/train_multiclass_modernbert.py

Default base model: local ``training/models/modernbert-base-zeroshot-v2.0`` (use
``ignore_mismatched_sizes=True`` to resize the classification head).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for metrics. Install with: pip install scikit-learn"
    ) from e

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

_LOG = logging.getLogger(__name__)

_REPO_TRAINING = Path(__file__).resolve().parent.parent
_BITEXT_DATA = _REPO_TRAINING / "data" / "bitext"
_DEFAULT_TRAIN = _BITEXT_DATA / "train.jsonl"
_DEFAULT_EVAL = _BITEXT_DATA / "eval.jsonl"
_DEFAULT_LABEL2ID = _BITEXT_DATA / "label2id.json"
_DEFAULT_MODEL = str(_REPO_TRAINING / "models" / "modernbert-base-zeroshot-v2.0")
_DEFAULT_OUTPUT_DIR = _REPO_TRAINING / "models" / "bitext"


def _training_args_eval_kwargs() -> dict[str, Any]:
    """HF renamed evaluation_strategy -> eval_strategy; support transformers 4.40 through current."""
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        return {"eval_strategy": "steps"}
    if "evaluation_strategy" in sig.parameters:
        return {"evaluation_strategy": "steps"}
    raise RuntimeError(
        "Unsupported transformers: TrainingArguments has neither eval_strategy nor evaluation_strategy."
    )


def _trainer_tokenizer_kwargs(tokenizer: Any) -> dict[str, Any]:
    """Trainer uses processing_class in recent versions; older releases used tokenizer."""
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def load_label_maps(path: Path) -> tuple[int, dict[int, str], dict[str, int]]:
    """Load label2id.json; return num_labels, id2label (int -> str), label2id (str -> int)."""
    if not path.is_file():
        raise FileNotFoundError(f"label2id file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected non-empty JSON object (string -> int)")
    label2id: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ValueError(f"{path}: label keys must be strings, got {type(k)}")
        label2id[k] = int(v)
    ids = sorted(label2id.values())
    expected = list(range(len(label2id)))
    if ids != expected:
        raise ValueError(
            f"{path}: label ids must be contiguous 0..{len(label2id) - 1}, got {ids[:8]}... "
            f"(len={len(ids)})"
        )
    num_labels = len(label2id)
    id2label: dict[int, str] = {int(v): k for k, v in label2id.items()}
    return num_labels, id2label, label2id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ModernBERT on bitext multiclass JSONL.")
    p.add_argument(
        "--model-name",
        type=str,
        default=_DEFAULT_MODEL,
        help="Hugging Face model id or local path (default: training/models/modernbert-base-zeroshot-v2.0).",
    )
    p.add_argument(
        "--train-file",
        type=Path,
        default=_DEFAULT_TRAIN,
        help="Path to train.jsonl.",
    )
    p.add_argument(
        "--eval-file",
        type=Path,
        default=_DEFAULT_EVAL,
        help="Path to eval JSONL.",
    )
    p.add_argument(
        "--label2id-file",
        type=Path,
        default=_DEFAULT_LABEL2ID,
        help="Path to label2id.json (string label -> int id).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Run output directory (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="modernbert-bitext-multiclass",
        help="Legacy label; use --output-dir for artifact location.",
    )
    p.add_argument("--num-epochs", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=16, help="Per-device train batch size.")
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1, help="If >0, overrides num_epochs.")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--fp16", action="store_true", help="Use fp16 (CUDA).")
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 on CUDA when available (default: on). Use --no-bf16 to disable.",
    )
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="HF TrainingArguments optim name (default: adamw_torch_fused).",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for memory (recommended on small GPUs).",
    )
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping on eval loss.",
    )
    return p.parse_args()


def _validate_jsonl_schema(
    rows: list[dict[str, Any]], path: Path, valid_label_ids: set[int]
) -> None:
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {i+1}: expected object, got {type(row)}")
        if "text" not in row or "label" not in row:
            raise ValueError(f"{path}: line {i+1}: missing 'text' or 'label'")
        if not isinstance(row["text"], str):
            raise ValueError(f"{path}: line {i+1}: 'text' must be string")
        lab = int(row["label"])
        if lab not in valid_label_ids:
            raise ValueError(
                f"{path}: line {i+1}: 'label' must be one of label2id values, got {lab!r}"
            )


def _load_jsonl_as_dataset(path: Path, valid_label_ids: set[int]) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    _validate_jsonl_schema(raw, path, valid_label_ids)
    slim = [{"text": r["text"], "label": int(r["label"])} for r in raw]
    return Dataset.from_list(slim)


def _load_datasets(
    train_path: Path, eval_path: Path, valid_label_ids: set[int]
) -> tuple[Dataset, Dataset]:
    """Load via datasets when possible; fall back to manual JSONL for Colab edge cases."""
    try:
        ds_dict = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(eval_path)},
        )
        train_ds = ds_dict["train"]
        eval_ds = ds_dict["validation"]
        train_list = train_ds.to_list()
        eval_list = eval_ds.to_list()
        _validate_jsonl_schema(train_list, train_path, valid_label_ids)
        _validate_jsonl_schema(eval_list, eval_path, valid_label_ids)
        train_ds = Dataset.from_list(
            [{"text": r["text"], "label": int(r["label"])} for r in train_list]
        )
        eval_ds = Dataset.from_list(
            [{"text": r["text"], "label": int(r["label"])} for r in eval_list]
        )
        return train_ds, eval_ds
    except Exception:
        _LOG.debug("load_dataset json failed, using manual JSONL reader", exc_info=True)
        return (
            _load_jsonl_as_dataset(train_path, valid_label_ids),
            _load_jsonl_as_dataset(eval_path, valid_label_ids),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def build_compute_metrics_fn(num_labels: int) -> Any:
    label_list = list(range(num_labels))

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        preds = np.argmax(logits, axis=-1)
        probs = _softmax(logits) if logits.shape[-1] >= 2 else None

        out: dict[str, float] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision_macro": float(
                precision_score(labels, preds, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "precision_weighted": float(
                precision_score(labels, preds, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(labels, preds, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, preds)),
        }

        uniq = np.unique(labels)
        if probs is not None and len(uniq) > 1 and logits.shape[-1] == num_labels:
            try:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(
                        labels,
                        probs,
                        multi_class="ovr",
                        average="macro",
                        labels=label_list,
                    )
                )
            except ValueError:
                out["roc_auc_ovr_macro"] = float("nan")
            try:
                # OVR macro PR-AUC via binarized labels (sklearn < 1.3 has no multi_class= on AP score).
                y_bin = label_binarize(labels, classes=label_list)
                out["pr_auc_macro"] = float(
                    average_precision_score(y_bin, probs, average="macro")
                )
            except (TypeError, ValueError):
                out["pr_auc_macro"] = float("nan")
        else:
            out["roc_auc_ovr_macro"] = float("nan")
            out["pr_auc_macro"] = float("nan")

        return out

    return compute_metrics


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()

    label2id_path = args.label2id_file.resolve()
    num_labels, id2label, label2id = load_label_maps(label2id_path)
    valid_ids = set(label2id.values())

    train_path = args.train_file.resolve()
    eval_path = args.eval_file.resolve()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("Loading data: train=%s eval=%s", train_path, eval_path)
    _LOG.info("num_labels=%s label2id=%s", num_labels, label2id_path)
    train_ds, eval_ds = _load_datasets(train_path, eval_path, valid_ids)
    _LOG.info("Train size=%s eval size=%s", len(train_ds), len(eval_ds))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tokenize_fn(batch: dict[str, list]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    train_tok = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    eval_tok = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    )

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        **_training_args_eval_kwargs(),
        eval_steps=max(1, args.eval_steps),
        save_strategy="steps",
        save_steps=max(1, args.eval_steps),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=max(1, min(100, args.eval_steps)),
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    callbacks = []
    if not args.no_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        compute_metrics=build_compute_metrics_fn(num_labels),
        callbacks=callbacks or None,
        **_trainer_tokenizer_kwargs(tokenizer),
    )

    _LOG.info("Training; artifacts -> %s", output_dir)
    train_result = trainer.train()
    _LOG.info("Train loss=%s", getattr(train_result, "training_loss", None))

    eval_metrics = trainer.evaluate()
    _LOG.info("Eval metrics: %s", eval_metrics)

    # Full confusion matrix for metrics.json (avoid nested lists in compute_metrics — breaks HF logging)
    predict_out = trainer.predict(eval_tok)
    pred_logits = np.asarray(predict_out.predictions, dtype=np.float64)
    pred_labels = np.asarray(predict_out.label_ids, dtype=np.int64)
    pred_ids = np.argmax(pred_logits, axis=-1)
    cm_full = confusion_matrix(pred_labels, pred_ids, labels=list(range(num_labels)))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

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

    metrics_path = output_dir / "metrics.json"
    train_out_metrics = getattr(train_result, "metrics", None)
    serializable: dict[str, Any] = {
        "model_name": args.model_name,
        "label2id_file": str(label2id_path),
        "num_labels": num_labels,
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "training_args": _json_safe(training_args.to_dict()),
        "train_runtime": _json_safe(train_out_metrics) if train_out_metrics else None,
        "eval_metrics": _json_safe(dict(eval_metrics)),
        "confusion_matrix": cm_full.tolist(),
    }
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    _LOG.info("Wrote %s", metrics_path)

    print(json.dumps(eval_metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
