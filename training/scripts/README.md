# Training scripts

## `train_modernbert.py`

Binary text classifier fine-tuning on JSONL produced by [`create_dataset.py`](../create_dataset.py).

- **Input:** `{"text": "...", "label": 0|1}` (0 = no issue, 1 = issue)
- **Default data:** [`../data/train.jsonl`](../data/train.jsonl), [`../data/eval.jsonl`](../data/eval.jsonl) (holdout `test.jsonl` is not used during training)
- **Default model:** `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`
- **Outputs:** `training/models/<run_name>_<timestamp>/` — model, tokenizer, checkpoints, `metrics.json`

### Local

From repo root (use a venv that has PyTorch installed, e.g. backend venv):

```bash
pip install -r training/requirements-train.txt
python training/scripts/train_modernbert.py
```

Custom paths and hyperparameters:

```bash
python training/scripts/train_modernbert.py \
  --train-file training/data/train.jsonl \
  --eval-file training/data/eval.jsonl \
  --output-dir training/models/my_run \
  --num-epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --fp16
```

### Google Colab (GPU)

```python
!pip install -q transformers datasets accelerate scikit-learn sentencepiece
# If Colab lacks a matching torch, install from pytorch.org or: !pip install -q torch
```

Upload this repo (or at least `training/scripts/train_modernbert.py` and your JSONL files), then:

```python
!python training/scripts/train_modernbert.py \
    --train-file training/data/train.jsonl \
    --eval-file training/data/eval.jsonl \
    --output-dir /content/drive/MyDrive/iBud-runs/modernbert-issue \
    --fp16
```

Mount Google Drive first if you use `/content/drive/...`.

### Metrics

Training prints validation metrics after `trainer.evaluate()`. The same numbers are saved in `metrics.json`, including:

- Accuracy, MCC
- Precision / recall / F1 (binary, macro, weighted)
- Confusion matrix counts: `tn`, `fp`, `fn`, `tp`
- `roc_auc`, `pr_auc` (when both classes appear in the eval set)

Best checkpoint is selected by **`f1_binary`** on the validation set.
