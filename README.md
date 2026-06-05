# Cat Detector

A Python 3.14 and PyTorch Lightning-based image classification model for detecting Vickie and Oka in images.

The model is a multi-label classifier:

- input: one RGB image;
- output: two independent probabilities, `p_vickie` and `p_oka`;
- training labels: `[1, 0]`, `[0, 1]`, `[1, 1]`, or `[0, 0]`.

## Repository Layout

- `models.py`: PyTorch Lightning model using a pretrained EfficientNet-B0 backbone and a 2-logit classifier head.
- `datasets.py`: CSV-backed image dataset loader for fixed `train`, `val`, and `test` splits.
- `trainer.py`: training entry point, train/eval transforms, dataloaders, logger, and checkpoint callback.
- `evaluate.py`: checkpoint evaluation script with per-cat precision/recall/F1 and a 4-class confusion matrix.
- `catdetector.py`: single-image inference script using the same preprocessing as evaluation.
- `data/`: local source images grouped by human-friendly label folders.
- `dataset/`: generated local fixed split with `labels.csv`; ignored by git.
- `checkpoints/` and `logs/`: local training artifacts; ignored by git.

## Setup

Use `uv` with a local cache on this Windows workspace:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync
```

## Train

The training code expects `dataset/labels.csv` and split folders under `dataset/`.

Run:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task train
```

Baseline run with explicit settings:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python trainer.py --max-epochs 50 --patience 8 --batch-size 16 --lr 1e-4
```

Training keeps the pretrained EfficientNet-B0 backbone frozen and trains only the classifier head. Checkpoints and early stopping monitor `val_loss`.

Optional fine-tuning experiment: unfreeze only the last EfficientNet block and keep a lower learning rate on that backbone block.

On the current dataset, this was worse than the frozen-backbone baseline, so keep the frozen run as the reference until more data is added:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python trainer.py --max-epochs 50 --patience 8 --batch-size 16 --lr 1e-4 --unfreeze-last-block --backbone-lr 1e-5
```

## Evaluate / Infer

Evaluate the latest checkpoint on the test split:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task evaluate
```

Useful options:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python evaluate.py --split all
uv run python evaluate.py --checkpoint checkpoints/YOUR_CHECKPOINT.ckpt --split val
uv run python evaluate.py --find-thresholds --split all
uv run python evaluate.py --find-thresholds --split all --export-errors
```

`--export-errors` copies false-positive and false-negative images under `reports/error-audit/` for manual inspection.

For a quick single-image checkpoint prediction:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python catdetector.py data/oka/IMG_20260201_161155.jpg
```

## Test / Check

After code or dataset changes:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task format
uv run task check
```

Documentation-only changes do not require these tasks unless explicitly requested.
