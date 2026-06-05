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
- `catdetector.py`: small inference/evaluation scratch script using a checkpoint and one image.
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
uv run python trainer.py
```

## Evaluate / Infer

For a quick single-image checkpoint prediction:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run python catdetector.py
```

The script currently has the checkpoint path and image path hardcoded, so edit those before using it on another image.

## Test / Check

After code or dataset changes:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task format
uv run task check
```

Documentation-only changes do not require these tasks unless explicitly requested.
