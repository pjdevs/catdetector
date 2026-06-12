# Cat Detector

A Python 3.14 monorepo for detecting Vickie and Oka in images.

It currently contains:

- a PyTorch Lightning trainer/evaluator package;
- a minimal FastAPI package that will later serve the phone-friendly photo test
  webapp and model inference API.

The model is a multi-label classifier:

- input: one RGB image;
- output: two independent probabilities, `p_vickie` and `p_oka`;
- training labels: `[1, 0]`, `[0, 1]`, `[1, 1]`, or `[0, 0]`.

## Repository Layout

- `pyproject.toml`: root uv workspace and shared `task` commands.
- `apps/catdetector_trainer/`: uv package for training, evaluation, and checkpoint prediction.
- `apps/catdetector_trainer/src/catdetector_trainer/models.py`: PyTorch Lightning model using a pretrained EfficientNet-B0 backbone and a 2-logit classifier head.
- `apps/catdetector_trainer/src/catdetector_trainer/datasets.py`: CSV-backed image dataset loader for fixed `train`, `val`, and `test` splits.
- `apps/catdetector_trainer/src/catdetector_trainer/trainer.py`: training entry point, train/eval transforms, dataloaders, logger, and checkpoint callback.
- `apps/catdetector_trainer/src/catdetector_trainer/evaluate.py`: checkpoint evaluation script with per-cat precision/recall/F1 and a 4-class confusion matrix.
- `apps/catdetector_trainer/src/catdetector_trainer/catdetector.py`: single-image inference script using the same preprocessing as evaluation.
- `apps/catdetector_api/`: uv package for the FastAPI service. For now it exposes only `GET /health`.
- `data/`: local source images grouped by human-friendly label folders.
- `dataset/`: generated local fixed split with `labels.csv`; ignored by git.
- `checkpoints/` and `logs/`: local training artifacts; ignored by git.

## Setup

Use `uv` with a local cache on this Windows workspace:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync
```

The root workspace installs both app packages:

- `catdetector-trainer`
- `catdetector-api`

## Train

The training code expects `dataset/labels.csv` and split folders under `dataset/`.

The current fixed local dataset contains 296 images split as 200 train, 48 val,
and 48 test images. Each split includes examples for `vickie`, `oka`, `both`,
and `none`.

## Current Lessons

The last evaluated baseline before the latest 30-image dataset update used:

- checkpoint: `checkpoints/catdetector-epoch=49-val_loss=0.5306.ckpt`;
- fixed split size at the time: 264 images;
- calibrated thresholds from validation: `vickie=0.37`, `oka=0.52`.

With the default `0.50` thresholds, the model was too conservative on Vickie:
test Vickie precision was high, but recall was low. The calibrated thresholds
improved the test trade-off:

- Vickie: precision `0.700`, recall `0.636`, F1 `0.667`;
- Oka: precision `0.810`, recall `0.739`, F1 `0.773`.

The main failure mode was not generic imbalance. The error audit showed that the
hardest cases are:

- `both` images where Oka is large or central and Vickie is small, dark, blurred,
  or in the background;
- Oka-only images with black-and-white or blurry close-up patterns that trigger a
  false Vickie prediction;
- Vickie-only images with ambiguous black-and-white patches that trigger a false
  Oka prediction;
- difficult `none` scenes containing dark textiles, blankets, furniture, or
  shapes that look cat-like.

The latest dataset update intentionally added more `both` and hard edge cases.
After the next training run, compare against the baseline above and export errors
again:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run catdetector-evaluate --find-thresholds --split all --export-errors
```

For future collection, prioritize real photos over synthetic data. Generated
images may help for broad augmentation experiments, but they should not be added
to validation or test splits, and they should not replace real edge cases of
Vickie and Oka in the actual home environment.

Run:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task train
```

Baseline run with explicit settings:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run catdetector-train --max-epochs 50 --patience 8 --batch-size 16 --lr 1e-4
```

Training keeps the pretrained EfficientNet-B0 backbone frozen and trains only the classifier head. Checkpoints and early stopping monitor `val_loss`.

Optional fine-tuning experiment: unfreeze only the last EfficientNet block and keep a lower learning rate on that backbone block.

On the current dataset, this was worse than the frozen-backbone baseline, so keep the frozen run as the reference until more data is added:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run catdetector-train --max-epochs 50 --patience 8 --batch-size 16 --lr 1e-4 --unfreeze-last-block --backbone-lr 1e-5
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
uv run catdetector-evaluate --split all
uv run catdetector-evaluate --checkpoint checkpoints/YOUR_CHECKPOINT.ckpt --split val
uv run catdetector-evaluate --split all --vickie-threshold 0.45 --oka-threshold 0.44
uv run catdetector-evaluate --find-thresholds --split all
uv run catdetector-evaluate --find-thresholds --split all --export-errors
```

`--export-errors` copies false-positive and false-negative images under `reports/error-audit/` for manual inspection.

For a quick single-image checkpoint prediction:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run catdetector-predict data/oka/IMG_20260201_161155.jpg
```

## API

Run the empty FastAPI base app:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task api
```

It starts Uvicorn on `0.0.0.0:8000` and currently exposes:

```powershell
curl http://localhost:8000/health
```

## Test / Check

After code or dataset changes:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task format
uv run task check
```

`uv run task check` runs Ruff, ty, and the package-local unittest suites under:

- `apps/catdetector_trainer/tests/catdetector_trainer_tests/`
- `apps/catdetector_api/tests/catdetector_api_tests/`

Documentation-only changes do not require these tasks unless explicitly requested.
