# Cat Detector

A Python 3.14 monorepo for detecting Vickie and Oka in images.

It currently contains:

- a shared `catdetector-model` package for the PyTorch Lightning model, checkpoint lookup, and deterministic inference preprocessing;
- a PyTorch Lightning trainer/evaluator package;
- a FastAPI package that serves the model inference API and the built web app;
- a Vite/Svelte phone-friendly web app for taking or selecting a photo.

The model is a multi-label classifier:

- input: one RGB image;
- output: two independent probabilities, `p_vickie` and `p_oka`;
- training labels: `[1, 0]`, `[0, 1]`, `[1, 1]`, or `[0, 0]`.

## Repository Layout

- `pyproject.toml`: root uv workspace for the Python apps and shared `task` commands.
- `packages/catdetector-model/`: shared uv package for the model, checkpoint lookup, and inference/evaluation preprocessing.
- `apps/catdetector-trainer/`: uv package for training, evaluation, and checkpoint prediction.
- `apps/catdetector-trainer/src/catdetector_trainer/datasets.py`: CSV-backed image dataset loader for fixed `train`, `val`, and `test` splits.
- `apps/catdetector-trainer/src/catdetector_trainer/trainer.py`: training entry point, train transforms, dataloaders, logger, and checkpoint callback.
- `apps/catdetector-trainer/src/catdetector_trainer/evaluate.py`: checkpoint evaluation script with per-cat precision/recall/F1 and a 4-class confusion matrix.
- `apps/catdetector-trainer/src/catdetector_trainer/catdetector.py`: single-image inference script using the same preprocessing as evaluation.
- `apps/catdetector-api/`: uv package for the FastAPI service.
- `apps/catdetector-api/src/catdetector_api/main.py`: FastAPI app factory, health route, router registration, and static frontend serving.
- `apps/catdetector-api/src/catdetector_api/routes.py`: HTTP route handlers, including `POST /api/predictions`.
- `apps/catdetector-api/src/catdetector_api/dependencies.py`: FastAPI dependency wiring for the mockable predictor interface.
- `apps/catdetector-api/src/catdetector_api/predictions.py`: prediction DTOs, labels, errors, and `CatPredictor` protocol.
- `apps/catdetector-api/src/catdetector_api/inference.py`: concrete checkpoint-backed predictor using `catdetector-model`.
- `apps/catdetector-web/`: Vite/Svelte phone web app. It is intentionally outside the uv workspace members because it is a Node package, but it still lives under `apps/`.
- `data/`: local source images grouped by human-friendly label folders.
- `dataset/`: generated local fixed split with `labels.csv`; ignored by git.
- `checkpoints/` and `logs/`: local training artifacts; ignored by git.

## Setup

Use `uv` with a local cache on this Windows workspace:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync
```

The root uv workspace installs the Python packages:

- `catdetector-model`
- `catdetector-trainer`
- `catdetector-api`

Install the web app dependencies from the committed npm lockfile:

```powershell
npm.cmd --prefix apps/catdetector-web ci
```

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

## Phone Web App / API

Build the Svelte app before running FastAPI:

```powershell
npm.cmd --prefix apps/catdetector-web run build
```

The build output is written to `apps/catdetector-web/dist/` and is ignored by git.
FastAPI serves that build at `/` when it exists. Set `CATDETECTOR_WEB_DIST` to
override the static frontend path, which is what the Docker image does.

Run the FastAPI app:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task api
```

It starts Granian on `0.0.0.0:8000`. From a phone on the same network, open:

```text
http://YOUR_PC_LAN_IP:8000/
```

The web app uses a normal file input with `accept="image/*"` and
`capture="environment"`, so phones can open the camera or photo picker and upload
the selected image to FastAPI.

For local web development without building the frontend first, run the API and
Vite dev server in separate terminals:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task api_dev
uv run task web_dev
```

Vite serves the web app on its dev port and proxies `/api` to
`http://localhost:8000`.

Set `CATDETECTOR_API_HOST` or `CATDETECTOR_API_PORT` to override the API bind
address or port.

The service exposes:

```powershell
curl http://localhost:8000/health
```

For inference, send a multipart upload:

```powershell
curl -F "image=@data/oka/IMG_20260201_161155.jpg" http://localhost:8000/api/predictions
```

`POST /api/predictions` returns:

- `label`: `vickie`, `oka`, `both`, or `none`;
- `probabilities`: independent sigmoid probabilities for Vickie and Oka;
- `detected`: per-cat booleans after thresholds;
- `thresholds`: the thresholds used for the prediction.

The default API thresholds are:

- Vickie: `0.50`;
- Oka: `0.50`.

The API loads the latest `.ckpt` from `checkpoints/` lazily on first prediction.
Invalid image uploads return HTTP 400. Missing checkpoints return HTTP 503.

## Test / Check

After code or dataset changes:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task format
uv run task check
npm.cmd --prefix apps/catdetector-web run format
npm.cmd --prefix apps/catdetector-web run check
npm.cmd --prefix apps/catdetector-web run test
npm.cmd --prefix apps/catdetector-web run build
```

`uv run task check` runs Ruff, ty, and the package-local unittest suites under:

- `packages/catdetector-model/tests/catdetector_model_tests/`
- `apps/catdetector-trainer/tests/catdetector_trainer_tests/`
- `apps/catdetector-api/tests/catdetector_api_tests/`

## Docker

Build the production image after a checkpoint exists under `checkpoints/`:

```powershell
uv run task docker_build
```

The Docker image builds the Svelte frontend, installs only the API package and
its model dependency, installs CUDA PyTorch wheels by default for GPU inference,
copies `checkpoints/`, and serves FastAPI plus the built frontend on port `8000`.
The API still falls back to CPU at runtime when CUDA is unavailable.

To build a smaller CPU-only fallback image:

```powershell
uv run task docker_build_cpu
```

Documentation-only changes do not require these tasks unless explicitly requested.
