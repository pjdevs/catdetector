# Phone Prediction App Design

## Goal

Build a small phone-friendly web app for taking or selecting a photo and getting a Vickie/Oka prediction from the trained checkpoint.

## Architecture

The repository stays as a uv workspace under `apps/`.

- `apps/catdetector_api` is the runtime entrypoint. It exposes the HTTP API, owns FastAPI route wiring, loads the predictor dependency, and serves the built frontend.
- `apps/catdetector_web` is a Vite/Svelte app. It builds static assets into `dist/`; FastAPI serves those assets for phone access.
- `apps/catdetector_trainer` remains the training/evaluation package. The API reuses its model class and deterministic `eval_transform()`.

The backend uses a light hexagonal split:

- route handlers parse HTTP inputs and shape HTTP responses;
- dependencies provide a `CatPredictor` interface to handlers;
- the concrete checkpoint predictor performs image decoding, preprocessing, model loading, and inference.

## API Contract

`POST /api/predictions` accepts multipart form data with one `image` file.

The success response contains:

- `label`: one of `vickie`, `oka`, `both`, `none`;
- `probabilities`: `{"vickie": float, "oka": float}`;
- `detected`: `{"vickie": bool, "oka": bool}`;
- `thresholds`: `{"vickie": float, "oka": float}`;

The default thresholds are the latest documented calibrated values:

- Vickie: `0.37`;
- Oka: `0.52`.

Invalid images return HTTP 400. Missing checkpoints return HTTP 503.

## Frontend

The Svelte app is a single mobile-first screen:

- file input with `accept="image/*"` and `capture="environment"` for phone camera flow;
- preview of the selected photo;
- analyze button;
- loading, error, and result states;
- clear display of the final label and per-cat probabilities.

## Testing

Backend tests use FastAPI dependency overrides with a fake predictor so route behavior is independent of PyTorch. Unit tests also cover label mapping. Frontend build is verified with the Vite build command.

## Commits

Use conventional commits by step:

1. `docs: plan phone prediction app`
2. `test: cover prediction api contract`
3. `feat(api): add cat prediction inference endpoint`
4. `feat(web): add phone prediction app`
5. `feat(api): serve web frontend`
6. `docs: document apps architecture`
