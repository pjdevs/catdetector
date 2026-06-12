# Phone Prediction App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a phone-friendly Vite/Svelte photo predictor served by FastAPI, with FastAPI performing Vickie/Oka checkpoint inference.

**Architecture:** FastAPI remains the runtime entrypoint. API route handlers depend on a mockable `CatPredictor` protocol, while a concrete checkpoint predictor reuses `catdetector_trainer` model code. The Svelte app lives in its own workspace app and builds to static files served by FastAPI.

**Tech Stack:** Python 3.14, FastAPI, unittest, PyTorch, PIL, Vite, Svelte, TypeScript.

---

### Task 1: Backend Prediction Contract

**Files:**
- Create: `apps/catdetector_api/tests/catdetector_api_tests/test_predictions.py`
- Create: `apps/catdetector_api/src/catdetector_api/predictions.py`
- Create: `apps/catdetector_api/src/catdetector_api/dependencies.py`
- Modify: `apps/catdetector_api/src/catdetector_api/main.py`
- Modify: `apps/catdetector_api/pyproject.toml`

- [ ] Write failing tests for `POST /api/predictions`, invalid images, and label mapping.
- [ ] Run `uv run task test_api` and confirm the tests fail because the route and model contract do not exist.
- [ ] Add response dataclasses, `CatPredictor` protocol, a fake-overridable FastAPI dependency, and the predictions route.
- [ ] Run `uv run task test_api` and confirm the API tests pass.
- [ ] Commit: `test: cover prediction api contract` then `feat(api): add cat prediction inference endpoint`.

### Task 2: Concrete Checkpoint Inference

**Files:**
- Create: `apps/catdetector_api/src/catdetector_api/inference.py`
- Modify: `apps/catdetector_api/src/catdetector_api/dependencies.py`
- Modify: `apps/catdetector_api/pyproject.toml`
- Modify: `uv.lock`

- [ ] Add a checkpoint predictor that loads the latest checkpoint from `checkpoints/`, decodes uploaded bytes with PIL, applies `eval_transform()`, runs sigmoid over two logits, and maps to `vickie`, `oka`, `both`, or `none`.
- [ ] Return HTTP 400 for invalid image bytes and HTTP 503 when no checkpoint exists.
- [ ] Run `uv run task test_api`.
- [ ] Commit: `feat(api): add cat prediction inference endpoint`.

### Task 3: Vite/Svelte Web App

**Files:**
- Create: `apps/catdetector_web/package.json`
- Create: `apps/catdetector_web/index.html`
- Create: `apps/catdetector_web/tsconfig.json`
- Create: `apps/catdetector_web/vite.config.ts`
- Create: `apps/catdetector_web/src/main.ts`
- Create: `apps/catdetector_web/src/App.svelte`
- Create: `apps/catdetector_web/src/app.css`
- Modify: `.gitignore`

- [ ] Build a single Svelte view with camera file input, preview, analyze action, loading/error states, and final label plus probabilities.
- [ ] Run `npm install` or `npm ci` in `apps/catdetector_web` if dependencies are missing.
- [ ] Run `npm run build` in `apps/catdetector_web`.
- [ ] Commit: `feat(web): add phone prediction app`.

### Task 4: Serve Frontend From FastAPI

**Files:**
- Modify: `apps/catdetector_api/src/catdetector_api/main.py`
- Create or modify: `apps/catdetector_api/tests/catdetector_api_tests/test_static_frontend.py`

- [ ] Add a failing test that a temporary built `index.html` is served from `/`.
- [ ] Mount `apps/catdetector_web/dist/assets` and serve `index.html` for `/` when the build exists.
- [ ] Keep `/health` and `/api/predictions` working without a frontend build.
- [ ] Run `uv run task test_api`.
- [ ] Commit: `feat(api): serve web frontend`.

### Task 5: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Document the three apps, API route, frontend build flow, serving flow, and phone access URL.
- [ ] Run `$env:UV_CACHE_DIR = ".uv-cache"; uv run task format`.
- [ ] Run `$env:UV_CACHE_DIR = ".uv-cache"; uv run task check`.
- [ ] Commit: `docs: document apps architecture`.
