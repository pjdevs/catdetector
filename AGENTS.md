# AGENTS.md

## Collaboration Rules

- Keep changes surgical and tightly scoped to the user's request.
- Do not refactor, reorganize, rename modules, or change architecture unless the user explicitly asks for it.
- Ask before any refactor, even if it looks obviously beneficial.
- Do not write or modify code by default. If the user asks for an audit, explanation, plan, or discussion, answer without editing files.
- When code changes are requested, avoid unrelated cleanup and preserve existing user changes in the working tree.
- Never revert changes you did not make unless the user explicitly asks.
- Keep the documentation synced with code and dataset changes. In this repo, that means updating `README.md` quickly after meaningful changes.

## Project Context

This project is a PyTorch Lightning multi-label image classifier for detecting the presence of two cats, Vickie and Oka.

The intended modeling approach is:

- Input: one RGB image.
- Output: two independent logits, `[vickie_logit, oka_logit]`.
- Labels are multi-label vectors:
  - `[1, 0]`: Vickie only
  - `[0, 1]`: Oka only
  - `[1, 1]`: both cats
  - `[0, 0]`: neither cat
- Use sigmoid at inference time.
- Use `BCEWithLogitsLoss` / binary cross entropy with logits for training.
- Do not replace this with softmax; the cats are not mutually exclusive.

## Dataset Lessons

- The human-friendly source layout can stay as folders by label.
- A fixed `train` / `val` / `test` split is preferable to random splitting during training.
- Avoid using the same random augmentation pipeline for validation or test.
- Validation and test preprocessing should be deterministic and match inference preprocessing.
- For this small dataset, metrics can lie easily if near-duplicate photo bursts are split across train and validation.
- The session-based split logic is a one-time dataset preparation hack, not core training logic.

## Python And Tooling

Use `uv` for the environment.

Start from:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync
```

After code or dataset changes, run:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run task format
uv run task check
```

For documentation-only changes, do not run the verification tasks unless the user asks.

After code or dataset changes, also update `README.md` so it still says what is in the repository, how to train, how to evaluate, and how to test/check the project.

The `task` commands are provided by taskipy in `pyproject.toml`.

Important `uv` pitfall on this Windows workspace:

- Always set `UV_CACHE_DIR` to the local repo cache before using `uv`.
- Do not rely on the global uv cache under `AppData`.
- If `uv` fails with cache initialization, access denied, or interpreter trampoline errors, retry with the local cache first.
- Prefer `.uv-cache` in the repository root for this project.

Do not reintroduce `requirements.txt` unless the user asks. `pyproject.toml` and `uv.lock` are the source of truth for dependencies.

## Training Notes

- Keep training transforms and eval transforms separate.
- Monitor validation loss for checkpoints, not training loss.
- Keep inference preprocessing aligned with eval preprocessing.
- Prefer small, explicit scripts and modules over broad framework reshuffles.
