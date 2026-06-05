# Cat Detector

A Python 3.14 and PyTorch Lightning-based image classification model for detecting Vickie and Oka in images.

Prepare the fixed local dataset from `data/{vickie,oka,both,none}`:

```powershell
uv run python prepare_dataset.py
```

Train:

```powershell
uv run python trainer.py
```
