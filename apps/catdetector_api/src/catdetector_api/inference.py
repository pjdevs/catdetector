from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from catdetector_trainer.models import CatPresenceModel
from catdetector_trainer.trainer import eval_transform
from PIL import Image, UnidentifiedImageError

from catdetector_api.predictions import (
    CatDetection,
    CatPrediction,
    CatProbabilities,
    CatThresholds,
    InvalidImageError,
    PredictorUnavailableError,
    prediction_label,
)

DEFAULT_VICKIE_THRESHOLD = 0.5
DEFAULT_OKA_THRESHOLD = 0.5


def latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoints = sorted(
        checkpoints_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        raise PredictorUnavailableError(f"No .ckpt file found in {checkpoints_dir}")
    return checkpoints[-1]


class CheckpointCatPredictor:
    def __init__(
        self,
        *,
        checkpoint: Path | None = None,
        checkpoints_dir: Path = Path("checkpoints"),
        vickie_threshold: float = DEFAULT_VICKIE_THRESHOLD,
        oka_threshold: float = DEFAULT_OKA_THRESHOLD,
        model_loader: Callable[[Path], Any] = CatPresenceModel.load_from_checkpoint,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        device: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.checkpoints_dir = checkpoints_dir
        self.thresholds = CatThresholds(vickie=vickie_threshold, oka=oka_threshold)
        self.model_loader = model_loader
        self.transform = transform or eval_transform()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any | None = None

    def predict(self, image: bytes, filename: str | None = None) -> CatPrediction:
        pil_image = self._decode_image(image)
        model = self._load_model()
        image_tensor = torch.unsqueeze(self.transform(pil_image), 0).to(self.device)

        with torch.no_grad():
            probabilities = torch.sigmoid(model(image_tensor)).squeeze(0).cpu()

        vickie_probability = float(probabilities[0].item())
        oka_probability = float(probabilities[1].item())
        detected = CatDetection(
            vickie=vickie_probability >= self.thresholds.vickie,
            oka=oka_probability >= self.thresholds.oka,
        )

        return CatPrediction(
            label=prediction_label(vickie=detected.vickie, oka=detected.oka),
            probabilities=CatProbabilities(
                vickie=vickie_probability,
                oka=oka_probability,
            ),
            detected=detected,
            thresholds=self.thresholds,
        )

    def _decode_image(self, image: bytes) -> Image.Image:
        try:
            return Image.open(BytesIO(image)).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidImageError("Invalid image upload.") from exc

    def _load_model(self) -> Any:
        if self._model is None:
            checkpoint = self.checkpoint or latest_checkpoint(self.checkpoints_dir)
            self._model = self.model_loader(checkpoint).to(self.device)
            self._model.eval()
        return self._model
