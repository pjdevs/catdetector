from dataclasses import dataclass
from typing import Literal, Protocol

CatLabel = Literal["vickie", "oka", "both", "none"]


@dataclass(frozen=True)
class CatProbabilities:
    vickie: float
    oka: float


@dataclass(frozen=True)
class CatDetection:
    vickie: bool
    oka: bool


@dataclass(frozen=True)
class CatThresholds:
    vickie: float
    oka: float


@dataclass(frozen=True)
class CatPrediction:
    label: CatLabel
    probabilities: CatProbabilities
    detected: CatDetection
    thresholds: CatThresholds


class InvalidImageError(Exception):
    pass


class PredictorUnavailableError(Exception):
    pass


class CatPredictor(Protocol):
    def predict(self, image: bytes, filename: str | None = None) -> CatPrediction:
        pass


def prediction_label(*, vickie: bool, oka: bool) -> CatLabel:
    if vickie and oka:
        return "both"
    if vickie:
        return "vickie"
    if oka:
        return "oka"
    return "none"
