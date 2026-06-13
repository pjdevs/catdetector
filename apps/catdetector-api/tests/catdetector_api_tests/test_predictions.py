import unittest
from pathlib import Path
from typing import cast

from catdetector_api.dependencies import create_cat_predictor
from catdetector_api.inference import CheckpointCatPredictor
from catdetector_api.main import app
from catdetector_api.predictions import (
    CatDetection,
    CatPrediction,
    CatPredictor,
    CatProbabilities,
    CatThresholds,
    InvalidImageError,
    PredictorUnavailableError,
    prediction_label,
)
from catdetector_api.settings import ApiSettings
from fastapi.testclient import TestClient
from wireup.integration.fastapi import get_app_container


class FakePredictor:
    def __init__(self, prediction: CatPrediction):
        self.prediction = prediction
        self.images: list[tuple[bytes, str | None]] = []

    def predict(self, image: bytes, filename: str | None = None) -> CatPrediction:
        self.images.append((image, filename))
        return self.prediction


class FailingPredictor:
    def __init__(self, error: Exception):
        self.error = error

    def predict(self, image: bytes, filename: str | None = None) -> CatPrediction:
        raise self.error


class PredictionEndpointTest(unittest.TestCase):
    def tearDown(self):
        get_app_container(app).override.clear()

    def test_prediction_endpoint_uses_predictor_dependency(self):
        prediction = CatPrediction(
            label="both",
            probabilities=CatProbabilities(vickie=0.91, oka=0.88),
            detected=CatDetection(vickie=True, oka=True),
            thresholds=CatThresholds(vickie=0.5, oka=0.5),
        )
        predictor = FakePredictor(prediction)
        client = TestClient(app)

        with get_app_container(app).override.injectable(CatPredictor, predictor):
            response = client.post(
                "/api/predictions",
                files={"image": ("cats.jpg", b"fake image bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "label": "both",
                "probabilities": {"vickie": 0.91, "oka": 0.88},
                "detected": {"vickie": True, "oka": True},
                "thresholds": {"vickie": 0.5, "oka": 0.5},
            },
        )
        self.assertEqual(predictor.images, [(b"fake image bytes", "cats.jpg")])

    def test_prediction_endpoint_requires_image_upload(self):
        client = TestClient(app)

        response = client.post("/api/predictions")

        self.assertEqual(response.status_code, 422)

    def test_prediction_endpoint_rejects_invalid_images(self):
        predictor = FailingPredictor(InvalidImageError("Invalid image upload."))
        client = TestClient(app)

        with get_app_container(app).override.injectable(CatPredictor, predictor):
            response = client.post(
                "/api/predictions",
                files={"image": ("bad.txt", b"nope", "text/plain")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid image upload."})

    def test_prediction_endpoint_reports_unavailable_predictor(self):
        predictor = FailingPredictor(PredictorUnavailableError("No checkpoint found."))
        client = TestClient(app)

        with get_app_container(app).override.injectable(CatPredictor, predictor):
            response = client.post(
                "/api/predictions",
                files={"image": ("cat.jpg", b"image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "No checkpoint found."})


class PredictionLabelTest(unittest.TestCase):
    def test_prediction_label_maps_detected_cats_to_human_label(self):
        self.assertEqual(prediction_label(vickie=True, oka=True), "both")
        self.assertEqual(prediction_label(vickie=True, oka=False), "vickie")
        self.assertEqual(prediction_label(vickie=False, oka=True), "oka")
        self.assertEqual(prediction_label(vickie=False, oka=False), "none")


class CatPredictorFactoryTest(unittest.TestCase):
    def test_cat_predictor_factory_uses_relevant_settings(self):
        predictor = create_cat_predictor(
            ApiSettings(
                checkpoint=Path("custom.ckpt"),
                checkpoints_dir=Path("custom-checkpoints"),
                vickie_threshold=0.37,
                oka_threshold=0.52,
                device="cpu",
            )
        )

        self.assertIsInstance(predictor, CheckpointCatPredictor)
        checkpoint_predictor = cast(CheckpointCatPredictor, predictor)
        self.assertEqual(checkpoint_predictor.checkpoint, Path("custom.ckpt"))
        self.assertEqual(
            checkpoint_predictor.checkpoints_dir, Path("custom-checkpoints")
        )
        self.assertEqual(checkpoint_predictor.thresholds.vickie, 0.37)
        self.assertEqual(checkpoint_predictor.thresholds.oka, 0.52)
        self.assertEqual(checkpoint_predictor.device, "cpu")


if __name__ == "__main__":
    unittest.main()
