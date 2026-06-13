import unittest

from catdetector_api.dependencies import get_cat_predictor
from catdetector_api.main import app
from catdetector_api.predictions import (
    CatDetection,
    CatPrediction,
    CatProbabilities,
    CatThresholds,
    InvalidImageError,
    PredictorUnavailableError,
    prediction_label,
)
from fastapi.testclient import TestClient


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
        app.dependency_overrides.clear()

    def test_prediction_endpoint_uses_predictor_dependency(self):
        prediction = CatPrediction(
            label="both",
            probabilities=CatProbabilities(vickie=0.91, oka=0.88),
            detected=CatDetection(vickie=True, oka=True),
            thresholds=CatThresholds(vickie=0.5, oka=0.5),
        )
        predictor = FakePredictor(prediction)
        app.dependency_overrides[get_cat_predictor] = lambda: predictor
        client = TestClient(app)

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
        app.dependency_overrides[get_cat_predictor] = lambda: FailingPredictor(
            InvalidImageError("Invalid image upload.")
        )
        client = TestClient(app)

        response = client.post(
            "/api/predictions",
            files={"image": ("bad.txt", b"nope", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid image upload."})

    def test_prediction_endpoint_reports_unavailable_predictor(self):
        app.dependency_overrides[get_cat_predictor] = lambda: FailingPredictor(
            PredictorUnavailableError("No checkpoint found.")
        )
        client = TestClient(app)

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


if __name__ == "__main__":
    unittest.main()
