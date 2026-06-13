import tempfile
import unittest
from io import BytesIO
from os import utime
from pathlib import Path

import torch
from catdetector_api.inference import CheckpointCatPredictor, latest_checkpoint
from catdetector_api.predictions import InvalidImageError
from PIL import Image


class FakeModel:
    def __init__(self):
        self.device = None
        self.eval_called = False

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[-2.0, 2.0]])


class CheckpointInferenceTest(unittest.TestCase):
    def test_latest_checkpoint_returns_most_recent_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            older = temp_dir / "older.ckpt"
            newer = temp_dir / "newer.ckpt"
            older.write_text("older")
            newer.write_text("newer")
            utime(older, (1, 1))
            utime(newer, (2, 2))

            self.assertEqual(latest_checkpoint(temp_dir), newer)

    def test_predict_decodes_image_and_maps_probabilities(self):
        model = FakeModel()
        loaded_checkpoints: list[Path] = []

        def load_model(checkpoint: Path) -> FakeModel:
            loaded_checkpoints.append(checkpoint)
            return model

        image_bytes = BytesIO()
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_bytes, format="PNG")

        with tempfile.TemporaryDirectory() as temp_dir_name:
            checkpoint = Path(temp_dir_name) / "cat.ckpt"
            checkpoint.write_text("checkpoint")
            predictor = CheckpointCatPredictor(
                checkpoint=checkpoint,
                model_loader=load_model,
                transform=lambda image: torch.zeros((3, 224, 224)),
                device="cpu",
            )

            prediction = predictor.predict(image_bytes.getvalue(), "cat.png")

        self.assertEqual(loaded_checkpoints, [checkpoint])
        self.assertTrue(model.eval_called)
        self.assertEqual(prediction.label, "oka")
        self.assertAlmostEqual(prediction.probabilities.vickie, 0.119203, places=6)
        self.assertAlmostEqual(prediction.probabilities.oka, 0.880797, places=6)
        self.assertFalse(prediction.detected.vickie)
        self.assertTrue(prediction.detected.oka)
        self.assertEqual(prediction.thresholds.vickie, 0.5)
        self.assertEqual(prediction.thresholds.oka, 0.5)

    def test_predict_rejects_invalid_image_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            checkpoint = Path(temp_dir_name) / "cat.ckpt"
            checkpoint.write_text("checkpoint")
            predictor = CheckpointCatPredictor(checkpoint=checkpoint)

            with self.assertRaises(InvalidImageError):
                predictor.predict(b"not an image")


if __name__ == "__main__":
    unittest.main()
