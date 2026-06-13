import unittest
from unittest.mock import patch
from pathlib import Path

from catdetector_api.main import api_host, api_port, app
from catdetector_api.settings import ApiSettings, LogFormat
from fastapi.testclient import TestClient


class HealthTest(unittest.TestCase):
    def test_health_endpoint_reports_ok(self):
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class ApiServerConfigTest(unittest.TestCase):
    def test_api_server_config_defaults_to_all_interfaces_on_port_8000(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(api_host(), "0.0.0.0")
            self.assertEqual(api_port(), 8000)

    def test_api_server_config_uses_environment_overrides(self):
        with patch.dict(
            "os.environ",
            {
                "CATDETECTOR_API_HOST": "127.0.0.1",
                "CATDETECTOR_API_PORT": "8011",
            },
        ):
            self.assertEqual(api_host(), "127.0.0.1")
            self.assertEqual(api_port(), 8011)

    def test_api_settings_reads_prediction_and_logging_environment(self):
        with patch.dict(
            "os.environ",
            {
                "CATDETECTOR_CHECKPOINT": "checkpoints/manual.ckpt",
                "CATDETECTOR_CHECKPOINTS_DIR": "model-checkpoints",
                "CATDETECTOR_VICKIE_THRESHOLD": "0.37",
                "CATDETECTOR_OKA_THRESHOLD": "0.52",
                "CATDETECTOR_DEVICE": "cpu",
                "CATDETECTOR_LOG_LEVEL": "debug",
                "CATDETECTOR_LOG_FORMAT": "text",
                "CATDETECTOR_LOG_ACCESS": "false",
            },
            clear=True,
        ):
            settings = ApiSettings()

        self.assertEqual(settings.checkpoint, Path("checkpoints/manual.ckpt"))
        self.assertEqual(settings.checkpoints_dir, Path("model-checkpoints"))
        self.assertEqual(settings.vickie_threshold, 0.37)
        self.assertEqual(settings.oka_threshold, 0.52)
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertEqual(settings.log_format, LogFormat.TEXT)
        self.assertFalse(settings.log_access)


if __name__ == "__main__":
    unittest.main()
