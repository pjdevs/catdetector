import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from catdetector_api.main import create_app, default_web_dist_dir
from fastapi.testclient import TestClient


class StaticFrontendTest(unittest.TestCase):
    def test_serves_index_when_frontend_build_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            dist_dir = Path(temp_dir_name)
            (dist_dir / "assets").mkdir()
            (dist_dir / "index.html").write_text(
                "<!doctype html><title>Cat Detector</title>",
                encoding="utf-8",
            )
            client = TestClient(create_app(web_dist_dir=dist_dir))

            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Cat Detector", response.text)

    def test_health_still_works_without_frontend_build(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            missing_dist_dir = Path(temp_dir_name) / "missing"
            client = TestClient(create_app(web_dist_dir=missing_dist_dir))

            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_default_web_dist_dir_prefers_environment_override(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            dist_dir = Path(temp_dir_name) / "web"

            with patch.dict("os.environ", {"CATDETECTOR_WEB_DIST": str(dist_dir)}):
                self.assertEqual(default_web_dist_dir(), dist_dir)


if __name__ == "__main__":
    unittest.main()
