import unittest
from unittest.mock import patch

from catdetector_api.main import api_host, api_port, app
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


if __name__ == "__main__":
    unittest.main()
