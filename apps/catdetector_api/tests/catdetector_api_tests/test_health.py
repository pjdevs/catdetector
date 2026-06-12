import unittest

from catdetector_api.main import app
from fastapi.testclient import TestClient


class HealthTest(unittest.TestCase):
    def test_health_endpoint_reports_ok(self):
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
