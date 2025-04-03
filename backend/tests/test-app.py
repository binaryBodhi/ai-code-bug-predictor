import unittest
import json
from app import app

class BugPredictorTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_bug_clean_code(self):
        code = "int add(int a, int b) { return a + b; }"
        response = self.client.post(
            "/predict_bug",
            data=json.dumps({"code": code}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], ["clean", "buggy"])
        self.assertIsInstance(data["confidence"], float)

    def test_predict_bug_buggy_code(self):
        code = "int divide(int a, int b) { return a / b; }"
        response = self.client.post(
            "/predict_bug",
            data=json.dumps({"code": code}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn(data["prediction"], ["clean", "buggy"])

    def test_missing_code_field(self):
        response = self.client.post(
            "/predict_bug",
            data=json.dumps({}),  # missing "code"
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
