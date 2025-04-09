import unittest
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

class BugPredictorTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_clean_code(self):
        response = self.client.post("/predict_bug", json={
            "code": "int add(int a, int b) { return a + b; }"
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)

    def test_buggy_code(self):
        response = self.client.post("/predict_bug", json={
            "code": "int divide(int a, int b) { return a / b; }"
        })
        data = response.get_json()
        self.assertIn(data["prediction"], ["buggy", "clean"])
        self.assertTrue(0.0 <= data["confidence"] <= 1.0)

    def test_missing_code(self):
        response = self.client.post("/predict_bug", json={})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()