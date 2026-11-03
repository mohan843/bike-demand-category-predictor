import unittest
from pathlib import Path

from predictor import NearestCentroidClassifier, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "bike_demand.csv"


class BikeDemandTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 20)
        self.assertEqual(set(labels), {"low", "medium", "high"})

    def test_centroids_predict_expected_categories(self):
        model, score = train(DATA)
        self.assertGreaterEqual(score, 0.50)
        self.assertEqual(model.predict_one([22, 50, 9, 0]), "high")
        self.assertEqual(model.predict_one([10, 85, 7, 1]), "low")

    def test_predictions_match_input_count(self):
        rows, _ = load_csv(DATA)
        model, _ = train(DATA)
        self.assertEqual(len(model.predict(rows)), 20)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            NearestCentroidClassifier().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy(["low", "high"], ["low", "low"]), 0.5)


if __name__ == "__main__":
    unittest.main()
