import unittest
from pathlib import Path

from semproject_mlengine.detector import DEFAULT_MODEL, PotholeDetector


ROOT = Path(__file__).parents[1]


class YoloIntegrationTests(unittest.TestCase):
    def test_default_model_is_yolo11_small(self) -> None:
        detector = PotholeDetector()

        self.assertEqual(DEFAULT_MODEL, "yolo11s.pt")
        self.assertEqual(detector.model_path, DEFAULT_MODEL)

    def test_confidence_must_be_normalized(self) -> None:
        with self.assertRaises(ValueError):
            PotholeDetector(confidence=1.1)

    def test_dataset_yaml_matches_repository_layout(self) -> None:
        config = (ROOT / "data/1/data.yaml").read_text()

        self.assertIn("train: train/images", config)
        self.assertIn("val: valid/images", config)
        self.assertIn("test: test/images", config)
        self.assertIn("0: pothole", config)


if __name__ == "__main__":
    unittest.main()
