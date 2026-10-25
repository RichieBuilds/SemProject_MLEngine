import unittest
import os
import tempfile
from unittest.mock import Mock, patch
from types import SimpleNamespace
from pathlib import Path

from semproject_mlengine.detector import DEFAULT_MODEL, PotholeDetector


ROOT = Path(__file__).parents[1]


class YoloIntegrationTests(unittest.TestCase):
    def test_default_model_is_trained_checkpoint(self) -> None:
        detector = PotholeDetector()

        self.assertEqual(DEFAULT_MODEL, "models/best.pt")
        self.assertEqual(detector.model_path, DEFAULT_MODEL)

    def test_missing_default_checkpoint_has_setup_instructions(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with self.assertRaisesRegex(FileNotFoundError, "docs/MODEL.md"):
                    PotholeDetector().model
            finally:
                os.chdir(previous)

    def test_custom_checkpoint_is_loaded_once_and_predictions_are_forwarded(self) -> None:
        backend = Mock()
        backend.predict.return_value = ["prediction"]
        loader = Mock(return_value=backend)
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=loader)}):
            detector = PotholeDetector("custom.pt", confidence=0.4, device="cpu")
            self.assertEqual(detector.predict("photo.jpg", save=True), ["prediction"])
            detector.predict("second.jpg")
        loader.assert_called_once_with("custom.pt")
        options = backend.predict.call_args_list[0].kwargs
        self.assertEqual(options["conf"], 0.4)
        self.assertEqual(options["device"], "cpu")
        self.assertEqual(options["project"], str(Path("runs/predict").resolve()))

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
