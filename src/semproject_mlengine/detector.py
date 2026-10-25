"""Inference wrapper for the project's YOLO11 pothole detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_MODEL = "models/best.pt"


class PotholeDetector:
    """Load YOLO11 lazily and run pothole detection.

    The model is loaded only when ``predict`` is called. This keeps imports,
    tests, and package commands that do not perform inference lightweight.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        confidence: float = 0.25,
        device: str | int | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.model_path = str(model_path)
        self.confidence = confidence
        self.device = device
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Return the loaded Ultralytics model, loading it on first use."""
        if self._model is None:
            if self.model_path == DEFAULT_MODEL and not Path(self.model_path).is_file():
                raise FileNotFoundError(
                    "Trained pothole checkpoint missing: models/best.pt. "
                    "Run from the repository root and download the cleaned-v3 "
                    "checkpoint as described in docs/MODEL.md, or pass --model PATH. "
                    "Keep the checkpoint as one file; do not extract it."
                )
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def predict(
        self,
        source: str | Path,
        *,
        save: bool = False,
        project: str | Path = "runs/predict",
        name: str = "pothole",
        **options: Any,
    ) -> list[Any]:
        """Detect potholes in an image, video, directory, or camera source."""
        predict_options: dict[str, Any] = {
            "source": str(source),
            "conf": self.confidence,
            "save": save,
        }
        if self.device is not None:
            predict_options["device"] = self.device
        if save:
            predict_options.update(project=str(Path(project).resolve()), name=name)
        predict_options.update(options)
        return list(self.model.predict(**predict_options))
