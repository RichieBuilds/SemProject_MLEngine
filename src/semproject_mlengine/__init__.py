"""YOLO11-based pothole detection tools."""

from __future__ import annotations

import argparse

from .detector import DEFAULT_MODEL, PotholeDetector


def main() -> None:
    """Run the inference command without training the model."""
    parser = argparse.ArgumentParser(prog="semproject-mlengine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Detect potholes in an image or video")
    predict.add_argument("source", help="Path to an image, video, or directory")
    predict.add_argument("--model", default=DEFAULT_MODEL, help="YOLO11 weights path")
    predict.add_argument("--confidence", type=float, default=0.25)
    predict.add_argument("--device", default=None, help="CUDA device such as 0, or cpu")
    predict.add_argument("--save", action="store_true", help="Save annotated predictions")
    predict.add_argument("--project", default="runs/predict")
    predict.add_argument("--name", default="pothole")

    args = parser.parse_args()
    if args.command == "predict":
        detector = PotholeDetector(
            model_path=args.model,
            confidence=args.confidence,
            device=args.device,
        )
        results = detector.predict(
            args.source,
            save=args.save,
            project=args.project,
            name=args.name,
        )
        for result in results:
            count = 0 if result.boxes is None else len(result.boxes)
            print(f"{result.path}: {count} pothole(s)")
