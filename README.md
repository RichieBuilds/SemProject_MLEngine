# RoadWatch ML Engine

YOLO11s pothole detection using the cleaned-v3 trained checkpoint.

## Setup for teammates

Run commands from this repository's root with Python 3.13+ and uv installed:

```bash
uv sync
```

Download the cleaned-v3 **best.pt** from the team's Google Drive and place it
at `models/best.pt`. Git does not distribute model weights. Keep it as one file;
do not unzip it. See [model setup and results](docs/MODEL.md) for the Drive
location, checksum, training details, and evaluation instructions.

## Predict

The detector uses `models/best.pt` automatically:

```bash
uv run semproject-mlengine predict path/to/image.jpg --device cpu --save
```

For a different checkpoint, add `--model path/to/checkpoint.pt`. The default
checkpoint must exist; the detector does not fall back to generic weights.
Annotated images go to `runs/predict/pothole/` (later runs may receive a numbered
suffix). The printed count is the number of predicted boxes, not verified potholes.

## Train and evaluate

Open [the Colab notebook](train_yollo_collab.ipynb). It mounts Drive, extracts the
cleaned-v3 ZIP, checks all three splits, trains from pretrained `yolo11s.pt`, saves
checkpoints to Drive, and evaluates the held-out test split. It does not use the
original `data/1` splits. A GPU is needed for practical training.

If Colab resets, rerun setup through the dataset and GPU checks, then load the
saved checkpoint for evaluation. Do not rerun training just to test a saved model.

## Repository guide

- `src/semproject_mlengine/`: inference wrapper and command-line entry point.
- `tests/`: lightweight integration tests; run `uv run python -m unittest discover -s tests -v`.
- `notebooks/roadwatch_dataset_cleaning.ipynb`: split decisions and data checks.
- `docs/MODEL.md`: checkpoint distribution, recorded results, and limitations.
- `docs/DESIGN.md`: detector rationale and references.
- `reports/dataset_review/`: evidence retained from the cleaning review.

Local checkpoints, cleaned images, and generated predictions are ignored by Git.
The original dataset is already tracked for historical reference; it is not the
training split for the cleaned-v3 model.
