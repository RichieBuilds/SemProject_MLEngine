# RoadWatch ML Engine

YOLO11s pothole detection using the cleaned-v3 trained checkpoint.

## Setup for teammates

Run commands from this repository's root with Python 3.13+ and `uv` installed:

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

Open [the Colab notebook](train_yollo_collab.ipynb). It mounts Drive, extracts
the cleaned-v3 ZIP, checks all three splits, trains from pretrained `yolo11s.pt`,
saves checkpoints to Drive, and evaluates the held-out test split. It does not
use the original `data/1` splits. A GPU is needed for practical training.

If Colab resets, rerun setup through the dataset and GPU checks, then load the
saved checkpoint for evaluation. Do not rerun training just to test a saved model.

Create the local cleaned-data YAML as described in [MODEL.md](docs/MODEL.md),
then evaluate the checkpoint on the test split:

```bash
uv run yolo detect val \
  model=models/best.pt \
  data=data/cleaned/data.yaml \
  split=test imgsz=640 device=cpu
```

## Repository guide

- `src/semproject_mlengine/`: inference wrapper and command-line entry point.
- `tests/`: lightweight integration tests; run `uv run python -m unittest discover -s tests -v`.
- `notebooks/roadwatch_dataset_cleaning.ipynb`: split decisions and data checks.
- `docs/MODEL.md`: checkpoint distribution, recorded results, and limitations.
- `docs/DESIGN.md`: detector rationale and references.
- `reports/original_cleaned_v3/`: recovered training and evaluation evidence.
- `reports/audit/`: detailed verification logs, plots, and study notes.

Local checkpoints, cleaned images, and generated predictions are ignored by Git.
The original dataset is retained for historical reference; it is not the
training split for the cleaned-v3 model.

## CAT 2 evidence and navigation

[YOLO11s_Base.ipynb](YOLO11s_Base.ipynb) organizes existing work, reproduced
cleaning outputs, recovered training evidence, and verified validation/test
results. It is AI-assisted organization; team-authored interpretation and
member contributions are still required.

See [model notes and asset checksums](docs/MODEL.md), [original training
evidence](reports/original_cleaned_v3/README.md), and the [concise verification
record](reports/verification_2026-10-27/README.md). Four reported test metrics
were reproduced to two decimal places on CPU; this does not establish mobile
performance.
