# SemProject ML Engine

YOLO11-based pothole detection project.

## Requirements

- Python 3.13 or newer
- `uv` for environment and dependency management
- An NVIDIA GPU is recommended for training; CPU inference is supported

## Install Dependencies

From the repository root, run:

```bash
uv sync
```

This installs the dependencies declared in `pyproject.toml`, including:

- `ultralytics` for YOLO11
- `torch` and `torchvision` for model execution
- `numpy`, `pandas`, `scikit-learn`, `pillow`, `matplotlib`, and `imagehash`

If `uv` is not available, install the main packages with:

```bash
python -m pip install ultralytics imagehash matplotlib numpy pandas pillow scikit-learn
```

## Trained Model

Place the trained checkpoint at:

```text
models/best.pt
```

The `models/` directory and model files are ignored by Git because trained
weights should be stored locally or in Google Drive, not committed to the
repository.

## Run Inference

Use the trained pothole model on an image:

```bash
uv run semproject-mlengine predict path/to/image.jpg \
  --model models/best.pt \
  --save
```

For CPU inference:

```bash
uv run semproject-mlengine predict path/to/image.jpg \
  --model models/best.pt \
  --device cpu \
  --save
```

Annotated results are saved under `runs/predict/pothole/`.

## Evaluate the Model

Create the local cleaned-data YAML as described in [MODEL.md](docs/MODEL.md), then evaluate the checkpoint on the test split:

```bash
uv run yolo detect val \
  model=models/best.pt \
  data=data/cleaned/data.yaml \
  split=test imgsz=640 device=cpu
```

## Train on Google Colab

The restored [cleaned-v3 training notebook](train_yollo_collab.ipynb) uses the externally supplied cleaned-v3 archive in Drive and Ultralytics 8.4.189. Original-split training code is preserved in [the notebook archive](notebooks/archive/train_original_split.ipynb). Start training only for a new explicitly planned experiment; do not retrain to replace evidence.

## CAT 2 evidence and navigation

[YOLO11s_Base.ipynb](YOLO11s_Base.ipynb) organizes existing work, actual reproduced cleaning outputs, recovered training evidence and verified validation/test results. It is AI-assisted organization, with team-authored interpretation and member contributions still required.

See [model notes and asset checksums](docs/MODEL.md), [original training evidence](reports/original_cleaned_v3/README.md), and the [concise verification record](reports/verification_2026-10-27/README.md). Cleaned-v3 data and weights are external assets. Follow MODEL.md to create the machine-specific evaluation YAML. Four reported test metrics reproduced to two decimal places on CPU; this does not establish mobile performance.

Detailed logs, extra plots and study notes are grouped in the [optional audit folder](reports/audit/2026-10-27/README.md).
