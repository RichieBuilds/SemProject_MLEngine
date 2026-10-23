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

Evaluate the trained checkpoint on the test split:

```bash
uv run yolo detect val \
  model=models/best.pt \
  data=data/1/data.yaml \
  split=test
```

## Train on Google Colab

Open [`train_yollo_collab.ipynb`](train_yollo_collab.ipynb) in Google Colab.
The notebook clones the `dek_Engine` branch, verifies the dataset, trains
YOLO11s, evaluates the test split, and saves the best checkpoint to Google
Drive. Select a GPU runtime before running the training cell.
