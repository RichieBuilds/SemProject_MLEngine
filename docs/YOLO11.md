# YOLO11 Integration

The project uses the pretrained `YOLO11s` model as its default pothole detector.
This integration does not train a model. Training can be added later using the
same dataset configuration in `data/1/data.yaml`.

## Install

From the repository root:

```bash
uv sync
```

The first prediction with `yolo11s.pt` downloads the pretrained weights if they
are not already available locally. Weights are intentionally excluded from Git.

## Predict

```bash
uv run semproject-mlengine predict path/to/image.jpg --save
```

Use a CPU explicitly when no CUDA GPU is available:

```bash
uv run semproject-mlengine predict path/to/image.jpg --device cpu --save
```

Annotated results are written under `runs/predict/pothole/` when `--save` is
provided. A custom trained checkpoint can later be supplied with `--model`:

```bash
uv run semproject-mlengine predict path/to/image.jpg \
  --model models/pothole_yolo11s.pt --save
```
