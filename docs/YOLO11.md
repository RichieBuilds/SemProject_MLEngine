# YOLO11 Integration

The inference wrapper defaults to generic pretrained `yolo11s.pt`. The authenticated pothole checkpoint is `models/best.pt`; select it explicitly. The cleaned-v3 training notebook and original run artifacts are available, and validation/test evaluation has been independently reproduced on CPU. See [model notes](MODEL.md) and [verification review](../reports/audit/2026-10-27/REVIEW.md) for checksums, conditions and limitations.

## Install

From the repository root:

```bash
uv sync
```

The first prediction with `yolo11s.pt` downloads the pretrained weights if they
are not already available locally. Weights are intentionally excluded from Git.

## Predict

```bash
uv run semproject-mlengine predict path/to/image.jpg --model models/best.pt --save
```

Use a CPU explicitly when no CUDA GPU is available:

```bash
uv run semproject-mlengine predict path/to/image.jpg --model models/best.pt --device cpu --save
```

Annotated results are written under `runs/predict/pothole/` when `--save` is
provided. A custom trained checkpoint can be supplied with `--model`:

```bash
uv run semproject-mlengine predict path/to/image.jpg \
  --model models/best.pt --save
```
