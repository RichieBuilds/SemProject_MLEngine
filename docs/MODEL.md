# Cleaned-v3 pothole model

## Get the checkpoint

Ask a teammate with access to the training Google Drive to share the checkpoint.
No public model download URL is configured in this repository.

Drive location:

```text
My Drive/RoadWatch_cleaned_training/pothole_yolo11s_cleaned_v3/weights/best.pt
```

Create `models/` in your local checkout and place that file at `models/best.pt`.
Use the cleaned-v3 run, not the earlier model trained on the original splits.
A `.pt` checkpoint can look like a ZIP internally; do not extract it. If your
browser appends `.zip` to this checkpoint, preserve its bytes and name it `best.pt`.

SHA-256 of the local checkpoint used for the demo:

```text
9dad6abd86b08e02d76a02f5e0190ddf373b39de9b8487646f01a541e8810291
```

On Linux, check with `sha256sum models/best.pt`. If it differs, confirm which
checkpoint you downloaded before comparing results. Weights are ignored by Git,
so pulling a code branch does not install or update the model on another machine.

## Inference

From the repository root:

```bash
uv sync
uv run semproject-mlengine predict path/to/image.jpg --device cpu --save
```

The default confidence threshold is 0.25. Use `--confidence 0.4` to override it,
or `--model /path/to/another.pt` for an explicit checkpoint. Pick any operational
threshold using validation data; do not tune it repeatedly on the held-out test set.

## Dataset and training record

The following record comes from the Colab output supplied during the cleaned-v3
run. The test metrics have not been independently rerun by this repository's
unit tests.

- Architecture: YOLO11s, one class, `0: pothole`; pretrained `yolo11s.pt` initialization.
- Cleaned-v3 training: 3,196 images / 8,141 boxes.
- Validation: 375 images / 1,097 boxes.
- Held-out test: 369 images / 873 boxes.
- Image size: 640; batch: 16; seed: 0; GPU: Tesla T4.
- Epoch limit: 100; early stopping patience: 20.
- Best epoch: 63; stopped after 83 epochs; reported duration: 1.168 hours.
- Ultralytics: 8.4.189; Python: 3.13.16; PyTorch: 2.11.0+cu130.

Validation: precision 77.7%, recall 66.6%, mAP50 73.9%, mAP50–95 41.5%.
Test: precision 78.97%, recall 67.95%, mAP50 77.12%, mAP50–95 48.41%.
Precision and recall are evaluator-reported values, not a guarantee at the CLI's
0.25 confidence threshold. mAP measures performance across confidence thresholds.

The previous checkpoint used the original splits and is superseded. Its scores
cannot be compared directly because the partitions differ. Do not fine-tune that
checkpoint for this experiment: it has seen images assigned to cleaned-v3 testing.

## Reproduce in Colab

Use [the training notebook](../train_yollo_collab.ipynb) with
`roadwatch_cleaned_v3.zip` in the root of My Drive. The archive contains
`roadwatch_cleaned/` with train, valid, and test image/label folders. The notebook
writes a Colab-specific YAML so local laptop paths cannot leak into training.
Checkpoints and evaluation plots are written to Drive. For another experiment,
use a new run name and reserve the test split for final evaluation.

The split manifests and original cleaning notebook document grouping and review
choices. Filename grouping and perceptual hashes are heuristics, not proof of
independent scenes. Augmented variants remain within splits, so image counts do
not represent unique road scenes.

## Evaluate locally

Obtain the cleaned-v3 dataset from the team separately. Extract it so
`data/cleaned/train`, `data/cleaned/valid`, and `data/cleaned/test` exist. Generate a
machine-specific evaluation configuration, then evaluate:

```bash
uv run python -c 'from pathlib import Path; p=Path("data/cleaned").resolve(); (p/"data.yaml").write_text("path: " + p.as_posix() + "\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames:\n  0: pothole\n")'
uv run yolo detect val model=models/best.pt data=data/cleaned/data.yaml split=test imgsz=640 device=cpu
```

Do not evaluate this model on `data/1` as a substitute for its held-out split.

## Limits and next checks

Visual inspection showed missed smaller potholes, occasional extra boxes, and
loose or overlapping boxes. A downloaded road photo produced plausible detections,
but its provenance does not establish independence from training data. The reported
178.2 ms inference on that single CPU image is not a phone or end-to-end benchmark.
More varied independent road images, including roads without potholes, are needed
before deployment. This checkpoint is not yet integrated into Android or exported
for mobile inference.
