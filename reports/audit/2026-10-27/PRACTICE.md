# Assessment practice — AI-assisted study notes

These are explanations and rehearsal prompts, not team-authored assessed analysis. Use the supplied code and outputs to check every answer. No member contributions have been inferred.

## Follow one example through the pipeline

A YOLO text line is `class x_center y_center width height`. Class 0 means pothole here. Coordinates are fractions of the image dimensions; multiply x and width by image width, and y and height by image height to recover pixels. Corners are `(x ± width/2, y ± height/2)`. One image can have several label lines. The CSV exploration pads extra object columns with NaN; padding is different from a missing image label. The cleaning notebook checks these cases separately.

The inference wrapper lazily loads a checkpoint and calls Ultralytics prediction. Its default `yolo11s.pt` is a generic pretrained checkpoint; select `models/best.pt` explicitly for the authenticated pothole model. Confidence filtering controls which predictions are returned. It does not establish correctness against a label.

## Explain this YOLO11s

The loaded checkpoint architecture is saved in `reports/verification_2026-10-27/loaded_architecture.json`. Ultralytics 8.4.189 implements it in `cfg/models/11/yolo11.yaml` and `nn/modules/`. The YAML contains several model scales; selecting the small scale matters even though some generic YAML comments say nano. This project uses one output class.

Convolutions learn spatial features from pixels. Successive stages downsample the image and combine features using C3k2 blocks. SPPF gathers broader spatial context; C2PSA adds attention operations. Upsampling, concatenation and further convolutions combine detailed and coarse features. The detection module uses P3, P4 and P5 features with strides 8, 16 and 32. For a square 640 input these correspond to 80×80, 40×40 and 20×20 feature maps. Actual rectangular batches can differ. The head has box-regression and class branches; it predicts multiple candidate locations and postprocessing suppresses redundant overlapping detections. Do not describe this as an image classifier or claim the attention block makes it a pure transformer.

## Assignment, loss and training

Read `utils/tal.py` and `utils/loss.py` from the recorded Ultralytics version. TaskAlignedAssigner chooses suitable candidate positions for labelled boxes using classification and localization quality, candidate restrictions and conflict resolution. Assignment during training is different from matching predictions to ground truth for evaluation, and different again from nonmaximum suppression.

The detection loss implementation retains the name `v8DetectionLoss`; that name does not mean the checkpoint is YOLOv8. It combines box localization loss, classification BCE and distribution focal loss for box-distance distributions. The recovered run gains are box 7.5, cls 0.5 and dfl 1.5. Backpropagation computes gradients; the optimizer updates weights using training batches. Loss values are optimization signals, not percentages or directly interchangeable with mAP. The original args say optimizer `auto`; recover the resolved optimizer from original console logs before asserting AdamW or SGD.

The recovered run used pretrained YOLO11s initialization, batch 16, image size 640, seed 0, patience 20 and a 100-epoch limit. Its CSV has 83 epochs; epoch 63 has the highest recorded validation mAP50–95. Early stopping explains why the limit was not reached. Initialization and inference checkpoints serve different roles: do not initialize this experiment from the earlier original-split trained pothole checkpoint, which has seen some reassigned test images.

## Explain validation and metrics

Training data update weights. Validation data support model selection and decisions such as an operational confidence threshold. A held-out test estimates performance after those decisions are fixed. Our new verification evaluates the fixed supplied checkpoint; it does not train or tune it. Grouping variants reduces known overlap but does not prove independence between road scenes.

IoU is intersection area divided by union area. With one class and a chosen confidence/IoU matching rule, precision is TP/(TP+FP), while recall is TP/(TP+FN). Example: six correct predictions, two extra predictions and four missed labelled potholes yield 75% precision and 60% recall. A detector can be selective and precise while still missing many potholes.

AP summarizes a precision–recall curve as the confidence threshold changes. mAP averages AP across classes; this dataset has one class. mAP50 uses an IoU match threshold of 0.50. mAP50–95 averages over thresholds 0.50, 0.55, …, 0.95, making accurate localization more demanding. The evaluation NMS IoU parameter 0.7 is not the AP matching threshold. Evaluator-reported precision/recall need not describe the CLI threshold 0.25. CPU timing here is neither a phone benchmark nor end-to-end application latency.

## Rehearsal with evidence

Open original `reports/original_cleaned_v3/test_evaluation/val_batch0_labels.jpg` beside `val_batch0_pred.jpg`. The top-left tile shows a plausible overlapping detection with displayed confidence 0.9. The second tile of the top row has three ground-truth boxes but only two displayed predictions. The fourth tile shows a broad prediction around adjacent labelled potholes and an extra lower box. These are qualitative displayed examples; verify exact matching with coordinates before assigning TP/FP/FN labels. Visualization thresholds can hide lower-confidence predictions. Discuss what evidence would distinguish missed detection from annotation disagreement.

Each member should fill in: name, files/cells actually authored, actual task, commit/run evidence, a short code demonstration and one limitation. Explain a different member's code after that member checks the explanation. Do not turn author metadata into a claim of contribution.

Further-validation rehearsal: propose new road/session images, include roads without potholes, separate scenes, define lighting/size subgroups and keep evaluation settings fixed. Use validation data for any threshold selection or model comparison. Collect labels and provenance before claiming those experiments are complete. CAT 2 does not require a completed model comparison.
