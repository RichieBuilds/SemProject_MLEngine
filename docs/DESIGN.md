# Design Notes

**Chosen detector:** YOLO11s, initialized from pretrained `yolo11s.pt`. Nano/small alternatives are not completed model comparisons.

## 1. Background: what is YOLO?
YOLO stands for "You Only Look Once". It is a family of deep learning models
used for real-time object detection. It finds objects in an image or video and
predicts where they are [1].

For each object, the model predicts three things:
- **Location:** a bounding box around the object.
- **Class:** what the object is, such as a car, person or pothole.
- **Confidence score:** how confident the model is in its prediction.

YOLO was introduced by Redmon, Divvala, Girshick and Farhadi (arXiv 2015;
CVPR 2016) [1]. Earlier detectors reused classifiers on many candidate
regions, which was slow. YOLO treats detection as one problem: a single
network sees the whole image once and outputs bounding boxes and class
probabilities, so it trains end-to-end and runs in real time [1].

## 2. From YOLO to YOLO11
After the original YOLO, other researchers and companies kept improving it.
Each new version made it faster, more accurate, or easier to use (for example,
moving to the PyTorch framework, which most people now work in) [2].

YOLO11, released by Ultralytics in 2024, builds on YOLOv8 [2][3]. It is better
at focusing on the important parts of an image [2], and it reaches similar or
better accuracy with a smaller, faster model (as reported by Ultralytics)
[2][3].

## 3. Why we chose YOLO11
Newer versions exist, such as YOLO26, which Ultralytics now recommends for new
projects [4]. We still chose YOLO11 because:
- It is well documented and has lots of tutorials and examples [3].
- It comes in sizes from nano to extra-large [2], and the small ones train on
  free GPUs.
- It looks at the whole image in one pass, which suits road images from a
  moving vehicle.

Known risk: the original YOLO paper reports more localization errors than the
best detectors of its time [1], and small potholes are likely to be affected
most. We check this in our results.

## References
[1] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once:
    Unified, Real-Time Object Detection," CVPR, 2016. arXiv:1506.02640.

[2] R. Khanam, M. Hussain, "YOLOv11: An Overview of the Key Architectural
    Enhancements," arXiv:2410.17725, 2024.

[3] G. Jocher, J. Qiu, "Ultralytics YOLO11," 2024.
    https://docs.ultralytics.com/models/yolo11/
    https://github.com/ultralytics/ultralytics

[4] Ultralytics, "YOLO11 vs. YOLO26," Ultralytics Docs (accessed Oct 2026).
    https://docs.ultralytics.com/compare/yolo11-vs-yolo26

Dataset attribution: the original export records CC BY 4.0, Roboflow workspace
`vasanthakumar662005`, project `pothole-pq3lu`, version 1. See
`data/1/README.roboflow.txt` and `data/1/data.yaml` for the retained source metadata.
See [MODEL.md](MODEL.md) for the current trained checkpoint and dataset splits.
