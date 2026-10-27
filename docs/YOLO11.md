# YOLO11 integration

The detector now defaults to the trained cleaned-v3 checkpoint at
`models/best.pt`. Generic `yolo11s.pt` is the starting point for training only.

See [MODEL.md](MODEL.md) for teammate setup, the checkpoint checksum, recorded
metrics, and evaluation instructions. See [the README](../README.md) for CLI usage.

The fixed cleaned-v3 checkpoint was independently evaluated on CPU using Ultralytics 8.4.189. All four reported test metrics reproduced to two decimal places. See the [verification record](../reports/verification_2026-10-27/README.md) for conditions and the distinction between supplied original evidence and newly reproduced outputs.
