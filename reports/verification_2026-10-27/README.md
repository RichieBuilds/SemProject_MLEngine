# Cleaned-v3 verification summary

AI-assisted verification on 27 October 2026, separate from supplied original Colab artifacts.

- [Dataset checks and split counts](cleaned_v3_summary.json): 3,940 images / 10,111 boxes; decoding, pairing, annotations and byte comparisons passed. Recorded eight v2 pHash pairs reproduce; v3 has zero cross-split matches at distance <=5. Grouping/hashing do not prove scene independence.
- [Primary EDA plot](cleaned_v3_eda.png)
- [Validation and test metrics](evaluation_metrics.json)
- [Evaluation conditions](evaluation_conditions.json)
- [Metric comparison](metric_comparison.json)

Test precision 78.97%, recall 67.95%, mAP50 77.12%, mAP50–95 48.41% reproduce the reported values to two decimals. Ultralytics 8.4.189 matched; CPU/Python/PyTorch differed from the original reported T4 environment. No retraining or test-driven tuning was performed.

[Optional detailed audit](../audit/2026-10-27/README.md) · [Original training evidence](../original_cleaned_v3/README.md) · [Model notes](../../docs/MODEL.md)
