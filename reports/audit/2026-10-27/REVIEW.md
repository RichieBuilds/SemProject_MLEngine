# RoadWatch CAT 2 verification — 27 October 2026

AI-assisted inspection, verification, organization and study support. Existing source notebooks and supplied run evidence are distinguished from newly reproduced outputs. This report and notebook organization are not team-authored assessed analysis; members must supply their own interpretations and actual contribution records.

## Verified outcome

The cleaned-v3 dataset archive and checkpoint match their supplied SHA-256 values. Every existing cleaning code cell ran successfully with `WRITE_COPY=False`. The fixed checkpoint was evaluated once on validation and once on the held-out test, without retraining or tuning. All four test metrics reproduce the supplied percentages to two decimal places.

| Evidence | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Original epoch 63 validation CSV | 77.770% | 66.653% | 73.841% | 41.528% |
| Reproduced validation | 77.813% | 66.545% | 73.820% | 41.527% |
| Reproduced held-out test | 78.966% | 67.946% | 77.123% | 48.406% |
| Previously reported test | 78.97% | 67.95% | 77.12% | 48.41% |

Validation differences relative to the original epoch-63 CSV are small (at most 0.109 percentage points among these four metrics). The recorded original environment is Python 3.13.16, PyTorch 2.11.0+cu130 and Tesla T4; this reproduction used Python 3.13.17, PyTorch 2.14.1+cpu and AMD Ryzen 5 7530U CPU. Both use Ultralytics 8.4.189. Hardware and numerical differences are plausible contributors; no controlled experiment isolated their individual effects. `evaluation_conditions.json`, package inventories and console logs record the actual reproduction. Original full console and package-lock evidence remain unavailable; the original environment values come from supplied model notes. Do not present new outputs as original Colab outputs.

Evaluation used 640 pixels, batch 16, CPU, four PyTorch threads, workers 0, float32, no test-time augmentation, confidence floor 0.001, NMS IoU 0.7 and max_det 300. These fixed settings are recorded; evaluator P/R are not the CLI confidence 0.25 operating point. Optional JSON export also emitted COCO metrics; the table above uses Ultralytics box metrics consistently with the recorded results. The evaluator installed faster-coco-eval for that export; final package versions are saved. CPU speed is not a mobile benchmark.

## Data checks and EDA measurements

| Split | Images | Boxes | Empty labels |
|---|---:|---:|---:|
| train | 3196 | 8141 | 0 |
| valid | 375 | 1097 | 0 |
| test | 369 | 873 | 0 |
| Total | 3940 | 10111 | 0 |

All original and cleaned images decode, image–label pairing passes, and every nonempty label satisfies the existing finite/class/normalized box-boundary checks. Every cleaned image and label matches its original file bytes. There are no byte-identical image duplicate sets. This verifies file integrity and syntax, not exhaustive semantic annotation quality.

Four reproducible observations for team interpretation:

1. Annotation counts vary from 1 to 21 potholes per image. There are 1,767 single-pothole images; distributions are saved per split in `cleaned_v3_summary.json`.
2. There are 834 filename source groups. In the original split, 385 groups span partitions; in v3, none do. Group names identify filename variants, not necessarily unique scenes.
3. The reproduced v2 pHash audit finds exactly the eight recorded cross-split pairs. Replaying the documented movement of five whole groups (16 images) yields the exact v3 manifest and zero v3 cross-split pairs at 64-bit pHash Hamming distance <=5. Both recorded similarity CSVs match. Heuristic grouping and hashing do not prove complete scene independence.
4. There are zero empty labels in all splits. This dataset therefore supplies no empty-label negative examples; this does not establish all visible potholes are annotated or that deployment false positives are controlled.

Additional measurement: median normalized box area is 0.02832; the middle half lies between 0.00952 and 0.08977. Original-split counts were 3345/397/198 images and 8705/973/433 boxes. Totals match v3 because the process reassigns groups rather than deleting images. The original split had 85 flagged cross-split pHash pairs. Those initial-check outputs remain as a historical inspection record, not the final v3 result.

## Assets, provenance and navigation

- [Model notes](../../../docs/MODEL.md): recovered source documentation; the inference command now explicitly selects the authenticated checkpoint.
- [Dataset exploration](../../../pothole_detection_dataset_YOLOv11_optimized_Exploration.ipynb): identical to supplied transfer copy; original saved outputs retained. Wide-CSV NaN padding does not alone prove dataset completeness.
- [Cleaning source notebook](../../../notebooks/roadwatch_dataset_cleaning.ipynb): identical to transfer copy; original remains without saved outputs.
- [Reproduced cleaning notebook](verification/cleaning_reproduced.ipynb): actual new outputs, explicitly labelled.
- [Cleaned-v3 training notebook](../../../train_yollo_collab.ipynb): restored from supplied records; original-split version preserved in [archive](../../../notebooks/archive/train_original_split.ipynb). Neither source training notebook has saved original outputs. No training cells were executed in this review.
- [Original training/evaluation evidence](../../original_cleaned_v3/README.md): downloaded CSV, settings, curves, confusion matrices and prediction/label mosaics.
- [Submission organizer](../../../YOLO11s_Base.ipynb): existing work and verified evidence assembled with provenance; team-authored interpretation and contribution record remain incomplete.
- [Study notes and rehearsal](PRACTICE.md): AI-assisted explanations and questions; no invented member contributions.
- [Verification inventory](verification/drive_asset_inventory.json): source Drive IDs, sizes and byte hashes for 38 downloaded files. Transfer-file comparison also saved.

Dataset ZIP SHA-256: `31fff83565a63da76e32419825cd8b613f5aac322ed00df62470b53f9bf4785e`.
Checkpoint SHA-256: `9dad6abd86b08e02d76a02f5e0190ddf373b39de9b8487646f01a541e8810291` (local, Drive and MODEL.md match).

Drive source: https://drive.google.com/drive/folders/1aFuFXOHfQRn9pq2Pk1dQmQjX-btcsqeV . The three dataset parts were assembled in numeric order and the ZIP checksum verified before extraction. ZIP paths were checked to remain within staging folders; CRC checks passed. Archives and downloads are staged outside the repository under this chat's work/drive_assets. Dataset files are installed under ignored `data/cleaned`; checkpoint bytes were already correct and were not replaced. The archive YAML contains a Colab path, so evaluation used a separately generated machine-specific YAML; it did not silently edit the supplied YAML.

## Training record and qualitative examples

The recovered args specify pretrained yolo11s.pt, image size 640, batch 16, seed 0, patience 20, maximum 100 epochs and optimizer auto. Original results.csv has 83 rows; best validation mAP50–95 is epoch 63 (0.41528), with last logged elapsed time 4202.44 seconds. Do not infer a resolved optimizer from the word auto. The earlier original-split checkpoint cannot be a cleaned-v3 independent baseline because some images were reassigned.

The original test batch-0 label and prediction mosaics offer actual success/failure examples. The top-left tile shows a plausible high-confidence overlapping detection. The next tile contains three labels and two displayed predictions. The fourth tile contains a broad prediction around adjacent labelled potholes and an extra lower box. These are qualitative visual observations, not independently counted per-box matching results; displayed plotting thresholds can hide low-confidence predictions. The team should explain these examples against exact coordinates before asserting TP/FP/FN.

## Checkout changes and remaining work

Inspected `dek_Engine` at `3f0086e`, initially clean. Origin moha6959/SemProject_MLEngine; upstream i-ssa/SemProject_MLEngine. No applicable AGENTS.md was found. Supporting notebooks, original data, review images and existing user changes were preserved. No files deleted, Git history rewritten, commits made or remote state changed.

Small manifests and similarity CSVs are restored and visible through narrow ignore exceptions; large datasets and weights remain ignored. Existing original data (7,884 files) were already tracked before this work; ignore rules do not retroactively remove them. No untracking or history cleanup was attempted. The empty `notebooks/setup_and_data.ipynb` remains a documented zero-byte placeholder, not a valid notebook. No unneeded notebook was deleted.

YOLO11s naming and navigation are reconciled in README, DESIGN and YOLO11 notes. The older methodology chapter mentioning YOLO11n is absent from this checkout and still needs the team's revision. Android integration, mobile export and phone performance have not been demonstrated. Further validation should use independently sourced roads/sessions, scene separation, negative images, predefined size/lighting subgroups and validation-only comparison/threshold selection; these remain plans, not completed experiments.

Members' names, actual contributions and evidence are still required. No contributions were inferred from Git author metadata. The notebook intentionally marks team explanation/contribution slots as unfinished rather than fabricating assessed work.

Final checks: existing three unit tests, notebook schema checks for all nonempty notebooks, local documentation/notebook asset links, Git whitespace checks and output decoding. These check repository integrity; model-quality evidence comes from the separately recorded cleaning and evaluation runs. The empty placeholder notebook is the known schema exception.

## Audit organization

Optional review/study material and verbose verification outputs were consolidated here after the initial review. The concise record and primary EDA plot remain in `reports/verification_2026-10-27/`; original training evidence and the earlier notebook archive remain separate. Historical paths in saved run records describe their original execution location.
