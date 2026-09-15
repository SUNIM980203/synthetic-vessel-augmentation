# Prospective HRSC2016-MS external validation protocol (v13)

## Registration state

This protocol was frozen on 2026-07-29 (Asia/Seoul) before downloading HRSC2016-MS, converting its annotations, or evaluating any checkpoint on it. No HRSC2016-MS image, annotation, or metric may be used for model selection, hyperparameter tuning, checkpoint selection, or early stopping.

## Fixed research question

On a completely external optical ship dataset, does the locked Unity Medium-150 head-only adaptation (1) improve ship detection over the exact-duplicate exposure control and (2) remain non-inferior to an equally dosed, geometry-matched real-vessel cutout control?

The intended claim is deliberately narrower than general remote-sensing robustness: synthetic Unity vessels can provide externally transferable ship-detection benefit under a fixed xView-to-HRSC domain shift, and that benefit is not meaningfully worse than the matched real-cutout intervention.

## External dataset and independence rule

The fixed external dataset is HRSC2016-MS version 1, distributed by Weiming Chen on Kaggle. Its data card reports 1,680 optical images and 7,655 ship instances. The benchmark is described in Chen et al., *MSSDet: Multi-Scale Ship-Detection Framework in Optical Remote-Sensing Images and New Benchmark*, Remote Sensing 14(21):5460 (2022), DOI 10.3390/rs14215460.

- Dataset page: https://www.kaggle.com/datasets/weiming97/hrsc2016-ms-dataset
- Paper: https://doi.org/10.3390/rs14215460
- Dataset license shown by the distributor: ODbL 1.0 for the database, original authors' rights for contents.
- Evaluation population: the official test identifiers supplied in the downloaded `ImageSets` directory.
- If a unique official test list cannot be identified, or listed images/annotations are materially missing, evaluation stops. The all-image set will not be substituted after inspecting model results.
- HRSC2016-MS remains evaluation-only for every condition. It is not a source of training crops, backgrounds, thresholds, or qualitative method changes.

## Fixed conditions and checkpoints

All conditions start from `runs/expanded_v1/duplicate_yolo26n_s20260723/weights/best.pt`. Modules 0-22 are frozen; only module 23 (`Detect`) is trained for exactly 20 epochs with image size 640, batch 16, workers 0, AdamW, `lr0=1e-4`, `lrf=0.1`, 0.5 warm-up epochs, and no effective early stopping. The evaluated checkpoint is the exact epoch-20 `last.pt`, never validation-selected `best.pt`.

| Condition | Added exposure | Fixed seeds | Existing before registration | New runs required |
|---|---|---|---:|---:|
| Exact duplicate | No new object appearance | 20260723-20260727 | 3 | 2 |
| Unity Medium-150 | 150 Unity vessels, 32-64 px bbox scale | 20260723-20260727 | 3 | 2 |
| RealCutout Medium-150 | 150 xView-train vessel crops at the same hosts, positions, and boxes | 20260723-20260727 | 1 | 4 |

The five runs are paired by seed. Training of all missing checkpoints must finish before the first HRSC metric is computed. A failed run may be rerun only with the identical condition and seed after documenting the technical failure; a different seed may not replace it.

## Annotation conversion lock

Pre-inference schema inspection confirmed that every HRSC2016-MS object supplies both `bndbox` and `robndbox`. Axis-aligned evaluation uses the official `bndbox` coordinates directly; it does not recompute an envelope from `robndbox`. Boxes are clipped to image bounds. Degenerate boxes are rejected and reported. Images with zero valid boxes remain in the test population if the official test list includes them.

Every ship is assigned detector class ID 4 (`maritime_vessel`). Dataset metadata retains the full five-class xView name map so the trained model's output indices are unchanged. Conversion must produce a machine-readable manifest containing source paths, official test IDs, image dimensions, object counts, rejected-object reasons, archive hash, and converter version/hash. A visual audit sample is selected deterministically before any inference.

Axis-aligned evaluation does not measure oriented localization and will be stated as a limitation. No hand correction of boxes is allowed after model predictions are viewed. This schema clarification was made before any checkpoint inference.

## Fixed evaluation

- Software: Python 3.12.13, Ultralytics 8.4.104, PyTorch 2.11.0+cu128.
- Device: CUDA device 0; workers 0; batch 8; image size 640.
- Split: the converted official HRSC2016-MS test list, exposed as `test` in the data YAML.
- Primary metric: class-4 ship AP50-95 for axis-aligned boxes.
- Secondary descriptive metrics: AP50, AP75, precision, recall, prediction count, and inference time. These cannot rescue a failed primary decision.
- The same dataset YAML, library versions, and evaluation command are used for all 15 checkpoints.
- Raw prediction/metric artifacts are retained. Results are not rounded before statistical analysis.

## Ordered hypotheses and decision rules

Let `U_s`, `D_s`, and `R_s` be external ship AP50-95 for Unity, duplicate, and RealCutout at seed `s`. For each paired difference vector with `n=5`, report its mean, sample standard deviation, and the lower one-sided 95% Student-t confidence bound:

`LCB95 = mean(d) - t_(0.95, 4) * sd(d) / sqrt(5)`.

The tests are hierarchical:

1. **External efficacy:** `d_UD = U_s - D_s`. Unity passes only if `LCB95(d_UD) > 0`.
2. **Real-data non-inferiority:** tested inferentially only if step 1 passes. Define `d_UR = U_s - R_s` and the smallest acceptable margin as `-0.005` absolute AP50-95 (0.5 AP point). Unity is non-inferior only if `LCB95(d_UR) > -0.005`.

The 0.005 margin was fixed before external evaluation and matches the prior aggregate safety tolerance while remaining smaller than the locked internal Unity vessel gain of 0.007705. This justification is design-based; the external result will not be used to revise the margin.

If external efficacy fails, all RealCutout comparisons are descriptive. If efficacy passes but non-inferiority fails, the conclusion is that Unity transfers but has not been shown to match the real-vessel control. Superiority over RealCutout may be reported only if `LCB95(d_UR) > 0`; it is not required for the primary claim.

Because five training seeds provide limited precision, exact per-seed values, paired differences, and confidence bounds will be reported without presenting a non-significant result as proof of equality.

## Cost and reproducibility outcomes

Cost is secondary and cannot override the accuracy decision. For each augmentation condition, retain:

- number of source assets and unique source objects;
- human interaction minutes measured from contemporaneous logs, or `not measured` rather than a retrospective estimate;
- Unity render GPU/CPU wall time, compositing time, training GPU wall time, and evaluation time;
- generated bytes and peak disk footprint;
- failures/retries and software/hardware identifiers.

Report AP gain over duplicate per 150 added objects and the Unity-to-RealCutout ratio of measured preparation time. Monetary cost is reported only if supported by an explicit rate table fixed before calculation.

## Blinding, amendments, and reporting

Dataset integrity and conversion audits may inspect annotations and rendered boxes but not model predictions. The evaluator should write all 15 result files before the comparison summary is opened. Every condition and seed is reported regardless of outcome.

Any unavoidable amendment must be written, timestamped, and hashed before the first model evaluation. After the first HRSC metric is produced, the dataset population, converter, checkpoint rule, margin, seed set, endpoint, and inferential method are immutable.

## Completed locked result

All 15 exact epoch-20 checkpoints were evaluated once on the official 610-image HRSC2016-MS test population (3,249 ship instances). Mean ship AP50-95 was 0.125004 for Duplicate, 0.136855 for Unity Medium-150, and 0.132310 for RealCutout Medium-150.

Unity exceeded Duplicate in 5/5 paired seeds. The mean paired improvement was +0.011851 and its one-sided 95% lower confidence bound was +0.010902, so the external-efficacy gate passed. Unity also exceeded RealCutout in 5/5 seeds; the mean paired difference was +0.004546 and its lower bound was +0.002554. The pre-registered non-inferiority gate therefore passed, and the lower bound above zero additionally supports the protocol-allowed superiority conclusion for this fixed external setting. The hierarchical primary claim passed.
