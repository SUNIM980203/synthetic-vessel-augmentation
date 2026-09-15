# Feature-support confirmation protocol v20

Frozen locally on 2026-07-31 before materializing the Small150 dataset or training any Small150 checkpoint. The five-seed Medium150 validation results are already known. This is therefore a prospective test of a new lower-support branch against a fixed historical comparator, not a blinded preregistration and not a new held-out-test claim.

## Question

Under the unchanged YOLO26n head-only adaptation interface, does the previously selected, higher-support Medium150 distribution produce greater maritime-vessel AP50-95 than a count-matched adjacent-scale Small150 distribution?

## Fixed distributions

- **Small150:** exactly 150 Unity vessel insertions, placed one per modified counterpart image, with realized bounding-box scale `sqrt(width x height)` in `[16, 32)` pixels.
- **Medium150:** the existing fixed dataset with exactly 150 Unity vessel insertions in `[32, 64)` pixels.
- Both branches retain the same 1,500 real xView training patches and add 1,500 counterpart patches. The Unity cutout library, rendering family, random seed, strict-water constraint, semantic placement, sensor-aware compositing, vessel dose, and training schedule are held fixed. Object scale is the intended distributional intervention.

Small150 is preferred to Tiny150 because it is adjacent to the selected medium bin and avoids making extreme small-object detectability the dominant contrast. Scale and feature support are nevertheless not experimentally separable in this design; the result tests the ranking value of the complete scale-support intervention, not a scale-independent causal effect of support.

## Training-only support gate

Before model training, Small150 must satisfy all data-integrity checks: exactly 1,500 counterpart images, exactly 150 modified images with one Unity vessel each, exactly 150 Unity vessel annotations, no non-vessel Unity annotation, all inserted scales in `[16, 32)`, all boxes inside image bounds, and successful image decoding. Its frozen-representation supported fraction must be lower than Medium150's fixed value of 0.740. If this contrast is absent, training remains locked.

The one-object-per-image rule was clarified after an initial materialization audit found that the builder's default catch-up behavior placed 150 objects in 127 images. A second structural audit with the one-object limit and four placement retries reached only 124 objects. No feature or detector result from either rejected materialization was computed. Both directories were quarantined. The final accepted build keeps every strict-water acceptance rule unchanged and uses 32 placement retries only to reach the fixed dose. The final protocol hash is computed after these structural amendments and before the accepted Small150 materialization or any training.

The support analysis uses the same seed-20260723 Duplicate initialization, xView training objects only, P3/P4/P5 ROIAlign representation, scale-specific cross-acquisition-scene real neighbors, and p95 support definition used in the original screening study. Support and MMD are manipulation checks, not efficacy endpoints.

## Fixed head-only training

Five paired optimization seeds are `20260723` through `20260727`. Every Small150 run starts from `runs/expanded_v1/duplicate_yolo26n_s20260723/weights/best.pt`. Modules 0-22 are frozen and only module 23 (`Detect`) is trainable. Training uses 20 complete epochs, 640-pixel inputs, batch size 16, workers 0, AdamW, `lr0=1e-4`, `lrf=0.1`, warm-up 0.5 epochs, patience 100, deterministic mode, and the existing augmentation configuration. The exact epoch-20 `last.pt` checkpoint is required. The five existing Medium150 checkpoints are not retrained or reselected.

## Evaluation and endpoints

Only the unchanged scene-disjoint xView validation split is used. The internal test set remains closed and HRSC2016-MS and DIOR are not used for training, selection, threshold adjustment, or the primary v20 decision. Evaluation uses 640-pixel inputs, batch size 8, workers 0, confidence floor 0.001, NMS IoU 0.7, and maximum 300 detections.

The primary paired difference is:

`d_seed = vessel_AP50-95(Medium150) - vessel_AP50-95(Small150)`.

The feature-support ranking is confirmed only if the one-sided 95% Student-t lower confidence bound across the five paired seeds is greater than zero. Aggregate mAP50-95, per-class AP50-95, paired wins, and Small150-minus-Duplicate effects are secondary descriptive results. No threshold, checkpoint, seed, or endpoint may be changed after Small150 training starts.

## Claim boundary

A positive primary result supports the statement that the training-only support procedure correctly ranked these two count-matched scale distributions under the fixed head-only YOLO recipe. It does not establish that feature support is a universal surrogate, that support rather than scale caused the difference, or that the result generalizes to other renderers, classes, detectors, or domains. A failed result must be retained and reported.
