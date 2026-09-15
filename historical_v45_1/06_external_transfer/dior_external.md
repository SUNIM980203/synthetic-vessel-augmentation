# DIOR external generalization protocol v14

Frozen locally on 2026-07-29 before any of the 15 existing checkpoints were run on DIOR. This protocol is a prospective second-external-dataset test; HRSC results were already known when it was written.

## Dataset and independence

Use every one of the 3,463 examples in the public `HichTala/dior` test split at repository revision `b9d3ee2b3f3a7c6ba33f7061b1ea898785c7c8e0`. The mirror is a COCO/Hugging Face restructuring of DIOR, an optical remote-sensing benchmark. It is selected instead of ShipRSImageNet because ShipRSImageNet explicitly incorporates xView and HRSC2016 imagery, which would compromise independence from this study's training and first external dataset.

The split is the mirror's declared test partition, not the original DIOR official test partition. This distinction must be stated in the paper. No DIOR image or annotation may be used for training, hyperparameter selection, early stopping, synthetic parameter selection, or condition selection.

## Conversion

- Retain all 3,463 test examples, including examples with zero ships.
- Map source category 13 (`Ship`) to the existing detector class 4 (`maritime_vessel`).
- Ignore other source categories as ground truth, while vessel-class detections on those objects or backgrounds remain eligible false positives.
- Interpret the provided boxes according to the Parquet feature metadata, clip them to image bounds, and reject only boxes with non-positive clipped width or height.
- Do not tile, crop, remove difficult cases, or select ship-positive images.
- Record source shard SHA-256 hashes, conversion counts, rejected/clipped counts, prepared-data hashes, and a visual annotation audit before inference.

## Locked models and inference

Evaluate the exact epoch-20 `last.pt` files from the five paired seeds (20260723--20260727) for Duplicate, Unity Medium150, and RealCutout Medium150. Use Ultralytics 8.4.104, Python 3.12.13, PyTorch 2.11.0+cu128, image size 640, batch 8, workers 0, confidence floor 0.001, NMS IoU 0.7, and maximum 300 detections. Save aggregate metrics and raw per-image predictions.

The primary endpoint is vessel AP50-95 across the complete test split. AP50, AP75, precision, recall, and size-stratified AP are descriptive secondary endpoints.

## Ordered decisions

1. Unity external efficacy: paired seed difference `Unity - Duplicate`; pass only if the one-sided 95% Student-t lower confidence bound (df=4) exceeds 0.
2. RealCutout comparison: tested only after step 1 passes; `Unity - RealCutout` is non-inferior only if the same lower bound exceeds -0.005 AP.

Report all five paired values even if a gate fails. A two-sided 95% paired image-cluster bootstrap (2,000 replicates; seed 20260729) is a secondary sensitivity analysis and cannot reverse the ordered primary decisions. No threshold, margin, split, or exclusion rule may be changed after inference.
