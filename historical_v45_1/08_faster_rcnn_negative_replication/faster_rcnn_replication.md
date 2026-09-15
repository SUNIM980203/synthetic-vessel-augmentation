# Independent Faster R-CNN replication protocol v16

Frozen locally on 2026-07-29 before any Faster R-CNN training or evaluation. The YOLO26n, HRSC, and DIOR results are already known. This experiment addresses detector-family dependence; it does not create a new method-selection opportunity.

## Detector and initialization

Use torchvision 0.26.0 `fasterrcnn_mobilenet_v3_large_fpn` with its default COCO_V1 weights and six output labels including background. Freeze the MobileNetV3/FPN backbone. Train the class-agnostic region proposal network and ROI classification/regression heads. This two-stage proposal-based detector is architecturally independent of the one-stage YOLO26n primary model.

For each seed, recreate a common initialized state after replacing the COCO ROI predictor. Duplicate, Unity Medium150, and RealCutout Medium150 must have identical initial trainable-parameter SHA-256 values within that seed. No pretrained xView YOLO weights are transferred.

## Data and training

Use the existing three exact exposure conditions: 1,500 real plus 1,500 exact duplicates; 1,500 real plus 1,500 Unity Medium150 counterparts; and 1,500 real plus 1,500 geometry-matched RealCutout counterparts. Preserve every YOLO box, map class IDs 0--4 to Faster R-CNN labels 1--5, and use no stochastic image augmentation beyond the already materialized dataset conditions.

Run five paired seeds (20260723--20260727), 10 epochs, batch 4, workers 0, 640-pixel minimum and maximum transform size, AdamW with learning rate 1e-4 and weight decay 1e-4, and AMP. Evaluate exact epoch-10 checkpoints; do not select by validation score. For COCO evaluation retain scores from 0.001, apply 0.5 ROI NMS, and retain at most 300 detections per image.

## Locked decisions

The primary endpoint is maritime-vessel COCO AP50-95 on the unchanged 375-image scene-disjoint xView validation split. Unity replication passes only if the one-sided 95% paired Student-t lower bound for `Unity - Duplicate` exceeds zero. Only after that gate passes, test Unity versus RealCutout non-inferiority using a -0.005 AP margin.

Report aggregate and per-class AP, all five paired results, initialization hashes, parameter counts, duration, and raw predictions. HRSC and DIOR evaluation is unlocked only if the internal Unity gate passes. A failed result is retained and reported without changing the schedule, initialization, or threshold.
