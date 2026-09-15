# Independent Faster R-CNN replication v16

## Vessel AP50-95

| Condition | Per-seed AP | Mean |
|---|---|---:|
| duplicate | 0.032576, 0.024696, 0.033266, 0.026732, 0.029558 | 0.029365 |
| unity_medium150 | 0.024829, 0.025517, 0.030762, 0.029259, 0.038231 | 0.029720 |
| realcutout_medium150 | 0.034685, 0.031756, 0.034151, 0.018868, 0.035379 | 0.030968 |

## Frozen decisions

- Unity - Duplicate: mean +0.000354, one-sided 95% LCB -0.005444: **FAIL**.
- Unity - RealCutout, margin -0.005: mean -0.001248, one-sided 95% LCB -0.008874: **FAIL/NOT TESTED**.
- External HRSC/DIOR evaluation: **LOCKED**.

The detector is a two-stage proposal-based Faster R-CNN with a frozen MobileNetV3/FPN backbone and trainable RPN/ROI heads. All paired initialization hashes matched.
