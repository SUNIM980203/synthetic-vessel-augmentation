# Frozen Small-scale feasibility protocol

Frozen before any new Small host pixel was opened.

- Only scientific change: bbox geometric mean Medium [32,64) -> Small [16,32), implemented by exact 0.5 width/height scaling about the unchanged center.
- 2,080 raw and first 2,048 unique valid final candidates per host.
- Contrast 1.25; unsharp 0.00; identical appearance taxonomy/ranges, calibration, support representation, S3 selector, geometry, placement, and integrity rules.
- Support: median >=0.060; fraction >=0.040 at least 0.80; minimum >=0.005.
- Absolute Weber standardized Wasserstein <=0.50 in each arm; all other Medium thresholds unchanged.
- One shot, no retry, no tuning, no host replacement, no detector training, no AP inspection.
