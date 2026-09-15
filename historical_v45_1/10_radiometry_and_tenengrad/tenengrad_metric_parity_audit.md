**POST_RESULT_DIAGNOSTIC**

# Tenengrad metric parity audit

**Classification: TENENGRAD_IMPLEMENTATION_PARITY = EQUIVALENT**

- Generic and historical branches both call `extended_appearance_metrics` and `distribution_gate` from `tools/support_scale_radiometry_calibration_common.py`.
- Current, generic-frozen, and historical-frozen source SHA-256 are all `4b936e59c521b0955d33aeca436d1178f00c6060d02bb50cfcfba8bf9bf99aea`: **TRUE**.
- RGB is converted with `cv2.COLOR_RGB2GRAY`, cast to `float32`, and divided by 255.
- Sobel uses `CV_32F`, kernel size 3, x/y derivatives; magnitude is `sqrt(gx^2 + gy^2)`.
- ROI is the clipped exact bbox versus the frozen expanded local ring from `object_and_ring_masks`; no metric-stage resize is performed.
- Tenengrad is mean foreground gradient divided by max(mean ring gradient, 1e-5).
- Group Wasserstein uses `scipy.stats.wasserstein_distance` divided by native sample SD (`ddof=1`).
- Both branches use the same 150-row `Native real` reference from the same quality CSV hash `cf7b9cb0e15a3a82972fb572974b7e3a55761f0546edc060465c2b930282b9be`.
- Exact package versions were not independently serialized for both completed runs; therefore byte-identical runtime provenance is not overclaimed. Current versions are recorded in `tenengrad_metric_parity_hashes.json`.
- Independent selected-JPEG recomputation maximum value discrepancy: 0.000000.

No implementation difference was found. This audit does not alter the frozen FAIL.
