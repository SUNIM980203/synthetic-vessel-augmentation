# Historical Medium150 orientation recovery gate report

## User-defined dual success criterion

1. Selected source-cutout sequence directly observed from the exact historical deterministic logic: **TRUE**
2. Instrumented replay byte-identical to the canonical historical Medium150 dataset: **TRUE**

`DUAL_SUCCESS_CRITERION_PASSED = TRUE`

Baseline replay canonical-identical: **TRUE**

Instrumented replay output-preserving: **TRUE**

Repeat replay deterministic: **TRUE**

Historical Medium150 frames: **150**

Source-cutout identities recovered: **150 / 150**

Source-cutout hashes recovered: **150 / 150**

Exact native raster orientations recovered: **150 / 150**

Numeric orientation metadata recovered: **0 / 150**

Independent in-plane rotation applied: **FALSE**

Current High/Low pipeline orientation compatibility: **150 / 150 with implementation mapping; 0 / 150 without mapping**

## Overall

```text
CASE = CASE_B_CUTOUT_RECOVERED_NUMERIC_ANGLE_ABSENT
SUCCESS_CONDITION_1_DIRECT_SOURCE_SEQUENCE_OBSERVED = TRUE
SUCCESS_CONDITION_2_INSTRUMENTED_REPLAY_CANONICAL_BYTE_IDENTICAL = TRUE
DUAL_SUCCESS_CRITERION_PASSED = TRUE
HISTORICAL_SOURCE_CUTOUT_RECOVERED = TRUE
HISTORICAL_NATIVE_ORIENTATION_RECOVERED = TRUE
EXACT_NATIVE_RASTER_ORIENTATION_RECOVERED = TRUE
NUMERIC_ORIENTATION_RECOVERED = FALSE
READY_FOR_MEDIUM150_HIGH_LOW_FRAME_FREEZE_REVIEW = TRUE
HIGH_LOW_CANDIDATES_GENERATED = FALSE
TRAINING_UNLOCKED = FALSE
```

## Scientific claim boundary

The exact historical source-cutout raster, including its native image-plane orientation, was deterministically reconstructed from a passive replay that reproduced the retained historical dataset bit-for-bit. Orientation was not explicitly recorded historically, and no degree-valued angle was recovered or inferred.

## Stop

No High/Low candidate was generated, no SegFormer host selection was run, no detector was trained, no AP was evaluated or used, and no manuscript was revised. This report authorizes review only, not the next experiment.

The two instrumented replay dataset trees are transient verification artifacts. Their complete tree and per-file hashes are frozen in the audit package; the canonical historical dataset, baseline r24 replay, source cutouts, instrumented code, selection logs, manifests, comparisons, and reports are retained. The transient A/B dataset trees also remain present because the execution environment blocked the requested post-hash deletion; no deletion bypass was attempted.
