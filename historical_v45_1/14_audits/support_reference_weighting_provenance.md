# Support Reference Weighting Provenance

## Disposition

SOURCE-BACKED: CONFIRMED.

The executed implementation is `tools/analyze_vessel_feature_support.py` (SHA-256 `10cafe0aa75692bdfe3519d929c3df06ae82d173b5bd8b6eae6e8dbd410f3e3c`). The retained comparison identifies the analysis as `frozen_duplicate_backbone_vessel_feature_support_v1` and reports `test_split_used=false`.

## Source semantics inspected

- `load_records` emits one record for every category-5 annotation; the real reference therefore begins as object-instance rows, not acquisition-scene aggregates (lines 50-78).
- `nearest_real_metrics` indexes those records by bbox-scale bin. If a bin exceeds 1,200 objects, `rng.choice(real_indices, 1200, replace=False)` samples object indices; there is no equal allocation by scene (lines 165-190).
- Same-scene synthetic-real and real-real distances are set to infinity before nearest-neighbor selection. This is cross-scene exclusion, not scene-equal weighting (lines 192-200).
- The scale-specific 95th percentile is calculated over one real leave-one-out nearest-neighbor distance per retained real object (lines 197-210).
- `matched_real_sample` likewise samples eligible real object indices after cross-scene exclusion, one object-level match per synthetic record; it does not assign equal scene mass (lines 226-249).

## What this supports

The real-feature reference was object-level and cross-scene. Scenes containing more eligible vessel instances could contribute more object rows, and therefore more empirical mass, than scenes containing fewer vessels. The v44 limitation sentence is supported.

## What this does not support

This audit does not show that the screen is invalid, does not quantify the effect of scene balancing, and does not provide a scene-balanced alternative result. No support value, threshold, MMD, effective rank, ranking, or candidate decision was recomputed.
