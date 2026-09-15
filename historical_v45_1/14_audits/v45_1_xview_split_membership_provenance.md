# v45.1 xView Split-Membership Provenance

## Retained exact mapping

An existing machine-readable mapping was found at `PreparedData/xview_expanded_512_v1/scene_splits.json`.

- SHA-256: `c7e6ed7fc95b344731a6e1e7eb01c6a6a01694282f8b8804013cf4099c28befa`
- retained timestamp: 2026-07-27 16:05:59 KST
- entries: 61 acquisition-scene identifiers
- assignments: 43 train, 9 validation, 9 test
- generator: `tools/prepare_xview_patches.py`, which writes the sorted deterministic `scene_splits.json` and enforces source-scene disjointness
- independent retained hash trace: `outputs/ieee_access_manuscript_v37_revision/revision_audit.md`

The assigned-scene map agrees with the source-scene values in the retained canonical COCO files: 1,500 train patches from 40 used scenes, 375 validation patches from 9 scenes, and 375 test patches from 9 scenes, with zero assignment mismatches. Three train-assigned scenes contribute no selected patch, explaining 43 assigned train scenes versus 40 used train scenes.

## Redistribution decision

The exact mapping is locally recoverable and source-traced. However, no retained license or permission record establishes that the xView acquisition-scene identifiers may be redistributed. The v45 exclusion audit also excludes generated per-image labels/manifests containing local identifiers when redistribution is not verified.

Therefore the exact mapping is not copied, transformed, sanitized, or regenerated for the public package. This is a rights/provenance boundary, not a claim that the mapping is missing locally.

- `EXACT_XVIEW_SPLIT_MEMBERSHIP_SOURCE_FOUND = TRUE`
- `EXACT_XVIEW_SPLIT_MEMBERSHIP_INTEGRITY_VERIFIED_LOCALLY = TRUE`
- `IDENTIFIER_REDISTRIBUTION_PERMISSION_SOURCE_FOUND = FALSE`
- `EXACT_XVIEW_SPLIT_MEMBERSHIP_PUBLICLY_INCLUDED = FALSE`
- `SPLIT_AVAILABILITY_STATE = STATE B`
