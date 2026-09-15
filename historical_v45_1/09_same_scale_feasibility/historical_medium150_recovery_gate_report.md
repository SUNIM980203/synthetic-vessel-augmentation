# Historical Medium150 recovery gate report

## Outcome

The exact historical host/location/bbox frame is recoverable for all 150 interventions, but the complete orientation/source-transform provenance is not. The audit therefore records **partial recovery** and does not authorize prospective candidate construction.

- Historical modified images identified: 150 / 150
- Unique original hosts recovered: 150 / 150
- Train-split hosts verified: 150 / 150
- Insertion centers recovered: 150 / 150
- BBox geometry recovered: 150 / 150
- Medium-scale compliance verified: 150 / 150
- Orientation recovered: 0 / 150
- Historical training membership verified: 150 / 150
- Current pipeline compatible: 0 / 150 without limitation; 150 / 150 with an orientation-policy limitation

Scale summary (`sqrt(width*height)`): min=32.000000, median=40.515293, max=63.340350; violations=0.

## Overall flags

```text
HISTORICAL_HOST_SET_RECOVERED = TRUE
HISTORICAL_PLACEMENT_BBOX_GEOMETRY_RECOVERED = TRUE
HISTORICAL_ORIENTATION_RECOVERED = FALSE
ORIENTATION_POLICY_REQUIRES_AUTHOR_DECISION = TRUE
HISTORICAL_MEDIUM150_FRAME_RECOVERED = FALSE
PARTIAL_RECOVERY = TRUE
RECOVERED_FRAMES_CORE_HOST_CENTER_BBOX = 150
FULLY_RECOVERED_FRAMES_ALL_REQUIRED_FIELDS = 0
READY_FOR_NEW_HIGH_LOW_CANDIDATE_GATE_REVIEW = FALSE
HIGH_LOW_CANDIDATES_GENERATED = FALSE
TRAINING_UNLOCKED = FALSE
```

## Scientific rationale

Reusing the actual historical host and placement frame is a targeted deconfounding design: it would hold training context and recorded target-box geometry fixed while a future prospective phase changes representation support. This audit does not create that future efficacy result. It only establishes what can be inherited exactly. Missing per-frame orientation/source-cutout metadata is not repaired with visual inference, bbox-axis guesses, a new SegFormer placement, or a newly optimized value.

## Stop

No new High/Low candidates were generated. No detector was trained. No AP was evaluated or used in the provenance decision. No manuscript was revised. The partial frame manifest is evidence only and is not an authorized prospective input package.

Protocol deviation disclosure: during broad artifact discovery, the full retained `vessel_medium_v10_summary.json` was inadvertently printed, which exposed pre-existing detector/AP fields. Those fields were not parsed by this audit script, not used to identify the dataset, not used in any recovery decision, and no new AP evaluation occurred. The source-linkage file is retained in the inventory for path/hash provenance only.
