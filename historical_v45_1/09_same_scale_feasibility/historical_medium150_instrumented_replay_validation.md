# Historical Medium150 instrumented replay validation

`INSTRUMENTATION_OUTPUT_PRESERVING = TRUE`

- COCO canonical == baseline == run A == run B: `TRUE`.
- Summary canonical == baseline == run A == run B: `TRUE`.
- Images canonical == baseline == run A == run B: 1500/1500.
- Historical modified subset identical: 150/150.
- Accepted direct selection rows: 150.
- Any output-image mismatch: 0.

The full per-image comparison is in `historical_medium150_replay_hash_comparison.csv`.
