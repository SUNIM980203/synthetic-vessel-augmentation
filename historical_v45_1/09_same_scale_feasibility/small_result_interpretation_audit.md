# Independent Small-result interpretation audit

The frozen machine result is preserved. This audit changes no host, candidate, support distance, selector input, threshold, solver setting, or gate value.

- The exact protocol, precommit, runner, host-selector, and host-manifest hashes match their frozen records.
- All 30 NEVER_USED hosts passed contamination checks.
- Candidate dose and integrity passed: 62,400 raw; 61,440 retained; exactly 2,048 retained per host; retained UID, RNG64, and final-image uniqueness 61,440/61,440; prior overlaps 0.
- Candidate-level support was computed for all 61,440 retained images.
- All 30 hosts had pair-level feasible options (120--821 per host; 1,376,129 full eligible pairs; 11,577 deterministic optimizer representatives).
- The unchanged global S3 MILP returned status 2, INFEASIBLE, and selected 0/30 pairs.
- Therefore the prespecified post-selection support, Weber, other-radiometry, appearance-balance, scale, geometry, and placement aggregates are not estimable. Their formal FAIL status follows the completion rule and must not be described as separate measured threshold exceedances.
- The primary outcome classification is **G. Joint frozen-S3 assignment infeasibility**. The binding constituent constraint was not isolated because post-result ablation, retry, and tuning were prohibited.
- `SMALL_FEASIBILITY_READY = FALSE`, `SUPPORT_SCALE_2X2_READY_FOR_PROTOCOL_REVIEW = FALSE`, and `TRAINING_UNLOCKED = FALSE`.
- Detector training, predictions, checkpoints, and AP inspection remain zero.
