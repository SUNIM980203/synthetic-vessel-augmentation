# YOLO26m postprocess provenance addendum

## Scope and authorization

The user explicitly authorized evaluation and analysis on 2026-09-09 after disclosure of the transient-validation-infinity finding below. All 80 original accepted final checkpoints, including ordinal42, are retained in the prespecified xView test, HRSC2016-MS, and DIOR evaluation. No cell was excluded, retrained, retuned, or replaced by another checkpoint. This note does not change the frozen estimands, M1–M5 hierarchy, significance criteria, or claim boundaries.

## Known transient validation anomaly

Cell ordinal42, `unity_medium150__head_only__lr2e-4__s20260728`, has `val/cls_loss=inf` at training epochs10,12,14. The final epoch CSV values are finite. The accepted final checkpoint SHA-256 is `a4d4cebbe2e33bd0c8f34e906067ea7d1d0240e528cbe37a0619d0d485b884e6`; all768 final model-state tensors (21,836,914 elements) were checked on CPU and contain no NaN or infinity. This is not proof of a resolved cause or a generally error-free training trajectory. The original CSV, final checkpoint, training ledger and review record remain unchanged. The finding must not be hidden by the training ledger's structural PASS or by a later numerical consistency PASS.

Evidence: `16_audits/yolo26m_training_final_reaudit_20260909.json`, `16_audits/YOLO26M_POSTPROCESS_REVIEW_HOLD_20260909.md`, and `16_audits/yolo26m_postprocess_20260909/user_authorization.json`.

## Frozen analysis execution

The unchanged `tools/run_yolo26m_factorial_postprocess.py`, SHA-256 `8b1cb0407ec051369eaa41a7471bd6f032ce1e2f50d32fe0a40182dc6dd48545`, was started exactly once on 2026-09-09 at11:56:16 KST (02:56:16.773740 UTC). External startup controls only constrain CPU threads4, interop1, affinity0–9, and BelowNormal priority; they do not modify model, validator, metric, RNG, precision flags, or evaluation kwargs. Resource bootstrap and launcher hashes, frozen script hashes, and pre-run v48-tree hashes are retained in `16_audits/yolo26m_postprocess_20260909/launch.json`.

This addendum is written at the start of evaluation and is not a final acceptance assertion. Final completion still requires the frozen evaluation/analysis/report pipeline and independent numerical, claim, prediction-retention, and protected-artifact audits to pass. The generated final report must reference this addendum. The v48 manuscript and supplement are not edited.
