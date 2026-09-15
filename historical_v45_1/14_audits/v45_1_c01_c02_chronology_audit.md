# v45.1 C01/C02 Chronology Audit

## Provenance basis

The original support values are read from the retained machine record `AnalysisResults/expanded_v1/vessel_feature_support_v10/comparison.json` (SHA-256 `9837fe21b88d882fad1ae554e1c4c7c5b97924c531909d4cea5d6b00daf2fb12`). Its retained timestamp is 2026-07-28 14:31:56 KST. The record states `test_split_used: false`; its companion report states that only real training objects and synthetic training objects were used and that no validation or test predictions were involved.

The retained support protocol (`docs/vessel_feature_support_protocol.md`, SHA-256 `78a47b2d1662d56975c940ae6ee3cbbbb155ebef9fa4cd352eea1bcc40864f84`) defines a two-stage order: the training-feature gate first, detector validation only for the feature-gate winner, and the test split locked until validation replication.

The first retained downstream head-only summary (`AnalysisResults/expanded_v1/vessel_head_only_v11_summary.json`, SHA-256 `ddd2fbe3174764eda381834086f9b07d15bf3b3028de3825ae2f5bcffbe4d7ab`) is timestamped 2026-07-29 12:10:17 KST. The corresponding head-only run directories begin on 2026-07-29, after the support record.

## Verified ordering

1. Training-only feature-support screening: 2026-07-28.
2. Medium150 selected as the support-gate winner under the retained protocol.
3. Downstream head-only detector training and AP evaluation: 2026-07-29 onward.

The evidence supports a configuration/screening chronology, not a post-result, post-lock, confirmatory-efficacy, or causal classification.

`C01_C02_CLASSIFICATION_CORRECTED = TRUE`

Correct class: `CONFIGURATION / SCREENING — TRAINING-ONLY; DOWNSTREAM AP UNOPENED`.
