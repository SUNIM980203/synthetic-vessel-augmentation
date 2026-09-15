# Final untouched Medium joint gate report

After prospectively separating candidate identity from RNG state and enforcing deterministic final-image uniqueness, the frozen balanced selector reproduced the prespecified same-scale support, native-radiometry, appearance-balance, and geometry criteria on an untouched Medium-scale cohort with no candidate-identity or content-overlap violations.

후보 신원과 RNG 상태를 분리하고 최종 이미지 중복 제거를 사전 동결한 뒤, 미사용 Medium cohort에서 과학·파이프라인 무결성 기준을 모두 충족했다. 이는 detector 효능이나 Small-scale 일반화를 증명하지 않는다.

## A. Scientific feasibility

| gate | status |
|---|---|
| SUPPORT_GATE | PASS |
| ABSOLUTE_WEBER_GATE | PASS |
| OTHER_RADIOMETRY_GATE | PASS |
| PAIRWISE_RADIOMETRY_GATE | PASS |
| APPEARANCE_BALANCE_GATE | PASS |
| SCALE_GATE | PASS |
| GEOMETRY_GATE | PASS |
| PLACEMENT_GATE | PASS |

- Median support gap: **0.096082** (required >=0.060).
- Fraction gap >=0.040: **0.8667** (required >=0.80).
- Minimum support gap: **0.005345** (required >=0.005).
- High/Low absolute Weber standardized Wasserstein: 0.482764/0.483451.
- Appearance-family TVD: 0.000000.

## B. Pipeline integrity

| gate | status |
|---|---|
| CANDIDATE_UNIQUENESS_GATE | PASS |
| UID_GATE | PASS |
| RNG_COLLISION_GATE | PASS |
| CONTENT_CONTAMINATION_GATE | PASS |
| SELECTED_INTERVENTION_INTEGRITY_GATE | PASS |

- Raw/retained counts: 62400/61440.
- Prior UID/RNG/final-image overlaps: 0/0/0.
- Selected UID/final-image uniqueness: 60/60 and 60/60.

ALL_MEDIUM_FEASIBILITY_GATES = PASS

MEDIUM_FEASIBILITY_READY = TRUE

TRAINING_READY_FOR_REVIEW = TRUE

TRAINING_UNLOCKED = FALSE

TRAINING_LOCKED = TRUE

SMALL_BRANCH_OPENED = FALSE

DETECTOR_TRAINING_RUNS = 0

AP_VALUES_INSPECTED = 0

DETECTOR_PREDICTIONS_GENERATED = 0

EFFICACY_CHECKPOINTS_CREATED = 0

This phase establishes only Medium-scale intervention feasibility and publication-grade pipeline integrity. It does not establish an AP effect, causality, universal scale independence, Small-scale feasibility, or cross-detector generalization.
