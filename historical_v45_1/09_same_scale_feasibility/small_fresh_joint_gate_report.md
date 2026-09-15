# One-shot untouched Small joint gate report

The frozen Medium-validated intervention did not satisfy all prespecified feasibility criteria in the Small scale stratum.

Medium에서 검증된 동결 개입은 Small scale에서 사전 명시된 모든 feasibility 기준을 충족하지 못했다.

The frozen S3 MILP returned status 2 (INFEASIBLE) and selected 0/30 pairs. Accordingly, every post-selection scientific gate below is formally FAIL under the prespecified completion rule, but the corresponding aggregate is not estimable; this is not evidence that a measured Weber, support, appearance-balance, scale, geometry, or placement value exceeded its threshold.

동결 S3 MILP가 status 2(INFEASIBLE)를 반환하여 30개 쌍 중 0개만 선택되었다. 따라서 아래 post-selection 과학 gate는 사전 명시된 완료 규칙에 따라 정식으로 FAIL이지만, 각 aggregate는 추정 불가능하다. 이는 측정된 Weber·support·appearance balance·scale·geometry·placement 값이 임계치를 초과했다는 증거가 아니다.

Primary failure classification: **G. Joint frozen-S3 assignment infeasibility**. The binding constituent constraint was not isolated because post-result ablation, retry, and tuning were prohibited.

## Scientific gates

| gate | status |
|---|---|
| SUPPORT_GATE | FAIL |
| ABSOLUTE_WEBER_GATE | FAIL |
| OTHER_RADIOMETRY_GATE | FAIL |
| PAIRWISE_RADIOMETRY_GATE | FAIL |
| APPEARANCE_BALANCE_GATE | FAIL |
| SCALE_GATE | FAIL |
| GEOMETRY_GATE | FAIL |
| PLACEMENT_GATE | FAIL |

- Frozen selector did not produce a complete 30-host assignment.
- Appearance-family TVD: not estimable (the stored 0.000000 is the empty-set convention, not a balance result).

## Integrity gates

| gate | status |
|---|---|
| CANDIDATE_COUNT_GATE | PASS |
| UID_UNIQUENESS_GATE | PASS |
| FINAL_IMAGE_UNIQUENESS_GATE | PASS |
| CONTENT_OVERLAP_GATE | PASS |
| SELECTED_INTERVENTION_INTEGRITY_GATE | FAIL |
| HOST_SCENE_CONTAMINATION_GATE | PASS |

- Raw/retained counts: 62400/61440.
- Prior UID/RNG/final-image overlaps: 0/0/0.
- Selected UID/final-image uniqueness: 0/60 and 0/60.

ALL_SMALL_FEASIBILITY_GATES = FAIL

SMALL_FEASIBILITY_READY = FALSE

SUPPORT_SCALE_2X2_READY_FOR_PROTOCOL_REVIEW = FALSE

TRAINING_UNLOCKED = FALSE

DETECTOR_TRAINING_RUNS = 0

AP_VALUES_INSPECTED = 0

DETECTOR_PREDICTIONS_GENERATED = 0

EFFICACY_CHECKPOINTS_CREATED = 0

SMALL_BRANCH_TUNING_ITERATIONS = 0

This is feasibility and pipeline-integrity evidence only. No detector efficacy, AP interaction, causality, or scale independence has been established.
