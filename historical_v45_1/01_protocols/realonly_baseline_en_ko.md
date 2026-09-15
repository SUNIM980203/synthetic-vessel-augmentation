# RealOnly Standard-Baseline Addendum v45 / RealOnly 표준 기준선 추가실험 v45

Frozen locally on 2026-08-11 before any v45 training or evaluation result was observed. This file is a local execution freeze, not a public preregistration.

## English

### Purpose

The existing Duplicate condition controls the 3,000-image optimization exposure used by Unity Medium150 and RealCutout Medium150, but it does not show how any of those conditions compare with the ordinary 1,500-image real-only training recipe. V45 adds that missing practical baseline.

### Design

- Initialization: the same locked checkpoint used by the existing head-only study, SHA-256 `7aab2bd4aebb6181df0350c80883e0c1b2615f363481649d579a23b0e4a7140f`.
- Training data: the original 1,500 scene-disjoint xView real training patches only.
- Seeds: 20260723-20260727, paired with the five existing conditions.
- Adaptation: YOLO26n modules 0-22 frozen; Detect module 23 trainable.
- Recipe: exact epoch 20 `last.pt`, 640 px, batch 16, AdamW, lr0 1e-4, lrf 0.1, warmup 0.5 epoch, mosaic 0.5, deterministic execution.
- Evaluation: xView held-out test, HRSC2016-MS, and the fixed DIOR public-mirror split. No v45 result is used for model or checkpoint selection.

### Interpretation guard

RealOnly-Standard contains 1,500 samples per epoch, whereas Duplicate, Unity, and RealCutout contain 3,000. It is therefore a practical standard-training baseline, not an exposure-matched causal control. The contrasts have distinct estimands:

- Unity - RealOnly: net practical effect of the augmentation recipe, including extra optimization exposure.
- Duplicate - RealOnly: effect associated with additional repeated exposure and extra optimizer updates.
- Unity - Duplicate: content-specific Unity effect at matched nominal exposure.
- Unity - RealCutout: source-appearance contrast at matched host, position, box geometry, and nominal exposure.

### Statistics

The primary presentation uses paired mean differences and two-sided Student-t 95% confidence intervals. One-sided results remain historical supplementary analyses only. All five per-seed values, sign consistency, and the exact two-sided sign-test sensitivity are retained. With n=5, the smallest attainable exact two-sided sign-test p-value is 0.0625.

V45 has no pass/fail efficacy gate and cannot alter the frozen conclusions of v13-v15. It may clarify whether the Unity recipe is above the ordinary real-only baseline and whether Duplicate itself improves or degrades performance.

## 한국어

### 목적

기존 Duplicate 조건은 Unity Medium150 및 RealCutout Medium150과 동일한 3,000장 학습 노출을 통제하지만, 이 조건들이 통상적인 실제 영상 1,500장 학습보다 나은지는 보여주지 않는다. v45는 이 실용적 기준선을 추가한다.

### 설계

- 초기화: 기존 head-only 연구와 동일한 고정 checkpoint를 사용한다.
- 학습 데이터: 장면이 분리된 원래 xView 실제 훈련 patch 1,500장만 사용한다.
- seed: 기존 다섯 조건과 짝지을 수 있도록 20260723-20260727을 사용한다.
- 적응 범위: YOLO26n modules 0-22를 고정하고 Detect module 23만 학습한다.
- recipe: 정확한 epoch 20 `last.pt`, 640 px, batch 16, AdamW, lr0 1e-4, lrf 0.1, warmup 0.5 epoch, mosaic 0.5, deterministic 실행이다.
- 평가: xView held-out test, HRSC2016-MS 및 고정 DIOR public-mirror split을 사용한다. v45 결과는 모델이나 checkpoint 선택에 사용하지 않는다.

### 해석 제한

RealOnly-Standard는 epoch당 1,500장이고 Duplicate, Unity 및 RealCutout은 3,000장이다. 따라서 RealOnly는 노출량이 일치하는 인과 통제군이 아니라 통상 학습 기준선이다.

- Unity - RealOnly: 추가 학습 노출을 포함한 전체 증강 recipe의 실용적 효과.
- Duplicate - RealOnly: 반복 노출과 optimizer update 증가에 연관된 효과.
- Unity - Duplicate: 명목 노출량을 맞춘 Unity content-specific 효과.
- Unity - RealCutout: host·위치·box geometry·명목 노출량을 맞춘 출처 외관 차이.

### 통계

주 분석은 paired mean difference와 양측 Student-t 95% 신뢰구간으로 제시한다. 기존 one-sided 결과는 역사적 보조 분석으로만 유지한다. 다섯 per-seed 값, 방향 일관성, 정확 양측 sign-test 민감도 결과를 모두 보존한다. n=5에서 정확 양측 sign-test의 최소 p값은 0.0625이다.

v45에는 효능 PASS/FAIL gate를 두지 않으며 v13-v15의 동결 판정을 변경하지 않는다. 이 실험의 역할은 Unity recipe가 통상 real-only 학습보다 높은지, 그리고 Duplicate 자체가 성능을 높였는지 또는 낮췄는지를 분리해 기술하는 것이다.
