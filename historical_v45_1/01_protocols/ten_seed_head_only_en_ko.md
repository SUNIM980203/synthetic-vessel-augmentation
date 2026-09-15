# Ten-Seed Head-Only Robustness Expansion v46 / 10개 시드 Head-Only 강건성 확장 v46

Frozen locally on 2026-08-11 before any v46 training or evaluation result was observed.

## English

### Purpose

The original paper comparisons used five paired optimization seeds. V46 adds five previously unused seeds (20260728–20260732) to RealOnly-Standard, Duplicate, RealCutout Medium150, and Unity Medium150. The original five seeds remain unchanged, producing a ten-seed paired sensitivity analysis. This expansion estimates robustness and precision; it does not retroactively alter the frozen v13–v15 decisions.

### Locked training design

All four conditions start from the same checkpoint SHA-256 and use the existing immutable training datasets. Every new run uses modules 0–22 frozen, Detect module 23 trainable, 20 complete epochs, exact `last.pt`, 640-pixel inputs, batch 16, workers 0, AdamW, `lr0=1e-4`, `lrf=0.1`, 0.5 warm-up epochs, deterministic execution, and the previously used mosaic/mixup/copy-paste settings. Dataset YAML and image-plus-label tree hashes are stored in `config/seed_expansion_v46.json`. An incomplete run directory is never overwritten.

### Evaluation and statistics

The exact epoch-20 checkpoints are evaluated on the unchanged xView test split, HRSC2016-MS official test identifiers, and the fixed DIOR public-mirror split with the existing evaluation parameters. Primary contrasts are Unity–Duplicate and Unity–RealCutout. Unity–RealOnly, Duplicate–RealOnly, and RealCutout–RealOnly are secondary. Primary reporting uses paired two-sided Student-t 95% confidence intervals across all ten seeds, with every per-seed AP50-95/AP50/AP75 value, effect size, paired p-value, positive-seed count, and exact two-sided sign-test sensitivity retained.

### Interpretation boundary

RealOnly has half the nominal optimizer updates of the 3,000-image conditions and is not an exposure-matched causal control. The experiment tests the stability of the observed head-only YOLO result; it does not establish detector-general efficacy, feature support as an independent cause or mediator, or state-of-the-art performance. HRSC and DIOR are two external datasets showing directional transport under fixed evaluation settings, not independent preregistrations.

## 한국어

### 목적

기존 원고의 주요 비교는 5개 대응 최적화 시드를 사용했다. v46은 사용하지 않았던 5개 시드(20260728–20260732)를 RealOnly-Standard, Duplicate, RealCutout Medium150, Unity Medium150에 동일하게 추가한다. 기존 5개 시드는 변경하지 않으며 총 10개 시드의 대응 민감도 분석을 구성한다. 이 확장은 결과의 강건성과 추정 정밀도를 평가하는 것이며 고정된 v13–v15 결정을 소급 변경하지 않는다.

### 고정 학습 설계

네 조건은 모두 동일한 SHA-256의 초기 체크포인트와 기존의 고정 학습 데이터셋을 사용한다. 새 학습은 모듈 0–22를 고정하고 Detect 모듈 23만 학습하며, 20 epoch, 정확한 `last.pt`, 입력 640, batch 16, workers 0, AdamW, `lr0=1e-4`, `lrf=0.1`, warm-up 0.5 epoch, 결정론적 실행, 기존 mosaic/mixup/copy-paste 설정을 그대로 사용한다. 데이터 YAML과 영상·라벨 트리 해시는 `config/seed_expansion_v46.json`에 기록한다. 불완전한 실행 폴더는 덮어쓰지 않는다.

### 평가와 통계

정확한 epoch-20 체크포인트를 기존 xView test, HRSC2016-MS 공식 test ID, 고정 DIOR 공개 미러 분할에서 동일한 평가 설정으로 측정한다. 주 비교는 Unity–Duplicate와 Unity–RealCutout이다. Unity–RealOnly, Duplicate–RealOnly, RealCutout–RealOnly는 보조 비교다. 주 보고는 총 10개 시드의 대응 양측 Student-t 95% 신뢰구간이며, 모든 시드의 AP50-95/AP50/AP75, 효과크기, 대응 p값, 양의 방향 시드 수, 정확 양측 부호검정 민감도를 함께 공개한다.

### 해석 범위

RealOnly는 3,000장 조건보다 명목 optimizer update 수가 절반이므로 노출량이 일치한 인과 대조군이 아니다. 본 실험은 관찰된 head-only YOLO 결과의 안정성을 검토한다. 검출기 일반 효능, feature support의 독립 인과효과나 매개효과, 최고성능을 주장하지 않는다. HRSC와 DIOR은 고정 평가 설정에서 같은 방향의 전달이 관찰되는 두 외부 데이터셋이며 독립 사전등록 재현으로 표현하지 않는다.
