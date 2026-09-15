# v48 Adaptation-Scope Fairness Protocol

Frozen on 2026-08-11 before creating the v48 inner split or training a v48 checkpoint.

## English

### Question

Does the head-only versus full-network Unity-effect interaction remain after each scope receives an independently selected learning rate and epoch budget?

### Leakage-safe development

Create a deterministic source-scene-disjoint inner split from the existing 1,500 real training patches. The public validation, test, HRSC2016-MS, and DIOR data are not used for tuning. For each scope, tune on the Duplicate exposure only; Unity results cannot influence hyperparameter selection.

- Head only: `freeze=23`, learning rate in `{5e-5, 1e-4, 2e-4}`, epochs in `{10,20,30}`.
- Full network: all modules trainable, learning rate in `{1e-5, 3e-5, 1e-4}`, epochs in `{10,20,30}`.
- Development seed: `20260819`; AdamW, `lrf=0.1`, warm-up 0.5 epoch, batch 16, image size 640, workers 0.
- Select the highest final-epoch inner-validation aggregate mAP50-95 separately within each scope. Ties are resolved by fewer epochs, lower learning rate, then lexical run name.

### Confirmation

After both settings are frozen, train Duplicate and Unity Medium150 from the same initialization with ten paired seeds `20260723` through `20260732`. Use the selected epoch budget and exact final `last.pt`; never select a confirmation checkpoint by validation performance.

The primary endpoint is the two-sided 95% interval for the paired vessel-AP difference-in-differences:

`(Unity - Duplicate)_head - (Unity - Duplicate)_full`.

Aggregate AP50-95, AP50, AP75, per-class AP, positive-sign counts, optimizer updates, exposure counts, and training time are secondary. xView validation is primary for this sensitivity experiment; held-out xView test and the two external datasets are opened only after the validation result is frozen.

The experiment assesses robustness to scope-appropriate optimization. It does not prove that the tested grid contains a universal optimum.

## 한국어

### 질문

head-only와 full-network 각각에 적절한 학습률과 epoch를 독립적으로 선택한 뒤에도 Unity 효과의 adaptation-scope 상호작용이 유지되는가?

### 누수 방지 개발

기존 실제 학습 patch 1,500개에서 원본 scene 단위 내부 분할을 결정론적으로 만든다. 기존 validation, test, HRSC2016-MS 및 DIOR는 튜닝에 사용하지 않는다. 각 scope의 설정은 Duplicate 조건만으로 선택하며 Unity 결과는 설정 선택에 사용할 수 없다.

- Head only: `freeze=23`, 학습률 `{5e-5, 1e-4, 2e-4}`, epoch `{10,20,30}`.
- Full network: 전체 모듈 학습, 학습률 `{1e-5, 3e-5, 1e-4}`, epoch `{10,20,30}`.
- 개발 seed `20260819`; AdamW, `lrf=0.1`, warm-up 0.5 epoch, batch 16, image 640, workers 0.
- scope별 마지막 epoch 내부 validation aggregate mAP50-95가 가장 높은 설정을 선택한다. 동률이면 epoch가 적은 설정, 낮은 학습률, run name 순으로 결정한다.

### 확인 실험

두 설정을 동결한 후 동일 초기화에서 Duplicate와 Unity Medium150을 시드 `20260723`부터 `20260732`까지 10개 paired seed로 학습한다. 선택된 epoch의 정확한 `last.pt`만 사용하며 확인 단계에서 validation 성능으로 checkpoint를 선택하지 않는다.

주 분석은 vessel AP difference-in-differences의 대응 양측 95% 신뢰구간이다. 이 실험은 scope별 최적화 공정성에 대한 강건성을 검증하지만 탐색 격자가 보편적 최적값을 포함한다고 주장하지 않는다.
