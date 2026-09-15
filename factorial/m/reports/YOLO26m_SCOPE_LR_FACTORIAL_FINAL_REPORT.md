# YOLO26m Scope × LR Factorial — Final Scientific Report

## Completion provenance and unresolved limitation

This package was completed by an explicitly user-authorized report-only reconciliation on 2026-09-09. The original frozen postprocess ran once: evaluation and analysis stages 1-7 completed, but stage 8 stopped at a historical STOP-marker gate before writing its audit/report outputs. The original STOP markers and stderr remain unchanged; this is not an uninterrupted original-pipeline PASS. Training, evaluation, statistical analysis, figure generation, model selection, and retuning were not rerun. The derived finalizer retains the frozen scientific/report body, with an explicit exact-history gate and this disclosure.

Ordinal 42 (unity_medium150__head_only__lr2e-4__s20260728) had val/cls_loss=inf at epochs 10, 12, and 14. All 80 prespecified final checkpoints were retained by user authorization, without exclusion or replacement. Its final epoch values and all 768 final model-state tensors were verified finite in the preserved review. The cause remains unresolved: structural/numerical audit PASS is not evidence of an error-free trajectory. The interpretation is conditional on this retained dataset and must undergo scientific review before manuscript use.

See [original provenance addendum](YOLO26m_POSTPROCESS_PROVENANCE_ADDENDUM.md) and [reconciliation audit](../16_audits/yolo26m_finalization_reconciliation_20260909_v2/reconciliation_audit.md).

## 1. Scientific question

How does the Unity-minus-Duplicate × adaptation-scope interaction behave across YOLO26n, YOLO26s, and the prospectively frozen YOLO26m third-model-size follow-up?

## 2. Frozen design

Duplicate/Unity Medium150 × head-only/full-network × LR 1e-4/2e-4 × ten paired seeds; 80 accepted fresh cells, batch 16, AdamW, 20 epochs, imgsz 640, deterministic mode, no early stopping, no resumed accepted run, and epoch-20 `last.pt` only. Schedule RNG seed: 20260831.

## 3. Architecture and semantic scope audit

Architecture PASS: 24 top-level modules and 21,780,598 five-class parameters. Semantic head-only PASS: only Detect module 23 is trainable (2,806,326 parameters); full-network has 21,780,598 trainable parameters.

## 4. Initial checkpoint

All cells began from `yolo26m_xview_initial_locked.pt` (selected epoch 28, metrics/mAP50-95(B)=0.189580000, SHA-256 `a8b5539273c0f416abb195bec923f0c41a343a5a4a514919eaccb50dbbe64c23`). Selection used the frozen historical validation-fitness rule and no factorial outcome.

## 5. Run integrity

ACCEPTED = 80; ACCEPTED_RESUMED = 0; INVALID_ACCEPTED = 0. Every accepted run was fresh, uninterrupted, exactly 20 rows, `resume=False`, and traceable to the single locked initial checkpoint. All 240 evaluations passed.

## 6. Absolute performance

Complete YOLO26m and combined n/s/m AP50–95/AP50/AP75 summaries are in `yolo26m_factorial_absolute_AP.csv` and `yolo26_n_s_m_absolute_AP.csv`.

| Scope | LR | Condition | xView vessel AP50–95 mean ± SD |
|---|---:|---|---:|
| full_network | 1e-4 | duplicate | 0.126277 ± 0.012802 |
| full_network | 1e-4 | unity_medium150 | 0.120695 ± 0.012014 |
| full_network | 2e-4 | duplicate | 0.117443 ± 0.009216 |
| full_network | 2e-4 | unity_medium150 | 0.115937 ± 0.014006 |
| head_only | 1e-4 | duplicate | 0.159636 ± 0.002047 |
| head_only | 1e-4 | unity_medium150 | 0.162290 ± 0.003204 |
| head_only | 2e-4 | duplicate | 0.154303 ± 0.003975 |
| head_only | 2e-4 | unity_medium150 | 0.149497 ± 0.026016 |

## 7. Fixed-LR Unity effects

- head_only @ 1e-4: +0.002655 (95% CI [+0.000702, +0.004607]; 8/10 positive).
- head_only @ 2e-4: -0.004806 (95% CI [-0.022140, +0.012528]; 7/10 positive).
- full_network @ 1e-4: -0.005582 (95% CI [-0.009464, -0.001701]; 1/10 positive).
- full_network @ 2e-4: -0.001506 (95% CI [-0.010820, +0.007808]; 5/10 positive).

## 8. Fixed-LR DiDs

- 1e-4: +0.008237 (95% CI [+0.003346, +0.013128]; 9/10 positive); t p=0.0041545; sign p=0.0214844.
- 2e-4: -0.003300 (95% CI [-0.024793, +0.018194]; 6/10 positive); t p=0.736363; sign p=0.753906.

## 9. Marginal DiD

YOLO26m designated endpoint: +0.002469 (95% CI [-0.009934, +0.014871]; 7/10 positive); SD=0.017337; median=+0.005865; range=[-0.038920, +0.017092]; t p=0.663175; exact sign p=0.34375. The ten paired values are retained in the statistics JSON/CSV.

## 10. LR moderation

ThreeWay = DiD(2e-4) − DiD(1e-4): -0.011536 (95% CI [-0.030418, +0.007345]; 3/10 positive). This is secondary and refers only to the two frozen LRs.

## 11. xView scene sensitivity

The 2,000-replicate scene bootstrap (RNG 20260813) yielded percentile interval [-0.008097, +0.005040], positive fraction 0.7885, and 7 zero-target draws. LOSO: 8/9 positive, range [-0.000749, +0.003237]. Only four of nine scenes contain vessel targets, so this is sparse-cluster sensitivity, not a population CI.

## 12. HRSC transfer

HRSC2016-MS vessel AP50–95 Marginal DiD: +0.004797 (95% CI [-0.001590, +0.011185]; 8/10 positive); fixed-LR DiDs: 1e-4 +0.006188 (95% CI [+0.003898, +0.008477]; 9/10 positive), 2e-4 +0.003407 (95% CI [-0.008662, +0.015476]; 8/10 positive); ThreeWay -0.002781 (95% CI [-0.014554, +0.008992]; 6/10 positive). No target-domain tuning was used.

## 13. DIOR transfer

DIOR vessel AP50–95 Marginal DiD: +0.031376 (95% CI [+0.024381, +0.038370]; 10/10 positive); fixed-LR DiDs: 1e-4 +0.031493 (95% CI [+0.029967, +0.033019]; 10/10 positive), 2e-4 +0.031258 (95% CI [+0.018331, +0.044186]; 9/10 positive); ThreeWay -0.000234 (95% CI [-0.012203, +0.011734]; 8/10 positive). No target-domain tuning was used.

## 14. YOLO26n comparison

YOLO26n Marginal DiD: +0.012647 (95% CI [+0.009721, +0.015574]; 10/10 positive). YOLO26m: +0.002469 (95% CI [-0.009934, +0.014871]; 7/10 positive). Paired n − m difference: +0.010179 (95% CI [-0.001177, +0.021534]; 7/10 positive). This comparison is descriptive, not causal.

## 15. YOLO26s comparison

YOLO26s Marginal DiD: -0.001205 (95% CI [-0.007689, +0.005279]; 4/10 positive). YOLO26m: +0.002469 (95% CI [-0.009934, +0.014871]; 7/10 positive). Paired s − m difference: -0.003673 (95% CI [-0.019573, +0.012226]; 4/10 positive). This comparison is descriptive, not causal.

## 16. Three-model pattern

n: +0.012647 (95% CI [+0.009721, +0.015574]; 10/10 positive); s: -0.001205 (95% CI [-0.007689, +0.005279]; 4/10 positive); m: +0.002469 (95% CI [-0.009934, +0.014871]; 7/10 positive). Paired n − s: +0.013852 (95% CI [+0.005595, +0.022109]; 8/10 positive); n − m: +0.010179 (95% CI [-0.001177, +0.021534]; 7/10 positive); s − m: -0.003673 (95% CI [-0.019573, +0.012226]; 4/10 positive). No pooled overall effect or monotonic size regression was fitted.

## 17. Predefined interpretation classification

**CASE M3** under the frozen M5 → M1 → M2 → M3 → M4 hierarchy. The YOLO26m interaction is near zero or statistically inconclusive under the frozen seed-level interval.

## 18. What is supported

The evidence supports the frozen CASE M3 interpretation only for YOLO26n/s/m, the retained datasets, two fixed LRs, ten paired seeds, and this 20-epoch recipe.

## 19. What is not supported

The result does not establish universal YOLO behavior, a universal or causal model-size effect, a monotonic capacity mechanism, a freezing mechanism, detector-general behavior, scene-population generalization, or support causality. It is not an independent untouched confirmation.

## 20. Recommended manuscript implication

Describe attenuation or an inconclusive medium-model interaction; avoid selecting either prior model as universally representative.

The v48 manuscript, supplement, and figures were not modified. Stop here and wait for scientific review; do not automatically start YOLO26l, SupportHigh/Low, dose response, a new LR/freeze grid, or another detector.
