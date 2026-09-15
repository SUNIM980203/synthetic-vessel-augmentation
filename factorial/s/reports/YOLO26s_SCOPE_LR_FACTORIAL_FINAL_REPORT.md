# YOLO26s Scope × LR Factorial — Final Scientific Report

## 1. Scientific question

Does the positive Unity-augmentation × adaptation-scope interaction observed in YOLO26n persist in the larger YOLO26s model under the same datasets, two LRs, paired seeds, and semantic adaptation scopes?

## 2. Why this experiment was run

The study addresses model-size transfer within the YOLO26 family. Prior YOLO26n results were known, so this is a prospectively frozen within-study replication rather than independent confirmation.

## 3. YOLO26s architecture audit

The exact YOLO26s graph contains 24 top-level modules and 9,951,734 parameters for five classes. The retained architecture audit enumerates backbone, neck, Detect modules, and every trainable tensor.

## 4. Head-only semantic-equivalence audit

Both YOLO26n and YOLO26s head-only conditions freeze modules 0–22 and train only Detect module 23. No backbone or neck tensor is trainable. Semantic equivalence: **PASS**.

## 5. Fixed xView starting checkpoint

All cells initialized from one validation-selected xView YOLO26s checkpoint: `yolo26s_xview_initial_locked.pt`, selected epoch 12, validation metric 0.170550, SHA-256 `cc0e9af1c3d2f6c2bfc39b906f44c0113e525b538410247bf0ab19db3b3a26d1`.

## 6. Frozen factorial design

Duplicate/Unity Medium150 × head-only/full-network × LR 1e-4/2e-4 × ten paired seeds yielded 80 fresh cells. Batch 16, 20 epochs, AdamW, imgsz 640, deterministic mode, epoch-20 `last.pt`, and the 3,000-images/epoch exposure were fixed before outcomes.

## 7. Run-integrity results

ACCEPTED_CELLS = 80; ACCEPTED_RESUMED_CELLS = 0; INVALID_ACCEPTED_CELLS = 0. All 240 domain evaluations and prediction archives passed.

## 8. Absolute performance

The complete Dataset × Scope × LR × Condition summaries for AP50–95/AP50/AP75 are in `yolo26s_factorial_absolute_AP.csv` and the combined YOLO26n/YOLO26s table. Selected vessel AP50–95 rows:

| Dataset | Scope | LR | Condition | Mean ± SD |
|---|---|---:|---|---:|
| dior_public_mirror | full_network | 1e-4 | duplicate | 0.014968 ± 0.003873 |
| dior_public_mirror | full_network | 1e-4 | unity_medium150 | 0.018601 ± 0.003197 |
| dior_public_mirror | full_network | 2e-4 | duplicate | 0.014645 ± 0.002709 |
| dior_public_mirror | full_network | 2e-4 | unity_medium150 | 0.017965 ± 0.002704 |
| dior_public_mirror | head_only | 1e-4 | duplicate | 0.021892 ± 0.001920 |
| dior_public_mirror | head_only | 1e-4 | unity_medium150 | 0.035834 ± 0.003176 |
| dior_public_mirror | head_only | 2e-4 | duplicate | 0.024397 ± 0.001869 |
| dior_public_mirror | head_only | 2e-4 | unity_medium150 | 0.042343 ± 0.002873 |
| hrsc2016_ms | full_network | 1e-4 | duplicate | 0.156087 ± 0.006580 |
| hrsc2016_ms | full_network | 1e-4 | unity_medium150 | 0.158183 ± 0.007754 |
| hrsc2016_ms | full_network | 2e-4 | duplicate | 0.141393 ± 0.007972 |
| hrsc2016_ms | full_network | 2e-4 | unity_medium150 | 0.144003 ± 0.005867 |
| hrsc2016_ms | head_only | 1e-4 | duplicate | 0.175506 ± 0.002601 |
| hrsc2016_ms | head_only | 1e-4 | unity_medium150 | 0.183780 ± 0.003260 |
| hrsc2016_ms | head_only | 2e-4 | duplicate | 0.177401 ± 0.002295 |
| hrsc2016_ms | head_only | 2e-4 | unity_medium150 | 0.184222 ± 0.002307 |
| xview_test | full_network | 1e-4 | duplicate | 0.103403 ± 0.011497 |
| xview_test | full_network | 1e-4 | unity_medium150 | 0.101303 ± 0.012379 |
| xview_test | full_network | 2e-4 | duplicate | 0.081300 ± 0.010700 |
| xview_test | full_network | 2e-4 | unity_medium150 | 0.080015 ± 0.009288 |
| xview_test | head_only | 1e-4 | duplicate | 0.091978 ± 0.003330 |
| xview_test | head_only | 1e-4 | unity_medium150 | 0.088822 ± 0.004240 |
| xview_test | head_only | 2e-4 | duplicate | 0.099624 ± 0.003384 |
| xview_test | head_only | 2e-4 | unity_medium150 | 0.096985 ± 0.006136 |

## 9. xView Unity effects by scope/LR

- head_only @ 1e-4: -0.003156 (95% CI [-0.005691, -0.000620]; 2/10 positive).
- head_only @ 2e-4: -0.002639 (95% CI [-0.005902, +0.000625]; 3/10 positive).
- full_network @ 1e-4: -0.002100 (95% CI [-0.007732, +0.003533]; 5/10 positive).
- full_network @ 2e-4: -0.001285 (95% CI [-0.007723, +0.005153]; 3/10 positive).

## 10. Fixed-LR DiDs

- 1e-4: -0.001056 (95% CI [-0.006961, +0.004849]; 4/10 positive). t-test p=0.695237; sign p=0.753906.
- 2e-4: -0.001354 (95% CI [-0.009187, +0.006480]; 4/10 positive). t-test p=0.704925; sign p=0.753906.

## 11. Designated Marginal DiD

Primary YOLO26s Marginal DiD: -0.001205 (95% CI [-0.007689, +0.005279]; 4/10 positive). SD 0.009064; median -0.003766; range [-0.011379, +0.017035]; t-test p=0.68407; sign p=0.753906. Primary positive criterion: **FAIL**.

## 12. LR moderation / ThreeWay

ThreeWay = DiD(2e-4) − DiD(1e-4): -0.000298 (95% CI [-0.005226, +0.004631]; 4/10 positive). This secondary contrast measures interaction variation across only the two frozen LRs.

## 13. HRSC transfer

HRSC vessel AP50–95 Marginal DiD: +0.005194 (95% CI [+0.002367, +0.008022]; 9/10 positive). Fixed-LR DiDs: 1e-4 +0.006177 (95% CI [+0.003016, +0.009339]; 10/10 positive); 2e-4 +0.004211 (95% CI [-0.000239, +0.008662]; 7/10 positive). No target-domain tuning was used.

## 14. DIOR transfer

DIOR vessel AP50–95 Marginal DiD: +0.012468 (95% CI [+0.011235, +0.013701]; 10/10 positive). Fixed-LR DiDs: 1e-4 +0.010309 (95% CI [+0.008369, +0.012249]; 10/10 positive); 2e-4 +0.014626 (95% CI [+0.013325, +0.015928]; 10/10 positive). Low absolute AP, if present, remains an evidence boundary.

## 15. Scene-level sensitivity

The 2,000-replicate xView scene bootstrap (seed 20260813) gave percentile interval [-0.006393, +0.004628], positive fraction 0.262, and 7 zero-target replicates. LOSO: 1/9 positive, range [-0.003677, +0.002664]. This is sparse-cluster sensitivity, not population-level uncertainty.

## 16. YOLO26n versus YOLO26s comparison

YOLO26n Marginal DiD: +0.012647 (95% CI [+0.009721, +0.015574]; 10/10 positive). YOLO26s: -0.001205 (95% CI [-0.007689, +0.005279]; 4/10 positive). YOLO26s-minus-YOLO26n: -0.013852 (95% CI [-0.022109, -0.005595]; 2/10 positive). This difference is descriptive, not a pure causal capacity effect.

## 17. Replication classification

**CASE D — REVERSED INTERACTION** under the prospectively frozen classification rule.

## 18. What the result supports

It supports only the conclusion encoded by CASE D — REVERSED INTERACTION, within two YOLO26 sizes, this 20-epoch recipe, the retained data, and the two fixed LRs.

## 19. What the result does not support

It does not establish universal YOLO behavior, a causal support mechanism, independent replication, population-level scene generalization, or superiority of one model size. It does not supersede the Faster R-CNN boundary result.

## 20. Implications for manuscript v47

Recommended direction: **weaken claim** according to the frozen evidence class. The v47 manuscript and supplement were not edited; human scientific review is required first.

## 21. Recommendation

- **weaken claim**.
- Do not run YOLO26m, Support High/Low efficacy, dose response, extra LRs, or post-result retuning.
- Stop and wait for author review.
