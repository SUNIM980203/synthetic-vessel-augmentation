# Scope × Learning-Rate Factorial Follow-up — Final Scientific Report

## 1. Scientific question

Does the Unity-minus-Duplicate vessel-detection effect differ between head-only and full-network adaptation when both scopes are crossed with the same two learning rates?

## 2. Frozen design

The fresh block crossed two conditions, two scopes, two LRs, and ten paired training seeds (80 cells). The protocol and all analysis scripts were hashed before the first outcome. The xView test endpoint had been used previously in the broader study, so this is a prospectively frozen factorial follow-up, not an untouched independent confirmation.

## 3. Runtime/integrity summary

- Accepted fresh cells: 80/80; invalid accepted cells: 0.
- Every accepted run was uninterrupted, used `resume=False`, and ended at epoch 20 `last.pt`.
- Nominal exposure per cell: 3000 images/epoch × 20 epochs = 60000 image exposures.
- Prediction archives: 240/240 (xView, HRSC2016-MS, DIOR).

## 4. Absolute performance

| Dataset | Scope | LR | Condition | Vessel AP50-95 mean ± SD |
|---|---|---:|---|---:|
| dior_public_mirror | full_network | 1e-4 | duplicate | 0.027261 ± 0.002087 |
| dior_public_mirror | full_network | 1e-4 | unity_medium150 | 0.028565 ± 0.001637 |
| dior_public_mirror | full_network | 2e-4 | duplicate | 0.029211 ± 0.002401 |
| dior_public_mirror | full_network | 2e-4 | unity_medium150 | 0.030916 ± 0.003277 |
| dior_public_mirror | head_only | 1e-4 | duplicate | 0.020324 ± 0.000809 |
| dior_public_mirror | head_only | 1e-4 | unity_medium150 | 0.025587 ± 0.000851 |
| dior_public_mirror | head_only | 2e-4 | duplicate | 0.020525 ± 0.000649 |
| dior_public_mirror | head_only | 2e-4 | unity_medium150 | 0.027272 ± 0.000965 |
| hrsc2016_ms | full_network | 1e-4 | duplicate | 0.107189 ± 0.004584 |
| hrsc2016_ms | full_network | 1e-4 | unity_medium150 | 0.114362 ± 0.004014 |
| hrsc2016_ms | full_network | 2e-4 | duplicate | 0.101113 ± 0.002559 |
| hrsc2016_ms | full_network | 2e-4 | unity_medium150 | 0.107767 ± 0.004313 |
| hrsc2016_ms | head_only | 1e-4 | duplicate | 0.127620 ± 0.001197 |
| hrsc2016_ms | head_only | 1e-4 | unity_medium150 | 0.138728 ± 0.001405 |
| hrsc2016_ms | head_only | 2e-4 | duplicate | 0.126995 ± 0.001194 |
| hrsc2016_ms | head_only | 2e-4 | unity_medium150 | 0.138692 ± 0.001641 |
| xview_test | full_network | 1e-4 | duplicate | 0.066958 ± 0.010184 |
| xview_test | full_network | 1e-4 | unity_medium150 | 0.059077 ± 0.009068 |
| xview_test | full_network | 2e-4 | duplicate | 0.060040 ± 0.006464 |
| xview_test | full_network | 2e-4 | unity_medium150 | 0.054934 ± 0.006507 |
| xview_test | head_only | 1e-4 | duplicate | 0.065179 ± 0.003088 |
| xview_test | head_only | 1e-4 | unity_medium150 | 0.070768 ± 0.002341 |
| xview_test | head_only | 2e-4 | duplicate | 0.061112 ± 0.002478 |
| xview_test | head_only | 2e-4 | unity_medium150 | 0.067831 ± 0.002377 |

## 5. Primary marginal factorial interaction

Mean Marginal_DiD = +0.012647; SD 0.004091; median +0.013048; range [+0.004153, +0.019419].
Two-sided Student-t 95% CI [+0.009721, +0.015574]; t-test p=4.31846e-06; positive seeds 10/10; exact sign-test p=0.00195312.
Frozen decision: **POSITIVE INTERACTION SUPPORTED**.

## 6. Fixed-LR interactions

- 1e-4 DiD: +0.013470, 95% CI [+0.009043, +0.017896], positive 10/10.
- 2e-4 DiD: +0.011825, 95% CI [+0.007494, +0.016155], positive 10/10.

## 7. Learning-rate moderation / ThreeWay

ThreeWay (DiD_2e-4 − DiD_1e-4) = -0.001645, 95% CI [-0.008160, +0.004869], positive 4/10.
Observed interpretation branch: **CASE A: the positive marginal interaction is accompanied by positive fixed-LR DiD means at both tested LRs.**

## 8. Cross-dataset transfer

HRSC2016-MS and DIOR were evaluated without target-domain tuning. HRSC is vessel-only AABB evaluation; DIOR retains vessel-negative images and ignores other GT classes under the retained definition. Full absolute and factorial results are in Tables A–D.

## 9. Scene-level sensitivity

The 2,000-replicate scene bootstrap (seed 20260813) gave an including-zero-target percentile interval [+0.000000, +0.018319]. Zero-target replicates: 7. Excluding-zero interval: [+0.000000, +0.018319]. LOSO range: [+0.009258, +0.017482].

## 10. Comparison with historical v48

Historical v48 is retained as descriptive context only; it was not included in the fresh test. See `scope_lr_factorial_vs_historical_v48.md`.

## 11. What the experiment supports

- CASE A: the positive marginal interaction is accompanied by positive fixed-LR DiD means at both tested LRs.

## 12. What the experiment does NOT support

- It does not establish a universal effect of freezing, learning rate, or optimization schedule.
- It is not an independent or globally untouched confirmation.
- It does not establish a causal feature-support mechanism or authorize the Support High/Low efficacy study.
- External transfer and scene-bootstrap results are secondary sensitivity evidence.

## 13. Implications for manuscript v46

The manuscript claim should follow the frozen observed branch above. No manuscript file was edited in this experiment.

## 14. Recommended manuscript claim revision

Recommended boundary: report that case a: the positive marginal interaction is accompanied by positive fixed-lr did means at both tested lrs. Limit the statement to the tested detector, fixed 20-epoch recipe, two learning rates, and retained evaluation datasets; disclose prior use of the xView test set and the sparse-scene boundary.
