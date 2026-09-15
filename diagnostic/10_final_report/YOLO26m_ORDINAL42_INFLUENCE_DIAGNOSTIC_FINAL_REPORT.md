# YOLO26m Ordinal42 Influence Diagnostic - Final Report

## 1. Purpose

Assess whether the frozen YOLO26m factorial conclusion is disproportionately influenced by training seed 20260728 or its known anomalous cell, ordinal42. This is a retained-artifact post-hoc diagnostic, not a new experiment. All 80 accepted cells and all ten paired training seeds remain in the authoritative primary analysis. No retraining, evaluation rerun, replacement, retuning, or manuscript edit was performed.

The endpoint is retained xView test vessel AP50-95 on its original 0-1 scale. All exclusion calculations below are **POST-HOC SENSITIVITY ONLY**. One AP unit is 100 AP percentage points; the table values are not percentages. The source manifest, separate numerical audit and figure source files make each calculation traceable.

## 2. Frozen primary result

ORIGINAL_RESULT_MATCH = PASS. Independent aggregation from the retained ten-seed contrast CSV reproduced the statistics JSON within absolute tolerance 1e-12, before any new omission calculation. The primary mean is 0.0024685755084019761, 95% Student-t CI [-0.0099339232281184537, 0.014871074244922405], with 7/10 positive seeds. Rounded: +0.002469 [-0.009934, +0.014871]. **CASE M3 remains authoritative.** Inconclusive does not establish an absent or equivalent-to-zero interaction.

The retained n/s/m comparison is unchanged: n mean +0.012647, s mean -0.001205, m mean +0.002469. No nine-seed n/s/m reclassification was performed; the frozen M5 -> M1 -> M2 -> M3 -> M4 hierarchy is untouched.

The original scientific report was completed by a disclosed, user-authorized report-only reconciliation after its original postprocess stopped at a historical STOP gate. This diagnostic does not erase that provenance or convert it into an uninterrupted original-pipeline PASS. See the [original report](../../yolo26m_scope_lr_factorial_followup_v1/17_final_report/YOLO26m_SCOPE_LR_FACTORIAL_FINAL_REPORT.md), [provenance addendum](../../yolo26m_scope_lr_factorial_followup_v1/17_final_report/YOLO26m_POSTPROCESS_PROVENANCE_ADDENDUM.md), and [reverification](../01_reverification/yolo26m_original_result_reverification.md).

## 3. Known anomaly

Ordinal42 is `unity_medium150__head_only__lr2e-4__s20260728`. Its original training CSV records `val/cls_loss=inf` at epochs 10, 12, and 14. Those observations are retained explicitly, not dropped, imputed or clipped. Its final CSV row is finite. Read-only CPU reconfirmation found all 768 model-state tensors (21,836,914 elements) finite. The checkpoint SHA-256 remains `a4d4cebbe2e33bd0c8f34e906067ea7d1d0240e528cbe37a0619d0d485b884e6` before and after inspection.

The anomaly's cause remains unresolved. Finite final tensors and numerical consistency are not proof of an error-free trajectory. The cell is not declared invalid by this diagnostic.

## 4. Per-seed factorial contrasts

For each LR and seed, Delta_head = Unity_head - Duplicate_head and Delta_full = Unity_full - Duplicate_full. DiD = Delta_head - Delta_full. Marginal DiD averages the two fixed-LR DiDs; ThreeWay = DiD(2e-4) - DiD(1e-4). Reconstructing these directly from retained per-cell AP files reproduced all 240 vessel contrast values across three datasets (eight contrasts x ten seeds x three datasets).

| Training seed | DiD 1e-4 | DiD 2e-4 | Marginal DiD | ThreeWay |
|---|---|---|---|---|
| 20260723 | +0.017983082 | +0.013559171 | +0.015771126 | -0.004423911 |
| 20260724 | +0.003926160 | -0.022256323 | -0.009165081 | -0.026182483 |
| 20260725 | +0.011732404 | -0.003253094 | +0.004239655 | -0.014985498 |
| 20260726 | +0.000452420 | +0.009063224 | +0.004757822 | +0.008610804 |
| 20260727 | +0.009297966 | +0.004647763 | +0.006972865 | -0.004650203 |
| 20260728 | +0.000423241 | -0.078262378 | -0.038919568 | -0.078685619 |
| 20260729 | -0.000670072 | -0.013734346 | -0.007202209 | -0.013064275 |
| 20260730 | +0.015100974 | +0.014843214 | +0.014972094 | -0.000257759 |
| 20260731 | +0.009195213 | +0.023138524 | +0.016166869 | +0.013943311 |
| 20260732 | +0.014926608 | +0.019257758 | +0.017092183 | +0.004331149 |

All four Unity-minus-Duplicate deltas are included at round-trip numeric precision in [per-seed contrasts](../02_per_seed_contrasts/yolo26m_per_seed_factorial_contrasts.csv). [Figure 1](../08_figures/figure_influence_1.pdf) shows the original marginal values, mean and interval.

## 5. Seed 20260728 factorial breakdown

| Scope | LR | Duplicate AP | Unity AP |
|---|---|---|---|
| head_only | 1e-4 | 0.157154892 | 0.157144096 |
| head_only | 2e-4 | 0.148831002 | 0.075677438 |
| full_network | 1e-4 | 0.115011765 | 0.114577728 |
| full_network | 2e-4 | 0.110317009 | 0.115425824 |

The seed's head delta is -0.000010796 at 1e-4 and -0.073153563 at 2e-4. Its full-network delta is -0.000434037 at 1e-4 and +0.005108814 at 2e-4. Thus fixed-LR DiDs are +0.000423241 and -0.078262378, and its marginal DiD is -0.038919568.

This seed's additive contribution to the equally weighted original marginal mean is its value divided by ten: -0.003891957. This additive contribution is different from the omission influence because omission reweights the remaining seeds from 1/10 to 1/9. Exact values for all deltas and contributions appear in the [eight-cell breakdown](../03_ordinal42/yolo26m_seed20260728_factorial_breakdown.md).

## 6. Ordinal42 distributional position

Ordinal42 AP is 0.075677438, rank 1/10 from lowest in Unity/head/2e-4. The next-lowest value is 0.153340983, a gap of 0.077663545. The ten-cell median is 0.157976403; its signed median deviation is -0.082298964. No threshold-based outlier label is used.

| Ten-cell distribution measure | Value |
|---|---|
| mean | 0.149496982 |
| sample_sd | 0.026015593 |
| median | 0.157976403 |
| mad | 0.001499744 |
| iqr | 0.002981840 |
| z | -2.837511435 |
| robust_mad_z | -37.012860321 |
| absolute_distance_from_median | 0.082298964 |

SD is sample SD; MAD is unscaled median absolute deviation; IQR uses linear/type-7 quartiles. Robust z = 0.6744897501960817*(AP - median)/MAD. Its large magnitude is an empirical distance relative to a tightly clustered sample, not a validation or exclusion rule. All ten AP values and conventions are in the [distribution audit](../03_ordinal42/yolo26m_ordinal42_distribution_position.md).

Cell-only dispersion check, **POST-HOC SENSITIVITY ONLY**:

| Configuration subset | n | Mean | Sample SD |
|---|---|---|---|
| Original Unity/head/2e-4 | 10 | 0.149496982 | 0.026015593 |
| Omit ordinal42 cell only | 9 | 0.157699154 | 0.002135834 |

SD falls by 91.79% and separately centered sample variance by 99.33%. The target contributes 89.46% of the original centered sum of squares. Descriptively, the large configuration dispersion is predominantly associated with this observation. These are not causal percentages. No factorial contrast or hypothesis test is constructed from this incomplete seed.

## 7. Matched Duplicate comparison

The paired Duplicate AP is 0.148831002, rank 2/10 from lowest, versus Unity 0.075677438, rank 1/10. The other-nine Unity range is [0.153340983, 0.160210141], while the other-nine Duplicate range is [0.148018342, 0.162087934]. Duplicate is below its other-nine median 0.154968844; it is not unusually high in this retained distribution.

The negative paired head/2e-4 delta (-0.073153563) is therefore descriptively associated with low Unity, not high Duplicate. Low Duplicate partially offsets that negative difference. This arithmetic description does not identify a root cause. Full other-nine mean, SD, median and ranges appear in the [pair diagnostic](../03_ordinal42/yolo26m_seed20260728_pair_diagnostic.md).

## 8. Leave-one-seed-out analysis

All ten leave-one-TRAINING-seed-out subsets were evaluated, each retaining nine complete eight-cell seed blocks. This is distinct from the original leave-one-SCENE-out sensitivity. Each interval uses the sample SD and two-sided Student-t critical value with df=8. These overlapping, post-hoc intervals are unadjusted diagnostics, not independent replications or new confirmatory tests.

All ten marginal intervals include zero. The omitted-subset means range from +0.000843730 to +0.007067258. No individual omission changes the original CI-based inconclusive direction. The complete fixed-LR and ThreeWay intervals and positive counts are in [LOSO CSV](../04_leave_one_seed_out/yolo26m_leave_one_seed_out.csv) and [Figure 2](../08_figures/figure_influence_2.pdf).

| Omitted seed | Marginal mean | 95% CI low | 95% CI high | Positive / 9 |
|---|---|---|---|---|
| 20260723 | +0.000990514 | -0.012621313 | +0.014602342 | 6/9 |
| 20260724 | +0.003761204 | -0.009975497 | +0.017497905 | 7/9 |
| 20260725 | +0.002271789 | -0.011854288 | +0.016397866 | 6/9 |
| 20260726 | +0.002214215 | -0.011905750 | +0.016334179 | 6/9 |
| 20260727 | +0.001968099 | -0.012108070 | +0.016044268 | 6/9 |
| 20260728 | +0.007067258 | -0.000628977 | +0.014763493 | 7/9 |
| 20260729 | +0.003543107 | -0.010317940 | +0.017404155 | 7/9 |
| 20260730 | +0.001079296 | -0.012594547 | +0.014753138 | 6/9 |
| 20260731 | +0.000946543 | -0.012633036 | +0.014526122 | 6/9 |
| 20260732 | +0.000843730 | -0.012656432 | +0.014343893 | 6/9 |

## 9. Seed 20260728 exclusion sensitivity

**Post-hoc sensitivity excluding training seed 20260728.** All eight cells of that training seed are omitted only in this derivative calculation. This is not a replacement primary result.

| Analysis | Original 10 seeds: mean [95% t CI] | Excluding seed 20260728: POST-HOC SENSITIVITY ONLY | Qualitative change? |
|---|---|---|---|
| Marginal DiD | +0.002468576 [-0.009933923, +0.014871074] | +0.007067258 [-0.000628977, +0.014763493] | inconclusive -> inconclusive |
| DiD 1e-4 | +0.008236800 [+0.003345989, +0.013127610] | +0.009104973 [+0.004000044, +0.014209902] | positive -> positive |
| DiD 2e-4 | -0.003299649 [-0.024793251, +0.018193953] | +0.005029543 [-0.006757237, +0.016816324] | inconclusive -> inconclusive |
| ThreeWay | -0.011536448 [-0.030417709, +0.007344812] | -0.004075429 [-0.013721613, +0.005570755] | inconclusive -> inconclusive |

Nine-seed marginal SD = 0.010012430; median = +0.006972865; positive seeds = 7/9; exact two-sided sign p = 0.179687500. The marginal mean increases, but the lower CI limit remains below zero (-0.000628977). The result is closer to a zero-excluding positive interval and should not be described as numerically insensitive. Its qualitative inconclusive status persists under the fixed 95% interval rule.

## 10. LR-specific influence

Influence_j = mean_without_j - original_mean. Absolute influences are ranked descending across all ten omissions.

| Estimand | 20260728 influence | Absolute rank / 10 |
|---|---|---|
| marginal_did | +0.004598683 | 1/10 |
| did_lr1e4 | +0.000868173 | 3/10 |
| did_lr2e4 | +0.008329192 | 1/10 |

Seed 20260728 is the most influential seed for marginal DiD, with absolute influence 2.830 times the next-largest marginal influence. It is also the largest influence at LR 2e-4. Its LR 1e-4 influence is not zero and ranks third, so the effect is not exclusively confined to 2e-4; its LR 2e-4 influence magnitude is approximately 9.594 times larger. The target seed's DiD is nearly zero at 1e-4 (+0.000423241) and strongly negative at 2e-4 (-0.078262378). See [marginal ranking](../07_statistics/yolo26m_seed_influence_ranking.csv), [LR2e-4 ranking](../07_statistics/yolo26m_lr2e4_seed_influence_ranking.csv) and [Figure 3](../08_figures/figure_influence_3.pdf).

Robust summaries below are descriptive only. The trimmed mean removes one observation from each tail (10% each at n=10). They never replace the prespecified t analysis.

| Contrast | Mean | Median | 10%-each-tail trimmed mean | MAD | IQR | Minimum | Maximum |
|---|---|---|---|---|---|---|---|
| marginal_did | +0.002468576 | +0.005865343 | +0.005814143 | 0.010103654 | 0.019913111 | -0.038919568 | +0.017092183 |
| did_lr1e4 | +0.008236800 | +0.009246590 | +0.008131873 | 0.005767201 | 0.012807202 | -0.000670072 | +0.017983082 |
| did_lr2e4 | -0.003299649 | +0.006855494 | +0.002765921 | 0.011255426 | 0.025636237 | -0.078262378 | +0.023138524 |

## 11. Cross-dataset diagnostic

| Dataset | Ordinal42 Unity AP | Unity rank / 10 | Unity median | Median deviation | Paired Duplicate AP | Unity - Duplicate |
|---|---|---|---|---|---|---|
| xview_test | 0.075677438 | 1/10 | 0.157976403 | -0.082298964 | 0.148831002 | -0.073153563 |
| hrsc2016_ms | 0.130234504 | 1/10 | 0.167986892 | -0.037752388 | 0.158322488 | -0.028087984 |
| dior_public_mirror | 0.036477588 | 1/10 | 0.089311428 | -0.052833840 | 0.042795169 | -0.006317581 |

Ordinal42 ranks lowest on all three datasets, not only xView. Its matched head/2e-4 Unity-minus-Duplicate effect is negative in all three. The paired Duplicate rank is 2/10 on xView, 10/10 on HRSC and 7/10 on DIOR. HRSC therefore combines the lowest Unity with the highest Duplicate observation in their respective configuration distributions.

For the entire seed 20260728, marginal DiD is -0.038919568 on xView, -0.017888978 on HRSC, and +0.004562059 on DIOR. The DIOR marginal remains positive despite its negative LR2e-4 DiD. These are the same training realization across test sets, not independent new training evidence. The [cross-dataset diagnostic](../05_cross_dataset/yolo26m_ordinal42_cross_dataset_diagnostic.md) and companion per-seed CSV retain all details.

## 12. Training-history diagnostic

The complete original 20-epoch scalar trajectories, including train/box_loss, train/cls_loss, train/dfl_loss, validation losses, precision, recall, mAP50(B), mAP50-95(B), cumulative time and all three learning-rate columns, are retained in the [derived history CSV](../06_training_history/yolo26m_ordinal42_training_history.csv). Original logger precision is preserved, not upgraded by inference.

Validation cls loss is approximately 1.29 in epochs 1-7, 14.7913 at epoch 8 and 9.03909 at epoch 9, before the recorded inf values at epochs 10, 12 and 14. Train/cls_loss rises from 0.78893 to 1.67175 between epochs 1 and 20. Validation mAP50-95(B) is 0.19142 at epoch 7 and 0.11157 at epoch 20. These temporal observations are not evidence that inf caused the later AP behavior. Training-validation mAP is not the test vessel endpoint.

The final epoch's retained values and all final model-state tensors are finite. The stored checkpoint epoch metadata is -1 and is unchanged; epoch-20 provenance relies on the CSV and accepted-ledger hash mapping. All 240 retained evaluation metric files contain finite aggregate and vessel AP values. The prior full prediction-archive audit remains the evidence for prediction-row finiteness; prediction rows were not re-scored here.

[Figure 4](../08_figures/figure_influence_4.pdf) shows gaps at nonfinite observations and explicit inf annotations above the finite plotting axis. Inf is not silently clipped or converted to a finite loss. The [history diagnostic](../06_training_history/yolo26m_ordinal42_training_history_diagnostic.md) records the finite-state result and the unexplained epoch-17 elapsed-time increase without inferring resume or a hardware cause.

## 13. Numerical audit

NUMERICAL_MISMATCH = 0. The separate verifier performed 7,242 comparisons using independent NumPy array reconstruction, explicit order statistics, t.interval, a closed-form influence identity, and exact binomial sums. It did not import the diagnostic builder or execute any training/evaluation runner. It verified the full result, all per-seed contrasts, all ten omissions, target exclusion, influence ranks, distribution statistics, cross-dataset values, all 16 exported CSVs, four figure source CSVs and training-history preservation.

All 598 protected files and 264 manifested scientific sources matched their recorded hashes. This includes all 80 accepted final checkpoints, original training CSVs, original protocol/statistics, STOP records, provenance addendum, and the v48 manuscript/supplement tree. PDF figures were rendered and visually inspected for labels, intervals, clipping and explicit inf annotations. See the [source manifest](../00_sources/yolo26m_influence_diagnostic_source_manifest.md), [numerical audit](../09_audits/yolo26m_influence_numerical_audit.md) and final acceptance record.

## 14. Influence classification

**CATEGORY II - MODERATE INFLUENCE.** Seed 20260728 has the largest marginal influence, and its removal substantially reduces observed dispersion, but its nine-seed marginal CI still includes zero. No other single-seed omission changes the CI-based qualitative conclusion either. Category III is not supported under the fixed rule; Category IV is not supported because multiple qualitative reversals were not observed. Category I would understate the empirically largest influence and dispersion contribution.

The user-defined category meanings were preserved. Before new sensitivity calculation, [diagnostic rules](../00_sources/diagnostic_rules_lock.md) fixed the operational CI-direction rule and precedence IV -> III -> II -> I, including the descriptive top-three convention for “among the largest.” These diagnostic operationalizations are not a new preregistered primary analysis. The target's rank is actually first, so its Category II interpretation does not depend on the top-three boundary. No M1-M5 reclassification is made.

## 15. What is supported

- The target seed is quantitatively influential, particularly at LR 2e-4. Ordinal42 is the lowest retained AP observation in its configuration on all three datasets.
- The large xView Unity/head/2e-4 configuration SD is predominantly associated with ordinal42 in the descriptive dispersion calculation.
- The frozen xView seed-level conclusion remains inconclusive after every individual training-seed omission, including 20260728. Its estimate and uncertainty are nevertheless numerically sensitive.
- Transparent supplementary sensitivity disclosure is warranted. These statements are conditional on the retained checkpoints, datasets, two LRs, ten seeds, and the fixed recipe.

## 16. What is not supported

This diagnostic does not establish that ordinal42 is invalid, that ordinal42 caused CASE M3, that inf loss caused poor AP, that omission reveals a true or corrected result, or that the experiment should be reclassified with nine seeds. It does not show that the ten-seed result was wrong. It cannot isolate a cell-specific causal effect from whole-seed omission, establish a hardware/optimizer root cause, prove an error-free training trajectory, or establish equivalence to zero. It does not justify detector-general, monotonic model-size, or causal transfer claims. No incomplete-seed paired test was performed.

## 17. Manuscript recommendation

**Recommendation B: supplement sensitivity paragraph and figure recommended.** Preserve the complete ten-seed primary result and CASE M3. In later manuscript work, disclose the transient validation-loss anomaly and use the all-seed LOSO figure (Figure 2), with Figure 1 or Figure 3 if space permits. The supplement should also state that the configuration SD is strongly affected and that ordinal42 ranks lowest across all three evaluation datasets. Do not present the nine-seed estimate as a substitute primary endpoint.

Proposed supplementary wording for scientific review only:

> A post-hoc training-seed influence diagnostic retained the prespecified ten-seed YOLO26m result. Seed 20260728, which contains a head-only/LR2e-4 Unity cell with transient nonfinite validation classification loss, had the largest marginal influence. Omitting that complete seed block changed the marginal DiD from +0.002469 (95% t CI -0.009934 to +0.014871) to +0.007067 (95% t CI -0.000629 to +0.014763). All ten leave-one-training-seed-out intervals included zero. This indicates quantitative sensitivity without a single-omission reversal of the inconclusive primary interpretation. The loss anomaly's cause remains unresolved, and no cell was removed from the primary analysis.

The v48 manuscript, supplement and figures have not been edited. Diagnostic work stops here after final acceptance and awaits scientific review. No follow-on training, new model, new grid, or automation is started.
