# Fresh Scope × Learning-Rate Factorial Protocol v1

Frozen before any new factorial outcome: **yes** (2026-08-19T08:09:16.902132+00:00).

## Scientific question

Does the Unity-minus-Duplicate vessel-detection effect differ between head-only and full-network adaptation when both scopes are evaluated at the same two fixed learning rates?

## Design

- 2 conditions: Duplicate and retained Unity Medium150.
- 2 scopes: head-only `freeze=23`; full-network `freeze=0`.
- 2 fixed LRs: `1e-4`, `2e-4`.
- 10 paired training seeds: 20260723–20260732.
- 80 accepted fresh, uninterrupted epoch-20 cells; historical cells are excluded from the primary test.

## Primary endpoint and estimand

Primary endpoint: xView scene-disjoint held-out test vessel AP50-95. Per seed, `Marginal_DiD = 0.5 × (DiD_1e-4 + DiD_2e-4)`. Report the mean across ten paired seeds with a two-sided Student-t 95% CI, one-sample two-sided t-test, and exact two-sided sign test. The positive interaction decision requires the CI lower endpoint to exceed zero.

## Training and interruption

Every cell begins from checkpoint SHA-256 `7aab2bd4aebb6181df0350c80883e0c1b2615f363481649d579a23b0e4a7140f`. Only condition, freeze, LR, and seed vary. The accepted checkpoint is epoch-20 `last.pt`; no early stopping, best-checkpoint selection, or validation-based selection is allowed. An interrupted attempt is `INVALID_TECHNICAL`, remains in the failed-attempt directory, and the cell restarts fresh. Three failed fresh attempts for one cell cause a stop.

## Evaluation and prediction retention

All xView evaluations precede HRSC2016-MS and DIOR. Full-precision per-image prediction JSON is retained with cell and evaluator metadata. HRSC is vessel-only AABB; DIOR uses the retained fixed public-mirror definition with vessel-negative images retained and other GT classes ignored.

## Scene sensitivity

Use 2,000 scene bootstrap replicates with RNG seed 20260813 and identical scene multiplicities for all cells. Zero-vessel-target replicates receive AP=0 and are reported both included and excluded. Also report LOSO. This sensitivity does not replace the paired training-seed analysis.

## Claim and stop boundaries

This is prospectively frozen but not globally untouched or an independent confirmation because the xView test set was previously used. Claims remain limited to this detector, recipe, and two LRs. Do not edit the manuscript or begin Support High/Low training.

## Prestart technical amendment

Before any training runner or factorial result existed, the PowerShell wrapper's Korean absolute path was replaced by root derivation from `$PSCommandPath`. Scientific design, schedule, configurations, and analyses were unchanged.
