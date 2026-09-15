# Feature-support v20 bootstrap amendment v21

Frozen locally on 2026-07-31 after the v20 five-seed AP results were known but before lossless-stat export or bootstrap resampling. This is a post-result uncertainty and reproducibility analysis. It cannot change the frozen v20 paired-seed decision.

## Population and models

Use all 375 images in the unchanged scene-disjoint xView validation split and the exact epoch-20 checkpoints for Duplicate, Small150, and Medium150 across seeds 20260723-20260727. Re-evaluate all 15 checkpoints with the fixed v20 settings and export full-precision, per-image post-NMS predictions, target counts, and IoU 0.50:0.95 matching indicators. Every full-population vessel AP50-95 value must reconstruct exactly before resampling.

## Bootstrap

Use 2,000 paired replicates with seed 20260731. In every replicate, sample 375 validation images with replacement and apply identical image multiplicities to every condition and training seed. Preserve each model's frozen global confidence ordering when reconstructing precision-recall curves. Compute:

1. an image-cluster bootstrap that averages paired effects across the five fixed training seeds; and
2. a two-level bootstrap that additionally resamples five paired training-seed indices with replacement.

The primary sensitivity contrast is Medium150 minus Small150 vessel AP50-95. Small150 minus Duplicate is secondary. Report percentile 95% intervals and the fraction of positive replicates. No test or external image is used.

## Interpretation

An interval above zero strengthens the uncertainty characterization of the already positive v20 ranking result but does not make this post-result amendment prospective. The optimization-seed paired Student-t lower bound remains the frozen primary decision.
