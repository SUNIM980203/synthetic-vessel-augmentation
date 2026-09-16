# Synthetic augmentation and adaptation-scope study — public review package

Public research archive: https://github.com/SUNIM980203/synthetic-vessel-augmentation . Python code is AGPL-3.0-only; other material has no additional open-content license (see LICENSE_STATUS.md). Archived release: https://doi.org/10.5281/zenodo.22760748. This archive does not assert manuscript submission readiness or journal acceptance.

The archive version label v49.4.2 identifies the retained research-material package, not the latest manuscript revision. Its author metadata, AI-assistance disclosure and public-access guidance were checked against the four-author manuscript on September 16, 2026. The current manuscript and submission Supplement are not included; this update does not claim that the package contains all later manuscript text, references, figures or supplementary presentation. It combines the earlier v45.1 evidence archive with the subsequent YOLO26n/s/m factorial follow-ups and the explicitly post-hoc ordinal-42 diagnostic. Scientific results, protocols and historical code are unchanged. Historical files retain their original chronological claims; they do not supersede the later disclosures below.

Authors: Hyunbin Choi, Donghyun Woo, Ruben D. Espejo Jr., and Sunjin Yu. See `docs/AUTHOR_METADATA.md` for affiliations and `CITATION.cff` for archive citation metadata. The Zenodo DOI identifies research materials, not a published journal article.

## Start here

- `docs/REPRODUCIBILITY.md`: what can and cannot be reproduced from this package.

- `docs/DATA_ACCESS.md`: withheld materials and access boundaries.

- `docs/SCIENTIFIC_NOTES.md`: estimands and limitations.

- `factorial/`: preserved protocols, seed-level contrasts, statistics and reports.

- `diagnostic/`: post-hoc influence analysis; not a replacement primary analysis.

- `historical_v45_1/`: earlier screening, head-only, scope, negative replication and feasibility evidence.

- `historical_code/`: selected historical source snapshots, NOT a portable executable training distribution.

- `provenance.json`: original and public-copy hashes; path redactions are explicit.

- `SHA256SUMS.json`: integrity index for this package (excluding the index itself).

## Safe CPU-only verification

With Python 3.10 or newer, run from this directory:

```text

python scripts/verify_package.py

python scripts/recompute_seed_summary.py

```

These commands use the standard library only. They neither train nor load models, access private data, change hardware settings, or contact the network.

## Publication gate

Read `docs/RELEASE_CHECKLIST.md` for verification scope and remaining author actions. The current manuscript is NOT included. Authors and contact information are recorded in `docs/AUTHOR_METADATA.md`; AI assistance in manuscript, figure, and public-package preparation is described in the manuscript Acknowledgment and `CODE_AI_DISCLOSURE.md`. The authors reviewed and revised the assisted materials and remain responsible for the scientific content. The public archive is available at https://doi.org/10.5281/zenodo.22760748. Historical prepublication license/status notes are chronological records; the root LICENSE_STATUS.md governs this package.

The package does not claim complete independent end-to-end reproduction. Exact split membership, private source inputs and per-image predictions are not distributed. Historical cross-links can point to withheld files. Historical checksum indexes were omitted because copies are normalized/redacted; use the root integrity index instead.

## Research question and interpretation

This study asks whether a fixed synthetic-vessel intervention changes detector accuracy differently under head-only and full-network adaptation at common learning rates. The fresh factorial contains eight configurations and ten paired seeds per model (80 runs each for YOLO26n/s/m). The primary xView marginal interaction is positive under paired-seed inference for n and inconclusive for s/m. It is a difference between augmentation effects, not a direct AP gain or a causal model-size effect. All four s within-scope mean effects are negative. The n result is sensitive to sparse acquisition-scene resampling. The m primary analysis retains all ten seeds, including the influential realization with transient nonfinite validation loss; seed omission is a separate post-hoc diagnostic.

The package supports integrity checks and seed-summary recomputation, not end-to-end training or independent AP/scene-bootstrap replay. Use the named release asset `synthetic-vessel-augmentation-v49.4.2.zip`; GitHub-generated Source code archives follow the tag and are not interchangeable with the corrected named asset.
