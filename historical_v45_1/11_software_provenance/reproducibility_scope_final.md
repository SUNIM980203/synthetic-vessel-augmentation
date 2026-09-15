
# Final reproducibility scope

## What the submission archive supports

- One-to-one integrity verification of every packaged file through SHA-256, with a documented self-referential manifest entry.
- Inspection of frozen protocols, configurations, analysis/evaluation implementation, RealCutout construction implementation, derived tables, validation reuse audit, and figure-selection logic.
- Reproduction of package-level tables/figures that depend only on redistributed derived results.
- Full retraining/evaluation by an authorized researcher who separately obtains the licensed datasets and supplies the locally retained checkpoint paths described by the frozen hashes.

## What the archive intentionally excludes

Third-party/raw imagery, raw datasets, annotations that redistribute restricted data, model checkpoints, per-image predictions, and embeddings. These exclusions prevent a fully standalone end-to-end rerun from the archive alone. In particular, the acquisition-scene bootstrap requires retained per-image predictions that are not redistributed. Scope-interaction scene bootstrap is impossible because those predictions were not retained.

## Public availability status

The archive is complete locally but has no public repository, persistent URL, DOI, or author-approved software license. The manuscript does not imply otherwise. Public deposit and license selection remain submission blockers owned by the authors.
