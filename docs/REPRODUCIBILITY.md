# Reproducibility levels

1. Package integrity: runnable, standard-library SHA-256 verification.
2. Seed summary recomputation: runnable from the included n/s/m contrast CSVs; checks contrast arithmetic and recomputes all supplied marginal mean intervals. This is an independent release utility, not a historical frozen script.
3. AP from detections, scene bootstrap, full training and image synthesis: NOT self-contained. Private inputs and full dependency/layout reconstruction are required.

Selected historical Python files are evidence snapshots, not a complete dependency closure. Do not launch them as a quickstart. They can contain workstation-specific runtime, safety and resume conventions. No historical script was executed for this release.

Historical environment versions must be taken from the respective protocol records, not the current workspace requirements file. No new universal training requirements file is asserted.

This package adds no training, evaluation, retuning or scientific selection. Original source files are untouched. Public text derivatives normalize UTF-8 and redact workstation paths; provenance records both hashes. Historical paths/checksum references may refer to excluded source artifacts and are not package-relative availability promises.
