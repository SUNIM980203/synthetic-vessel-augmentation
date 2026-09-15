# Synthetic augmentation and adaptation-scope study — public review package

Author-authorized research archive for `SUNIM980203/synthetic-vessel-augmentation`. Python code is AGPL-3.0-only; other material has no additional open-content license (see LICENSE_STATUS.md). No DOI or manuscript submission-readiness claim is made.

This package accompanies manuscript v49.4.2. It combines the earlier v45.1 evidence archive with the subsequent YOLO26n/s/m factorial follow-ups and the explicitly post-hoc ordinal-42 diagnostic. Historical files retain their original chronological claims; they do not supersede the later disclosures below.

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

Read `docs/RELEASE_CHECKLIST.md` for verification scope and remaining author actions. The current manuscript is NOT included. Authors and contact information are recorded in `docs/AUTHOR_METADATA.md`; AI-assisted package preparation is disclosed in `CODE_AI_DISCLOSURE.md`. A Zenodo DOI has not been minted. Historical prepublication license/status notes are chronological records; the root LICENSE_STATUS.md governs this package.

The package does not claim complete independent end-to-end reproduction. Exact split membership, private source inputs and per-image predictions are not distributed. Historical cross-links can point to withheld files. Historical checksum indexes were omitted because copies are normalized/redacted; use the root integrity index instead.
