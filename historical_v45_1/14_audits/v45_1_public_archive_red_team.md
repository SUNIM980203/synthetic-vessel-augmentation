# v45.1 Public-Archive Red Team

Scope reviewed: the v45.1 manuscript, unchanged v45 supplement, and the corrected public archive. The review treated retained machine records and protocol/chronology artifacts as authoritative; manuscript and prior assistant prose were not accepted as provenance.

| Question | Answer | Evidence and resolution |
|---|---|---|
| 1. Does any archive file still label the original support screen as post-result? | NO | Repository-wide text search found no retained occurrence of the erroneous label. C01/C02 are `CONFIGURATION / SCREENING — TRAINING-ONLY; DOWNSTREAM AP UNOPENED`, and the chronology audit establishes screening before downstream AP. |
| 2. Are ten-seed HRSC/DIOR effects clearly distinguished from v48 post-lock scope transfer? | YES | `00_manifest/v45_1_effect_stage_mapping.csv`, the corrected traceability table, and README Sections 4–5 separate the head-only robustness expansion from v48 DiD transfer. |
| 3. Is README claim strength no stronger than manuscript claim strength? | YES | Detector effects are described as “seed-consistent paired”; the historical-validation endpoint is “locally frozen”; no independent, causal, globally confirmatory, untouched, or end-to-end reproducibility claim remains. |
| 4. Can a reader tell whether exact xView split membership is actually provided? | YES | README Section 8 and `THIRD_PARTY_DATA.md` say exact scene/patch membership is not included. The provenance audit records that a local mapping exists but identifier-redistribution permission was not established. |
| 5. Does Data Availability accurately describe what is distributed? | YES | The manuscript states that construction, aggregate metadata, retained hashes, and license constraints are documented. It does not claim that exact membership is public. |
| 6. Are historical negative findings still visible? | YES | The Faster R-CNN failed gate, Small-scale infeasibility, failed historical Tenengrad gate, and absence of same-scale detector training/AP remain explicit in the README, manuscript, and retained reports. |

Additional challenges found no contradiction in public availability: the archive is classified `READY_FOR_AUTHOR_UPLOAD`, not public, with no URL/DOI and an explicit author-license action. No scientific result, negative decision, or stopping rule was removed.

`PUBLIC_ARCHIVE_RED_TEAM = PASS`
