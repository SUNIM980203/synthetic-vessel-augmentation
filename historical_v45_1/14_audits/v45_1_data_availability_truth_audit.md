# v45.1 Data-Availability Truth Audit

## Applied state

`SPLIT_AVAILABILITY_STATE = STATE B`

The exact xView split mapping is locally retained and hash-traced, but identifier redistribution permission is not established by a retained source. The public archive therefore does not include exact xView scene/patch membership.

## Publication-facing wording

The main manuscript Data Availability sentence is corrected to:

> Access routes, split construction, aggregate split metadata, retained split hashes, and license constraints for xView, HRSC2016-MS, and the DIOR public mirror are documented in the reproducibility records.

The following sentence remains unchanged:

> Third-party imagery is not redistributed.

`THIRD_PARTY_DATA.md` now states that the retained split hash can compare an independently recovered candidate mapping but does not itself guarantee exact reconstruction. The README states plainly that exact xView membership is not included.

`MANUSCRIPT_AVAILABILITY_MATCHES_ARCHIVE = TRUE`

`MANUSCRIPT_TEXT_CHANGE_REQUIRED = TRUE`
