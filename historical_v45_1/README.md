# IEEE Access reproducibility archive v45.1

Repository state: **READY_FOR_AUTHOR_UPLOAD**. This is a sanitized local package prepared for author deposit. It has not been deposited publicly and has no verified public URL or DOI. Repository/code/documentation licensing remains an author action.

## 1. Study purpose

The study asks whether a small vessel-focused synthetic augmentation is associated with seed-consistent paired detector effects under the tested training designs, and whether those effects depend on the adaptation interface. Representation support is treated as detector-specific configuration/screening evidence, not as an established independent cause.

## 2. Evidence hierarchy

The public evidence chronology is:

1. the original AllScale150-to-Medium150 training-only feature-support configuration/screening, completed while downstream AP remained unopened;
2. locked internal YOLO head-only evaluation followed by a separate ten-seed head-only robustness expansion;
3. a locally frozen historical-validation adaptation-scope interaction, with broader earlier efficacy outcomes already known and scope settings selected on a separate Duplicate-only inner split;
4. secondary post-lock xView-test, HRSC2016-MS, and DIOR public-mirror scope-transfer estimates;
5. post-result scene-sensitivity analyses;
6. the negative Faster R-CNN replication; and
7. prospective same-scale feasibility work that preserved its negative stopping decisions.

The corrected claim-stage mapping is in `00_manifest/v45_1_effect_stage_mapping.csv`, and the corrected source traceability table is in `00_manifest/v45_1_claim_source_traceability.csv`.

## 3. Primary endpoint

The original efficacy comparison uses maritime-vessel AP50-95 for Unity Medium150 versus the duration-matched Duplicate condition under the locked head-only YOLO interface. The historical xView-validation vessel interaction is the locally frozen primary endpoint of the later v48 adaptation-scope study; it is not an untouched or globally preregistered confirmation.

## 4. Head-only ten-seed robustness

`04_head_only_results/` contains the ten-seed head-only robustness expansion across xView test, HRSC2016-MS, and the DIOR public mirror. These estimates extend the earlier five-seed efficacy results; they are not v48 adaptation-scope transfer interactions and do not retroactively change the earlier locked decisions.

## 5. v48 adaptation-scope stage

`05_scope_interaction/` contains the head-minus-full difference-in-differences stage. The xView historical-validation vessel interaction was decided first. The xView-test and external HRSC/DIOR interactions are secondary post-lock transfer estimates. `06_external_transfer/` contains the associated external-evaluation code and reports.

## 6. Negative Faster R-CNN replication

The failed Faster R-CNN primary gate remains visible in `08_faster_rcnn_negative_replication/`. External evaluation remained locked. This is a negative replication in one tested two-stage configuration, not a general detector-validation result.

## 7. Same-scale feasibility stopping decision

The same-scale sequence in `09_same_scale_feasibility/` is feasibility evidence only. The Small branch was infeasible under the frozen selector. The historical Medium150 High/Low construction failed the frozen Tenengrad gate; no same-scale detector training occurred and no same-scale AP exists. Post-result localization in `10_radiometry_and_tenengrad/` does not retroactively convert that failure to PASS.

## 8. Archive scope

The archive contains source-audited protocols, decision rules, aggregate and per-seed result tables, analysis utilities, evidence chronology, sanitized audits, and a complete integrity inventory. For xView it documents split construction, aggregate split metadata, and the retained canonical split hash. Exact xView scene/patch membership is not included because the retained record does not establish redistribution permission for the source identifiers.

## 9. Excluded third-party artifacts

Raw xView, HRSC2016, and DIOR imagery; third-party source cutouts; checkpoints; embeddings; per-image predictions; generated training images; exact xView membership identifiers; and local historical inputs are not included. See `THIRD_PARTY_DATA.md`.

## 10. Reproducibility boundary

This archive is designed to make the reported protocols, decision rules, aggregate results, evidence chronology, analysis utilities, and package integrity auditable. Full end-to-end independent recomputation requires separately obtained third-party imagery and certain retained historical artifacts that are not redistributed.

## 11. Integrity verification

1. Keep the extracted directory structure unchanged.
2. From the archive root, verify each line in `00_manifest/SHA256SUMS.txt` with a SHA-256 tool.
3. Compare the inventory with `00_manifest/release_manifest.csv`.
4. The checksum index deliberately omits its own hash to avoid circular self-reference; the manifest contains explicit generated-index rows for both manifest files.

No public availability should be claimed until the author uploads this package, selects an appropriate license, and supplies a verified URL or DOI.
