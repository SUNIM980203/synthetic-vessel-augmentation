# Validation Reuse Audit v39

## Decision

**Case A — same validation split. Disclosure is required.** The common xView-trained YOLO26n `best.pt` was selected using the same 375-image, nine-acquisition-scene xView validation split later used for the locally frozen primary scope-interaction endpoint.

This is historical source-model selection reuse, not condition-specific Unity selection leakage. All scope-interaction conditions shared the same initialization, and scope-specific settings were selected using Duplicate-only inner development without Unity, public-validation, test, HRSC2016-MS, or DIOR outcomes.

## Exact comparison

| Check | Initial checkpoint selection | Scope-interaction endpoint | Result |
|---|---:|---:|---|
| Validation images | 375 | 375 | 375/375 overlap (100.0%) |
| Acquisition scenes | 9 | 9 | 9/9 overlap |
| Resolved directory | `PreparedData/yolo_expanded_v1/real/images/val` | `PreparedData/yolo_expanded_v1/real/images/val` | exact path equality |
| Ordered image-ID/path/scene/content manifest | `1d3b66467af1f227be781b7220615b0e661c2075c1ec1816ca7bc53aa94a22f2` | `1d3b66467af1f227be781b7220615b0e661c2075c1ec1816ca7bc53aa94a22f2` | exact SHA-256 equality |
| COCO validation manifest | `8d2c9c39df45f5e3221683cd26c9e38d330fc824cd942541c9c205ab4c53b456` | `8d2c9c39df45f5e3221683cd26c9e38d330fc824cd942541c9c205ab4c53b456` | same source manifest |

The two YAML files differ in their training entries but both resolve `val: real/images/val` under the same dataset root. Every one of the 375 validation images matched by image ID and SHA-256 content hash. The COCO manifest mapped them to the same nine `source_scene` identifiers.

## Initial `best.pt` evidence

The initial run recorded validation enabled on `split=val`, 50 configured epochs, patience 20, and 47 completed rows. Validation mAP50-95 had a unique maximum of 0.14549 at epoch 27. The finalized `best.pt` embedded validation metric matches this row exactly. The checkpoint's serialized `epoch=-1` field is an Ultralytics finalization artifact and is not used to infer selection.

## Scientific interpretation

The primary interaction remained locally frozen before its one-time evaluation, and the later condition-specific training/settings did not use Unity or public validation outcomes for selection. Nevertheless, the xView validation endpoint was not independent of historical selection of the common source model. Publication text must therefore call it a **locally frozen primary scope-interaction endpoint on the historical xView validation split**, not an untouched or fully independent confirmatory validation cohort.

Post-lock xView test, HRSC2016-MS, and DIOR results provide important transport checks, but they are not relabeled as newly confirmatory evidence.

## Integrity status

PASS WITH DISCLOSURE REQUIRED. No training or inference was performed for this audit.
