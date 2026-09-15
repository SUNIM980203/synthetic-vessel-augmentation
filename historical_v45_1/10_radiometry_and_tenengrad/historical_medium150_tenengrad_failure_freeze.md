# Historical Medium150 Tenengrad failure freeze

**POST_RESULT_DIAGNOSTIC — immutable observed outcome**

No subsequent diagnostic may change, rescue, reinterpret, or retroactively pass this outcome.

| Endpoint | Frozen result |
|---|---:|
| SUPPORT_GATE | PASS |
| ABSOLUTE_WEBER_GATE | PASS |
| OTHER_RADIOMETRY_EXCEPT_TENENGRAD | PASS |
| TENENGRAD_HIGH | 0.5229159353550353 — **FAIL** against <= 0.50 |
| TENENGRAD_LOW | 0.5584633752276690 — **FAIL** against <= 0.50 |
| PAIRWISE_TENENGRAD | PASS |
| APPEARANCE_BALANCE | PASS |
| GEOMETRY | PASS |
| ORIENTATION | PASS |
| INTEGRITY | PASS |
| MEDIUM150_PRETRAINING_GATE | **FAIL** |
| TRAINING_UNLOCKED | **FALSE** |
| DETECTOR_TRAINING_RUNS | 0 |
| AP_VALUES_INSPECTED | 0 |

The associated source hashes are recorded in `historical_medium150_tenengrad_failure_freeze.json`. A demonstrated implementation/reference defect may call calculation validity into question, but it still does not authorize correction, rerun, new candidate generation, detector training, or AP inspection.
