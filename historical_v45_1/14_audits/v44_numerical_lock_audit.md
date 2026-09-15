# v44 Numerical Lock Audit

No numerical result was added or changed. The v43 and v44 main numeric-token multisets are identical, and the v43 and v44 supplement numeric-token multisets are identical.

| Check | Frozen | Source-observed | Source | Status |
|---|---:|---:|---|---|
| xView U-D | 0.007947 | 0.007946996 | `release/ieee_access_reproducibility_submission/results/ten_seed_head_only_results.csv` | PASS |
| xView U-R | 0.002402 | 0.002401645 | `release/ieee_access_reproducibility_submission/results/ten_seed_head_only_results.csv` | PASS |
| HRSC U-D | 0.012097 | 0.012096789 | `release/ieee_access_reproducibility_submission/results/ten_seed_head_only_results.csv` | PASS |
| HRSC U-R | 0.005035 | 0.005035098 | `release/ieee_access_reproducibility_submission/results/ten_seed_head_only_results.csv` | PASS |
| DIOR U-D | 0.005525 | 0.005525134 | `release/ieee_access_reproducibility_submission/results/ten_seed_head_only_results.csv` | PASS |
| DIOR U-R | 0.004363 | 0.004363291 | `release/ieee_access_reproducibility_submission/results/ten_seed_head_only_results.csv` | PASS |
| xView-validation vessel DiD | 0.008532 | 0.008531760 | `release/ieee_access_reproducibility_submission/results/scope_summary_metrics.csv` | PASS |
| xView-validation vessel CI low | 0.005423 | 0.005422857 | `release/ieee_access_reproducibility_submission/results/scope_summary_metrics.csv` | PASS |
| xView-validation vessel CI high | 0.011641 | 0.011640662 | `release/ieee_access_reproducibility_submission/results/scope_summary_metrics.csv` | PASS |
| Historical support median | 0.074731 | 0.074731022 | `AnalysisResults/expanded_v1/medium150_historical_pretraining_gate/medium150_pretraining_gate_results.json` | PASS |
| High Tenengrad W | 0.522916 | 0.522915935 | `AnalysisResults/expanded_v1/medium150_historical_pretraining_gate/medium150_radiometry_gate_metrics.csv` | PASS |
| Low Tenengrad W | 0.558463 | 0.558463375 | `AnalysisResults/expanded_v1/medium150_historical_pretraining_gate/medium150_radiometry_gate_metrics.csv` | PASS |
| Retained-pool Tenengrad W | 0.483845 | 0.483845301 | `AnalysisResults/expanded_v1/medium150_historical_pretraining_gate/tenengrad_distribution_comparison.csv` | PASS |

Additional source-preservation checks:

- xView vessel-positive acquisition scenes remain 16/4/4 for train/validation/test.
- The zero-target convention, 3/2,000 test replicates, AP=0 handling, percentile intervals, and 1,997 retained nonzero-target replicates are unchanged.
- Unique-annotation dose remains 150/3,727 = 4.0247%; the two-stream vessel-label fraction remains 150/7,604 = 1.9726%. These are distinct denominators.
- `MEDIUM150_PRETRAINING_GATE` remains FAIL; detector-training and AP counters remain zero.

This script calculated only consistency deltas and source means already represented by the frozen ten-seed aggregate; it produced no new scientific result.
