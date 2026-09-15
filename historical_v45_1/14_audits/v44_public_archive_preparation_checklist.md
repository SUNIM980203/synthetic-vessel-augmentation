# v44 Public Archive Preparation Checklist

Status: repository URL/DOI not supplied. Do not claim public deposit or full reproducibility.

The planned package should be described as **auditable but not fully independently recomputable** because restricted imagery, checkpoints, per-image predictions, embeddings, and local historical inputs are not redistributed.

| Sanitized/public item where licensing permits | Required action | Boundary |
|---|---|---|
| Evidence hierarchy | Include | Preserve primary/secondary/post-result labels. |
| Frozen protocol summaries | Include | Do not rewrite historical decisions. |
| Exact hashes | Include | Hash restricted artifacts without distributing them. |
| Split manifests | Include if distributable | Exclude licensed imagery and restricted location metadata as required. |
| Aggregate result CSV/JSON | Include | No raw predictions. |
| Numerical consistency audits | Include | Mark checks as audits, not new results. |
| Same-scale stopping-decision records | Include | Preserve `MEDIUM150_PRETRAINING_GATE = FAIL`. |
| Candidate-slot equivalence report | Include | Do not include candidate imagery or binaries. |
| Tenengrad method definition | Include | Describe the retained executed implementation. |
| Statistical-analysis scripts | Include | Remove local paths and restricted inputs. |
| Figure/table generation scripts | Include | Confirm they consume distributable aggregates only. |
| Software environment provenance | Include | Report only source-verified versions. |
| README for unavailable third-party assets | Include | Explain licensed imagery, checkpoints, predictions, embeddings, and local historical inputs. |

Author action remains required: create the sanitized public deposit, verify its contents and licensing, then provide the genuine repository URL/DOI.
