**POST_RESULT_DIAGNOSTIC**

# Medium candidate-slot allocation equivalence

- `FAMILY_LEVEL_SLOT_BUDGET_PRESERVED = TRUE`
- `CANDIDATE_SLOT_CHANGE_CLASS = MECHANICAL_WITHIN_FAMILY_SUBSTITUTION`
- Both branches have exactly 2080 raw candidates per host.

| branch | family | host_count | minimum_slots_per_host | maximum_slots_per_host | mean_slots_per_host | total_slots |
|---|---|---|---|---|---|---|
| pre_amendment_generic | baseline | 30 | 128 | 128 | 128.000000 | 3840 |
| pre_amendment_generic | fine_detail | 30 | 256 | 256 | 256.000000 | 7680 |
| pre_amendment_generic | material_chroma | 30 | 256 | 256 | 256.000000 | 7680 |
| pre_amendment_generic | spatial_frequency | 30 | 640 | 640 | 640.000000 | 19200 |
| pre_amendment_generic | texture_realization | 30 | 800 | 800 | 800.000000 | 24000 |
| post_amendment_historical | baseline | 150 | 128 | 128 | 128.000000 | 19200 |
| post_amendment_historical | fine_detail | 150 | 256 | 256 | 256.000000 | 38400 |
| post_amendment_historical | material_chroma | 150 | 256 | 256 | 256.000000 | 38400 |
| post_amendment_historical | spatial_frequency | 150 | 640 | 640 | 640.000000 | 96000 |
| post_amendment_historical | texture_realization | 150 | 800 | 800 | 800.000000 | 120000 |

The standalone audit is source-backed by the retained generic raw manifest and all 150 historical candidate identity manifests. It does not assume the expected counts when artifacts disagree.
