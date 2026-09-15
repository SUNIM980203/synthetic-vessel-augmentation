# Vessel detector-feature support report

## Design

A frozen seed-20260723 Duplicate YOLO26n detector produced P3/P4/P5 ROI-aligned embeddings for real training vessels and each synthetic vessel condition. Metrics use training objects only; no validation or test predictions are involved. Nearest-real support is computed within bbox-scale bins and always excludes candidates from the same acquisition scene.

## Outcome

medium150 has the lowest scale-matched MMD, while medium150 has the highest cross-scene support fraction (0.740). A support fraction far below 0.5 means most synthetic vessels remain outside the feature neighborhoods normally occupied by real vessels. Compare effective rank with real to distinguish excessive unsupported variation from simple diversity shortage.

## Feature support

| Domain | Objects | NN distance ratio vs real | Supported at real p95 | Scale-matched MMD2 | Effective rank |
|---|---:|---:|---:|---:|---:|
| allscale150 | 150 | 1.859 | 0.360 | 0.1907 | 17.74 |
| medium150 | 150 | 1.263 | 0.740 | 0.0612 | 17.03 |

## Scale-specific support

### allscale150

| Scale | Synthetic n | NN ratio | Supported fraction |
|---|---:|---:|---:|
| tiny_lt16 | 53 | 1.720 | 0.226 |
| small_16_32 | 91 | 2.213 | 0.407 |
| medium_32_64 | 6 | 1.395 | 0.833 |

### medium150

| Scale | Synthetic n | NN ratio | Supported fraction |
|---|---:|---:|---:|
| medium_32_64 | 150 | 1.263 | 0.740 |

## Interpretation

A nearest-real distance ratio above 1 means synthetic objects lie farther from real examples than real objects lie from other real examples at the same bbox scale. The support fraction uses the real leave-one-out p95 distance as a scale-specific threshold. MMD measures the distribution difference after matching the real sample to the synthetic scale-bin frequencies. Effective rank is an embedding-diversity indicator, not a quality score.

Use this analysis to distinguish unsupported synthetic appearance from insufficient diversity. It is diagnostic and must not be used to unlock the held-out test split.
