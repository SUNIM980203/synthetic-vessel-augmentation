**POST_RESULT_DIAGNOSTIC**

# Tenengrad distribution comparison

| distribution | n | mean | median | sample_sd | q05 | q25 | q75 | q95 | standardized_wasserstein_to_native |
|---|---|---|---|---|---|---|---|---|---|
| native_reference | 150 | 2.746095 | 2.249065 | 1.955767 | 1.138139 | 1.638506 | 2.789338 | 7.270350 | 0.000000 |
| generic_medium_high | 30 | 2.303861 | 2.103598 | 0.770525 | 1.318791 | 1.922233 | 2.625001 | 3.886406 | 0.313589 |
| generic_medium_low | 30 | 2.199630 | 1.949418 | 0.841568 | 1.262509 | 1.641731 | 2.797372 | 3.905866 | 0.297953 |
| historical_medium_high | 150 | 1.723393 | 1.642968 | 0.803762 | 0.662396 | 1.060111 | 2.235772 | 3.194443 | 0.522916 |
| historical_medium_low | 150 | 1.653871 | 1.650239 | 0.854949 | 0.555687 | 0.859438 | 2.276812 | 3.193738 | 0.558463 |
| historical_retained_pool | 307200 | 1.799806 | 1.589122 | 0.954863 | 0.619822 | 1.027802 | 2.434814 | 3.580108 | 0.483845 |

The historical High and Low distributions are jointly lower than the native center, whereas the previous generic Medium selected distributions were closer to native. This is descriptive and not a detector-efficacy result.
