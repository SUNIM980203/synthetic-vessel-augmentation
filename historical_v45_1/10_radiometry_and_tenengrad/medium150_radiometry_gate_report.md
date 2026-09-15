# Medium150 radiometry gate

- High absolute Weber standardized Wasserstein: 0.3864619718043419 (<=0.50)
- Low absolute Weber standardized Wasserstein: 0.3824895591364185 (<=0.50)
- Other frozen native-radiometry metrics: FAIL
- Pairwise radiometry: PASS
- Overall radiometry: **FAIL**

| gate | arm | metric | standardized_wasserstein | median_absolute_standardized_difference | pass |
|---|---|---|---|---|---|
| absolute | high | absolute_weber_contrast | 0.386462 |  | True |
| absolute | high | local_cnr | 0.254832 |  | True |
| absolute | high | tenengrad_ratio | 0.522916 |  | False |
| absolute | high | laplacian_variance_ratio | 0.421557 |  | True |
| absolute | high | lab_delta_e76 | 0.486380 |  | True |
| absolute | low | absolute_weber_contrast | 0.382490 |  | True |
| absolute | low | local_cnr | 0.247020 |  | True |
| absolute | low | tenengrad_ratio | 0.558463 |  | False |
| absolute | low | laplacian_variance_ratio | 0.436403 |  | True |
| absolute | low | lab_delta_e76 | 0.482030 |  | True |
| pairwise | high_low | absolute_weber_contrast |  | 0.068550 | True |
| pairwise | high_low | local_cnr |  | 0.063415 | True |
| pairwise | high_low | tenengrad_ratio |  | 0.089443 | True |
| pairwise | high_low | laplacian_variance_ratio |  | 0.038490 | True |
| pairwise | high_low | lab_delta_e76 |  | 0.048829 | True |