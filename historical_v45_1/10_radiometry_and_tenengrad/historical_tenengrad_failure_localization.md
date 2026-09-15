# Historical Medium150 Tenengrad failure localization

**POST_RESULT_DIAGNOSTIC**

Historical pretraining gate remains: **FAIL**

High Tenengrad Wasserstein: **0.522916**

Low Tenengrad Wasserstein: **0.558463**

Frozen threshold: **0.50**

Metric implementation parity: **EQUIVALENT**

Native reference parity: **TRUE**

Direct recomputation parity: **TRUE**

Candidate pool already shifted before selection: **FALSE**

S3 selection worsened Tenengrad: **TRUE**

Historical source-raster shift supported: **FALSE**

Resize/anisotropy association: **anisotropy Spearman rho=-0.034**

Scene concentration: **WEAK**

Appearance-family concentration: **TRUE**

Support/Tenengrad tradeoff: **selected rho=-0.277; candidate median rho=-0.229**

Slot allocation preserved: **TRUE**

Final diagnostic classification: **VALID_NEGATIVE_FEASIBILITY_RESULT**

Training remains: **LOCKED**

AP inspected: **0**

## Required localization answers

1. **Metric/reference identical?** Common source and reference hashes are identical; runtime provenance is classified EQUIVALENT because both completed runs did not separately freeze package versions.
2. **Shift present before S3?** FALSE; retained-pool W=0.483845.
3. **Did S3 worsen it?** TRUE; selected mean within-host percentile=0.407.
4. **Appearance-family concentration?** TRUE; selected family median range=1.099.
5. **Variant concentration?** Variant associations are reported, but mechanical slot redistribution alone does not establish causality.
6. **Historical source concentration?** Source-raster shift supported=FALSE; reuse effects are separately reported.
7. **Scene concentration?** WEAK; dominant scene fraction=0.660.
8. **Target scale relation?** Scale rho=-0.006.
9. **Resize anisotropy relation?** anisotropy Spearman rho=-0.034.
10. **Host/background relation?** Ring-sharpness rho=-0.246.
11. **Support-gap relation?** selected rho=-0.277; candidate median rho=-0.229.
12. **Strongest supported explanation.** The strongest retained evidence is a selector-associated shift: S3 selected candidates below the within-host Tenengrad center (mean percentile 0.407), moving a retained pool that remained within the frozen absolute threshold (W=0.484) to failed High/Low distributions. Negative support-Tenengrad associations and low selected spatial-frequency-family values were consistent with this localization, but do not prove that selection alone caused the failure.
13. **Unresolved.** Standalone resized-object and pre-composite intermediate stages were not retained, package versions were not independently serialized for both historical runs, and all associations remain post-result descriptive.

## Evidence table

| hypothesis_id | hypothesis | status | evidence_summary |
|---|---|---|---|
| H1 | Tenengrad implementation/reference mismatch | NOT_SUPPORTED | Common code/reference hashes match and selected JPEGs reproduce the frozen values. |
| H2 | Historical source-raster sharpness shift | PARTIALLY_SUPPORTED | Historical/generic source medians 0.742819/0.810981; the historical median is lower, but not by the strong >=10% diagnostic rule. |
| H3 | Source-raster reuse concentration | NOT_SUPPORTED | Reuse correlation 0.037; 58 unique sources underlie 150 frames. |
| H4 | Historical bbox resize/anisotropy effect | NOT_SUPPORTED | Anisotropy correlation -0.034. |
| H5 | Historical host/context effect | PARTIALLY_SUPPORTED | Host ring correlation -0.246. |
| H6 | Source-scene concentration effect | NOT_SUPPORTED | Dominant fraction 0.660; LOSO ranges 0.016/0.023. |
| H7 | Orientation-amendment variant redistribution effect | PARTIALLY_SUPPORTED | MECHANICAL_WITHIN_FAMILY_SUBSTITUTION; family budgets preserved=True. |
| H8 | S3 selector-induced low-Tenengrad selection | SUPPORTED | Pool W=0.484; selected mean percentile=0.407. |
| H9 | Support-Tenengrad tradeoff | PARTIALLY_SUPPORTED | Selected/candidate correlations -0.277/-0.229. |
| H10 | Frozen calibration interacts differently with historical frame | PARTIALLY_SUPPORTED | Pre/final stage metrics retained, but resized-object and pre-composite standalone stages were unavailable. |

The frozen FAIL is retained. No candidate was generated, no threshold or calibration was changed, no detector was trained, and no AP was inspected.
