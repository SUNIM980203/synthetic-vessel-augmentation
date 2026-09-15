
# RealCutout method audit

**Status: PASS with explicit limitations.** The implemented control is accurately described as a host-, placement-, target-box-, count-, exposure-, and largely compositing-matched cross-scene real-vessel control. It is **not** a pure appearance-only control and does not use ground-truth instance masks.

## Source and reuse

- Source pool: xView training vessels only; source-box geometric-mean scale in `[24, 80)` pixels.
- Target host scene was excluded from the source candidates.
- Realized library: 209 candidates from 11 source scenes.
- 150 insertions used 104 unique cutouts; maximum reuse was 3; same-scene insertion count was 0.

## Crop and alpha mask

The source code begins from a padded annotated bbox (18% of the larger side; minimum four pixels) and rejects overlaps with another annotation. It estimates background from the two-pixel crop border and thresholds Euclidean RGB distance at 22. A 3×3 morphological open/close is followed by center-weighted connected-component selection, nearby-component retention, a 3×3 Gaussian blur (σ=0.65), tight alpha cropping, and an occupancy gate. This is heuristic image-derived segmentation, not a rectangular crop and not ground-truth segmentation. It can retain source water/context or omit vessel pixels.

## Selection, resizing, and compositing

Candidates are ordered by absolute log aspect mismatch; a shortlist of 12 is then reuse balanced, with a seeded tie break. The RGBA crop is resized with Lanczos to the exact target width and height. Reference geometric-mean scale matches exactly; measured absolute log aspect distortion was 0.0733 at the median and 0.4282 at the 95th percentile. The shared sensor-aware vessel path performs local brightness/color adjustment, weak point-spread blur, a low-strength blurred shadow, and RGBA alpha compositing. There is no histogram-matching stage or constant-opacity paste.

## Matched and unmatched factors

Matched: host image, insertion position, target width/height/scale, object count, modified-image count, total-image exposure, and shared compositing function/settings.

Not matched or not fully isolated: object/source appearance, alpha provenance/quality, source-context contamination, possible object-pixel omission, aspect stretching, and Unity-versus-heuristic input-alpha differences.

## Figure 6 audit

The training pair was selected by the preserved middle-scale rule, not detector performance or boundary visibility. The panel shows no basis for claiming that crop boundaries are generally obvious, subtle, or representative. The caption now identifies the mask provenance and states that the panel does not estimate boundary prevalence or effect size. Figure SHA-256: `d98fcf1af9b03b4a877d2be42da4e112613ab082f01dd14fecca4d84709662be`.

## Manuscript action

The main Methods and Limitations and supplementary S1A now disclose all material construction choices and confounding boundaries. No result or claim was added.
