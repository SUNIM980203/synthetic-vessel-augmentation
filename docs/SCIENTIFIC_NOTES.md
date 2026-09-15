# Scientific interpretation and formulas

For each model and paired seed i, the factorial is Condition (Unity Medium150, Duplicate) × scope (head-only, full-network) × LR (1e-4, 2e-4). Ten paired seeds give 80 runs per model. Use the model-specific preserved protocol for architecture and execution details.

For AP outcome Y, define delta(i,s,l) = Y(i,Unity,s,l) - Y(i,Duplicate,s,l).
DiD(i,l) = delta(i,head,l) - delta(i,full,l).
MarginalDiD(i) = [DiD(i,1e-4) + DiD(i,2e-4)] / 2.
ThreeWay(i) = DiD(i,2e-4) - DiD(i,1e-4).
The seed-level mean interval is mean ± t(0.975,n-1) × sample_SD/sqrt(n).
The included compact recomputation uses n=10 and t(0.975,9)=2.2621571628540993.
AP is on a 0–1 scale; multiply differences by 100 for percentage points.

Model-size comparisons are paired descriptive comparisons, not causal size effects. Scene sensitivity is not a population confidence interval. The studies reuse evaluation sets; do not label them globally untouched confirmation. Negative Faster R-CNN and feasibility stopping outcomes remain in the historical archive.

YOLO26m ordinal 42 recorded infinite validation classification loss at epochs 10, 12 and 14. The cause is unresolved. All 80 prespecified final checkpoints were retained; finite final tensors do not establish an error-free trajectory. The seed-removal analysis is post-hoc, not a corrected primary analysis. Intervals spanning zero do not prove no effect.

YOLO26m postprocessing stopped at its final report gate after evaluation/analysis; final reporting was reconciled separately. It must not be described as an uninterrupted original-pipeline PASS. See the retained final report and provenance addendum.
