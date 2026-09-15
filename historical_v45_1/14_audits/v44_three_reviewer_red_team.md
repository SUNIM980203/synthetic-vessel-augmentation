# v44 Three-Reviewer Red Team

## Reviewer A - Remote sensing / domain transfer

- Strongest major concern: vessel targets occupy only four validation and four test acquisition scenes, while the external datasets were used earlier in the broader study. A Reject argument would be that domain generalization is too narrow.
- Strongest minor concern: DIOR absolute AP is low and its mirror is not the official test partition.
- Does v44 preempt it? PARTLY. It discloses 16/4/4 positive-scene concentration, interprets the bootstrap as sparse-cluster sensitivity, and labels external estimates secondary post-lock rather than globally untouched.
- Would a full fix require new experiments? YES - new acquisition scenes and an untouched external cohort.
- Does the bounded claim remain valid? YES. The claim is conditional on the tested splits/domains and does not assert broad remote-sensing generalization.

## Reviewer B - Deep learning / object detection

- Strongest major concern: the support representation comes from YOLO and the effect failed the tested Faster R-CNN gate. A Reject argument would be detector-family circularity and lack of architecture transfer.
- Strongest minor concern: head-only and full-network use different learning rates, so freezing alone is not isolated.
- Does v44 preempt it? YES AS A LIMITATION. Support is detector-specific; Faster R-CNN failure is prominent; the comparison is explicitly between scope-specific regimens.
- Would a full fix require new experiments? YES - external encoders, additional architectures, and a design isolating scope from schedule.
- Does the bounded claim remain valid? YES. It is limited to seed-consistent YOLO head-only effects and heterogeneity across the tested regimens.

## Reviewer C - Statistics / experimental design

- Strongest major concern: paired-seed intervals do not represent sampling uncertainty over scenes/domains, and the historical validation endpoint is reused. A Major Revision argument would request new independent data.
- Strongest minor concern: the object-level support reference is not equal-scene weighted.
- Does v44 preempt it? YES IN WORDING, NOT IN EVIDENCE. It separates seed from scene uncertainty, concedes validation reuse, and discloses support-reference weighting.
- Would a full fix require new experiments? YES - new independent scenes/cohorts and a prospectively specified scene-balanced support design.
- Does the bounded claim remain valid? YES. It remains a conditional training-stochasticity result with sparse-cluster sensitivity, not a population-domain estimate.

## Red-team disposition

All three reviewers can still identify legitimate intrinsic limitations. None can fairly claim that v44 hides them or answers them by strengthening the scientific claim.
