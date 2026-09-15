# Frozen candidate identity and uniqueness protocol

Frozen before opening hosts 121--150: `2026-08-14T09:30:00.360212+00:00`.

- Identity: full SHA-256 of canonical UTF-8 JSON; never truncated.
- RNG: separately derived collision-checked unsigned 64-bit state; never identity.
- Uniform raw budget: 2080/host; retained scientific pool: 2,048 unique valid final JPEGs/host.
- Deduplication: final decoded calibrated JPEG SHA-256, global cohort scope.
- Retention: validity, UID uniqueness, final-image uniqueness, canonical generation order, first 2,048/host.
- Frozen S3 selector, support representation, radiometry rules, family taxonomy, geometry, placement, contrast 1.25, and unsharp 0.00 remain unchanged.
- Any final-cohort failure is terminal for that cohort; no retry, host replacement, budget change, training, AP, or Small-scale branch.

TRAINING_UNLOCKED = FALSE

SMALL_BRANCH_OPENED = FALSE
