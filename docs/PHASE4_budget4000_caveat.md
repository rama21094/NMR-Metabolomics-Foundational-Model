# The 4,000-row fixed-budget axis, and what it can and cannot separate

## Why it was run

Gate G2's fixed-budget sweep at 2,000 rows showed transfer saturating after 4
studies. That is ambiguous: it could mean four studies are enough, or it could
mean the 2,000-row budget was the binding constraint and study count never got a
chance to matter. Re-running at 4,000 rows was the cheap way to separate them.

## What building the subsets revealed

The separation is not clean, because of the corpus composition:

| study | train rows |
|---|---|
| MTBLS798 | 4,866 (57%) |
| MTBLS147 | 979 |
| MTBLS395 | 978 |
| MTBLS540 | 716 |
| MTBLS424 | 699 |
| seven others | 1–76 each, 307 total |

The builder gives each of the k studies a quota of BUDGET//k rows. The small
studies cannot fill their quota, so the shortfall is redistributed to the large
ones — and the larger the budget, the bigger the shortfall. Effective study count
(exp of the entropy of the study composition, mean of 3 seeds):

| budget | k=2 | k=4 | k=8 | k=12 |
|---|---|---|---|---|
| 2,000 | 2.00 | 4.00 | 6.20 | 6.89 |
| 4,000 | 1.74 | 3.91 | 5.39 | 5.55 |

So the 4,000-row axis spans a **narrower** diversity range than the 2,000-row
axis, not an equal one at higher volume. Doubling the budget trades diversity for
rows rather than holding diversity fixed.

## How the result must therefore be read

Comparing `fixed4000_k12` against `fixed2000_k12` confounds two changes in
opposite directions: +2,000 rows and −1.34 effective studies. A null result there
means nothing on its own.

The analysis instead pools all 24 fixed-budget checkpoints (2 budgets × 4 k × 3
seeds) and regresses transfer on **effective** studies with row count as a
covariate. That uses the budget difference as extra leverage on the row axis
rather than as a separate comparison, and it is the only reading the design
supports.

## The limitation this exposes

Nominal k overstated diversity in the 2,000-row sweep too — "12 studies" was
effectively 6.89, because seven of those studies contribute 307 rows between
them. The saturation reported on slide 6 of the 2026-09-18 deck is real, but it
saturates at an effective ~4, and the corpus cannot reach an effective 12 at any
budget. Whether diversity keeps paying beyond that is a question this corpus
cannot answer — which is itself the G2 answer, stated more sharply.
