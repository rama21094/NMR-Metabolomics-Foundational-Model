# Gate G2, settled: 45 pretraining runs on three scaling axes

## Headline

Both diversity and volume help, but **volume saturates below ~2,000 spectra and
diversity does not**. Neither closes the gap to classical ML, which still leads
by 0.059 (k=all) and 0.064 (k=10) over the best checkpoint in the sweep.

This **corrects** the 2026-09-18 deck, which said "diversity beats quantity"
based on nominal study counts and two confounded axes. The direction survives;
the reasoning behind it did not.

## The measurement that reframes everything

Diversity was being counted wrong. Nominal study count badly overstates it,
because seven of the twelve studies hold 1-76 rows between them and MTBLS798
holds 57%. Measured as **effective studies** -- `exp(entropy of the study
composition)` -- the picture changes:

| subset | nominal studies | **effective** | rows |
|---|---|---|---|
| full training corpus | 12 | **4.09** | 8,545 |
| rows010 … rows075 | 12 | 4.15 … 4.09 | 858 … 6,410 |
| studies3 / 6 / 9 | 3 / 6 / 9 | 1.57 / 2.30 / 3.67 | 5,846 … 8,339 |
| fixed2000_k12 | 12 | 6.88 | 1,992 |
| fixed4000_k12 | 12 | 5.49 | 3,996 |

Two consequences:

1. **The corpus is effectively a 4-study corpus.** "Twelve studies" was never
   twelve.
2. **Balanced subsampling buys diversity.** `fixed2000_k12` reaches an effective
   6.88 -- well above the full corpus's 4.09 -- by refusing to let MTBLS798
   dominate. More of the corpus is not automatically more diverse.

## Volume, measured where it is identifiable

The row axis is the clean test: effective diversity pinned at 4.09-4.15 while
rows vary 7.5x. Slope per doubling of rows, permutation test, 3 seeds each:

| label budget | slope | p | 858 | 2,137 | 4,274 | 6,410 |
|---|---|---|---|---|---|---|
| k=10 | +0.0060 | 0.023 | 0.551 | 0.575 | 0.574 | 0.568 |
| k=all | +0.0016 | 0.332 | 0.639 | 0.647 | 0.642 | 0.646 |

The k=10 slope is significant but the values are **not monotonic**: everything is
won between 858 and 2,137 rows, and it is flat or declining after. A log-linear
fit reads that first step as a slope. So the honest statement is *rows help up to
roughly 2,000 and then stop*, not *rows help*.

## Diversity, measured the same way

Pooled regression over all 45 checkpoints, both predictors standardised so the
coefficients are comparable, permutation test on each:

| label budget | effective studies | row count | R² |
|---|---|---|---|
| k=10 | +0.0076 per sd (p=0.0001) | +0.0089 per sd (p=0.0000) | 0.413 |
| k=all | +0.0091 per sd (p=0.0004) | +0.0074 per sd (p=0.0033) | 0.276 |

Per standard deviation the two look equal. That is misleading on its own, and the
row axis above is why: the row term is carrying a saturating effect that a linear
fit spreads across the whole range. Across the *observed* range, on the
fixed-budget axis where the two are near-orthogonal (corr −0.18), diversity is
worth +0.026 and rows +0.016 at k=all.

R² of 0.28-0.41 is the other half of the story: most of the variance between
these checkpoints is explained by neither axis.

## What this answers, and what it does not

**Answers G2.** More spectra from the studies we already have will not help --
that saturated below 2,000, and we have 8,545. More *studies* would, and the
effect has not saturated anywhere in the range this corpus can reach (effective
1.57 to 6.90).

**Does not answer** whether diversity keeps paying beyond an effective ~7. This
corpus cannot be made more diverse than that, so the question is outside its
reach. That is itself the G2 terminal branch: more publicly available
experimental data is needed, and specifically more *independent studies*, not
more spectra.

**Does not rescue SSL.** The best checkpoint in the whole sweep is studies9 at
0.672 (k=all) against classical 0.721. Diversity narrows the gap from ~0.10 to
~0.06; it does not close it.

## Bearing on G3 (synthetic data)

Weakly negative, and for a sharper reason than before. A generator trained on
this corpus reproduces an effective-4-study composition. The one intervention
shown to help is raising effective study count, which a generator fitted to these
studies cannot do -- it can resample the composition it was given, which is what
`fixed2000_k12` already does directly, and better. Worth one honest test, with a
low prior.
