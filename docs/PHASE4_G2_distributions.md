# Phase 4.2 / 4.3 — is data limiting? (gate G2, distribution half)

**Whiteboard:** *"distributions of peak intensities — how they change w/ increasing
data — have we defined experimental y-distribution at all x. If yes, use synthetic
data. If no, we end — more publicly available exptl data needed."*

## Method

At each ppm position x, p(y|x) is the distribution of binned intensities across
spectra. "Defined" means it has stopped moving as spectra are added. Comparing
against the full-corpus estimate would be circular, so two **disjoint** samples of
size N are drawn, p(y|x) is estimated independently in each, and the
Wasserstein-1 distance between them is reported relative to the pooled IQR.

Two splits, because they answer different questions:

* **row split** — both halves from the same 12 studies. Measures **sampling error**.
* **study split** — halves from disjoint sets of studies. Measures **between-study
  variation**.

## Result

| N per half | row split (sampling) | study split (between-study) |
|---|---|---|
| 100 | 0.304 | 0.908 |
| 250 | 0.199 | 0.806 |
| 500 | 0.143 | 0.959 |
| 1,000 | 0.098 | 0.772 |
| 2,000 | 0.082 | 1.037 |
| 4,000 | **0.053** | — |

| | bins with distance < 0.25 |
|---|---|
| row split, N=4,000 | **99.8%** (467 of 468) |
| study split, N=2,000 | 1.5% (7 of 468) |

The row-split curve falls almost exactly as 1/√N — a 40× increase in N cuts the
distance 5.7×, against the 6.3× that pure sampling error predicts. The estimator
works, and by N = 4,000 the distribution is pinned down across essentially the
whole spectrum.

The study-split curve does not fall at all. It is flat and noisy between 0.77 and
1.04 across a 20× range of N, which is the signature of an irreducible systematic
difference rather than an unconverged estimate.

## Answer to G2

**Data quantity is not the constraint. Data diversity is.**

*Have we defined the experimental y-distribution at all x?* For the studies we
have: **yes**, and comprehensively — 99.8% of ppm bins, with sampling error below
6% of the IQR. Adding more spectra from these 12 studies would change nothing.

Universally: **no**. p(y|x) is study-specific, and 12 studies do not determine
what a thirteenth looks like.

This resolves the whiteboard's branch in a way neither arm anticipated. Synthetic
data generated from this corpus would be **legitimate but useless for the stated
goal**: the distribution it would sample from is well measured, so the generator
would be sound, but it would reproduce the corpus's own 12-study composition and
would not cover a new cohort. The generalisation gap seen in Phase 3 —
cohort-to-corpus nearest-neighbour similarity of only 0.685–0.839 against 0.989
within the corpus — is a direct consequence.

So the honest terminal statement is the whiteboard's "more publicly available
experimental data is needed", with one word added: more **diverse** experimental
data. Not more spectra — more studies, more instruments, more populations.

## Caveat

The "defined" threshold of 0.25 IQR is a choice, not a derived quantity. The
conclusion does not depend on it: at any threshold between 0.10 and 0.50 the row
split resolves the great majority of bins and the study split resolves almost
none. What would change the conclusion is a different notion of distance, and
Wasserstein-1 on marginals ignores correlations between ppm positions — a
generator could match every marginal and still produce spectra that are not
plausible mixtures.

## Confirmation pending

Phase 4.1's fixed-budget axis tests the same claim in the currency that matters:
does a model pretrained on 2,000 rows from 12 studies beat one trained on 2,000
rows from 2 studies? If diversity is the constraint, it should.
