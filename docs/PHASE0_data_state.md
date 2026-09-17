# Phase 0 — frozen data state and corpus diagnostics

## Datasets (all on one byte-identical ppm axis, +10.0 to -0.5, 131,072 points)

| dataset | rows | usable | role |
|---|---|---|---|
| corpus serum | 1,961 | 1,961 | pretraining |
| corpus plasma | 6,739 | 6,739 | pretraining |
| corpus workbench | 780 | 780 | pretraining |
| **corpus total** | **9,480** | | |
| MTBLS563 | 142 | 113 (excl. 29 "unknown") | evaluation |
| BrC-T2D | 115 | 115 | evaluation |
| MTBLS326 | 43 | 41 | **excluded from claims** (see below) |
| Barth | 40 | 37 (excl. 3 pooled QC) | evaluation |

Class balances: MTBLS563 control 56 / viral 31 / bacterial 26; BrC-T2D BC-ND 38,
NC-D 38, NC-ND 22, BC-D 17; Barth case 23 / control 14.

**Leakage:** MTBLS326, MTBLS563 and ST000223 appear zero times in the corpus,
checked by study accession across all 9,480 provenance rows.

**MTBLS326 stays out of the evidence base.** SSL_vs_classical_analysis.md §11
showed its labels are collinear with acquisition order by design (cases 1-27,
controls 101-130) and that metabolite-free regions classify it at 0.726. The
rebuild does not touch that: it is a study design problem, not a processing one.

**Corpus composition:** 18 distinct studies; MTBLS798 alone is 4,866 rows (51%).
Large in spectra, small in studies.

## Corpus diagnostics re-measured after the rebuild

§19 and §20 of SSL_vs_classical_analysis.md characterised the corpus as
redundant and narrow. Both were measured on the defective corpus -- the one whose
0 ppm anchoring drove the 95% peak-position span to exactly 0.000 ppm. Forced
alignment inflates every similarity statistic, so those numbers needed re-taking.

| diagnostic | old pipeline | rebuilt | reading |
|---|---|---|---|
| variance in first 5 PCs | 86% | **57.7%** | the corpus is substantially richer than measured |
| variance in first 10 PCs | -- | 73.6% | |
| variance in first 50 PCs | -- | 95.5% | |
| within-corpus nearest-neighbour r | 0.991 | **0.989** | unchanged -- local near-duplication is real |
| cohort -> corpus nearest match r | 0.37-0.78 | **0.685-0.839** | the domain gap is real but smaller than thought |

Measured on 2,100 sampled spectra, 0.5-9.0 ppm, water window excluded.

**What this means for the prior negative results.** §19 argued that masked
reconstruction is nearly free because 86% of variance sits in 5 components, so
the pretext task cannot force the model to learn anything disease-discriminative.
At 57.7% that argument is materially weaker, though the unchanged nearest-
neighbour figure means copy-the-neighbour remains a strong reconstruction
baseline. §20's "narrow, not merely small" is softened: cohort coverage improved
from 0.37-0.78 to 0.69-0.84.

Neither finding is overturned, and neither can be inherited. **The prior
negatives (§6, §18) were measured on data with a known geometric defect and must
be re-established, not assumed.** That is Phases 1-3.

## Protocol (fixed before any result is seen)

- MTBLS563 (n=113) and BrC-T2D (n=115): repeated stratified 10-fold, 10 repeats.
- Barth (n=37): LOOCV across 10 seeds.
- Primary metric balanced accuracy; macro-F1 and AUC secondary. Mean with 95% CI
  across repeats. No single-split numbers.
- **Noise floor: the measured single-run sd is 0.045** (§15). A difference
  smaller than that is not a result.
- Hyperparameters selected only inside a nested inner loop.
- Corpus splits are **by study**, never by row.
