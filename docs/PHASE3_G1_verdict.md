# Phase 3 — Gate G1 verdict

**Whiteboard node:** "ML is better — sets bar, can we better it"

**Answer: no.** Across 6 targets × 5 label budgets, on correctly-aligned data,
with 5 pretraining seeds per objective and 50 paired episodes per cell:

| | cells |
|---|---|
| SSL wins | **0** |
| classical wins | 24 |
| tie (within the 0.045 noise floor) | 5 |

Mean balanced accuracy over all targets:

| k per class | classical | masking | jigsaw | joint |
|---|---|---|---|---|
| 5 | **0.597** | 0.549 | 0.498 | 0.516 |
| 10 | **0.653** | 0.593 | 0.531 | 0.548 |
| 20 | **0.705** | 0.633 | 0.561 | 0.580 |
| 40 | **0.738** | 0.672 | 0.587 | 0.611 |
| all | **0.729** | 0.657 | 0.573 | 0.607 |

Classical ML is ahead at every budget, including the smallest.

## The result that is positive

Pretraining clearly works — it just does not work well enough. Against a
random-init backbone of identical architecture on identical episodes:

| objective | k=5 | k=10 | k=20 | k=40 | all |
|---|---|---|---|---|---|
| masking | +0.081 | +0.107 | +0.112 | +0.121 | +0.117 |
| jigsaw | +0.059 | +0.082 | +0.090 | +0.109 | +0.079 |
| joint | +0.015 | +0.017 | +0.009 | +0.032 | +0.072 |

Masking gains ~0.11 balanced accuracy over its own untrained backbone, well
above the noise floor and consistent across all six targets. The pretext task is
teaching the encoder something real. That something is simply worth less than
what a shrinkage LDA extracts from a 500-bin spectrum.

## Two claims retracted

**Effective rank does not predict transfer.** Phase 2 found only joint preserved
the data's intrinsic rank (15.6 of 15.3, against masking's 3.5) and I predicted
joint would transfer best and masking worst. The opposite is true at every
budget: masking is the strongest objective and joint is middling. The rank
measurement stands as a fact about the representations; its predictive value
does not, and it is dropped as a selection criterion.

**"SSL beats classical on Barth" — again a selection artefact.** Reporting the
best of 15 checkpoints put Barth at +0.057 over classical at k=20. Per objective,
the seed means are 0.707 / 0.663 / 0.635 against classical's 0.770 — all clearly
below. §18 retracted this same claim for this same reason; the summariser now
reports seed means as the headline and prints best-of-15 only as a marked upper
bound.

## What this does and does not establish

It establishes that the **axis defect was not the reason SSL was losing.** The
corpus is now geometrically correct, every cohort is on the same axis, and the
direction of the comparison is unchanged from §6. That is worth knowing: it
closes off "the data was broken" as an explanation.

It does **not** establish that the foundation-model approach fails. Under the
whiteboard, G1 failing is not terminal — it is the trigger for G2, "is data
limiting". Specifically:

* the corpus is 18 studies with one contributing 51%;
* cohort-to-corpus nearest-neighbour similarity is 0.685–0.839, so the evaluation
  domains sit at the edge of what was pretrained on;
* only frozen linear probes were tested here, not fine-tuning.

## Next

Phase 4 (gate G2). Corpus-size scaling subsampled **by study**, peak-intensity
distributions, and whether p(y|x) is defined across the informative range — which
decides whether synthetic data is legitimate or whether the honest conclusion is
that more public experimental data is needed.
