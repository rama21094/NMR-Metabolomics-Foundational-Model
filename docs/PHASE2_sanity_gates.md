# Phase 2 sanity gates — results

Run on the 935 held-out spectra from 7 studies withheld from pretraining.
Effective rank = exp(entropy of the normalised eigenvalue spectrum).

## The reference that matters

Judging an embedding's effective rank against its own dimension is misleading —
by that measure the data itself looks collapsed. The right reference is the
intrinsic rank of the same spectra:

| representation | dim | effective rank |
|---|---|---|
| raw spectra, in-band | 116,716 | **15.3** |
| binned 0.02 ppm (what classical ML uses, 0.85 bal.acc) | 500 | **10.5** |

There are only ~15 effective dimensions of variation in these spectra. That is
the ceiling any encoder can preserve.

## Results

| objective | seeds | dim | effective rank | vs raw (15.3) | study probe |
|---|---|---|---|---|---|
| masking | 0–4 | 192 | 3.3 – 3.9 | 0.22 – 0.25 | 0.981 – 0.984 |
| jigsaw | 0–3 | 192 | 3.7 – 5.5 | 0.24 – 0.36 | 0.975 – 0.988 |
| **joint** | 0–3 | 960 | **11.7 – 17.9** | **0.76 – 1.17** | 0.980 – 0.987 |
| masking random-init | — | 192 | 1.2 | 0.08 | 0.984 |
| jigsaw random-init | — | 192 | 2.2 | 0.14 | 0.973 |
| joint random-init | — | 960 | 1.1 | 0.07 | 0.988 |

## Reading

**Pretraining works, on all three objectives.** Every trained checkpoint has
substantially higher effective rank than its own random-init control — joint
most dramatically, 15.6 against 1.1. Whatever else is true, the pretext tasks
are not leaving the backbone where they found it.

**Only joint preserves the data's intrinsic dimensionality.** At 11.7–17.9 it
sits at or above the raw data's 15.3 and well above the binned features' 10.5.
Masking and jigsaw compress to 3.3–5.5 — roughly a quarter of what is there —
discarding structure the classical pipeline keeps. This is a concrete, falsifiable
prediction for Phase 3: **joint should transfer best, and masking worst.** If
Phase 3 contradicts it, the effective-rank framing is wrong and should be dropped
rather than explained away.

It also reframes the prior negative result. §6 found masked SSL never beats
classical LogReg at any label budget. An encoder retaining ~3.5 of 15.3 available
dimensions is a mechanism for exactly that, and it is a different mechanism from
§19's "the pretext task is too easy" — this one says the task may be fine while
the *representation* throws information away.

**The study probe does not discriminate and must not be used as a gate.** Raw
spectra probe at 0.989 and random-init backbones at 0.973–0.988, so study identity
is trivially decodable from this data before any model touches it. It is reported
because that is a standing confound risk for every downstream result — the same
risk that disqualified MTBLS326 — not because any checkpoint failed.

## Caveat

Effective rank is a summary statistic, and joint's advantage is partly a larger
embedding (960 vs 192). The comparison against each objective's own random-init
control is the part that controls for architecture; the cross-objective
comparison does not. Phase 3 is the real test.
