# Phase 6: capacity helps, slowly, and does not close the gap

## The prediction, and that it was wrong

I predicted that models *smaller* than the 4.0 M baseline would do as well or
better, because the G1 fine-tuning result showed the baseline already overfits
40–142 sample cohorts. That prediction was **wrong in direction**, and the sweep
was run two-sided precisely so it could be.

Smaller is worse (−0.016 at 1.8 M), larger is better (+0.013 at 10.5 M). Largest
beats smallest in **6 of 6 targets at both label budgets** — not a noisy
preference.

The fine-tuning result does not actually license the prediction I drew from it.
Fine-tuning overfits because it adapts ~4 M parameters on 10–40 labelled support
spectra. Pretraining fits the same parameters on 8,545 unlabelled spectra, where
there is enough signal to support more capacity. Two different regimes; I
conflated them.

## The measurement

12 runs, 4 new architectures × 3 seeds, baseline reused from Phase 2. Probed
exactly as Phase 3 probed G1, 6 targets × 2 label budgets.

| params | d_model / layers | k=10 | k=all | seed sd |
|---|---|---|---|---|
| 1.82 M | 96 / 2 | 0.574 | 0.644 | 0.022 |
| 2.38 M | 128 / 3 | 0.575 | 0.647 | 0.025 |
| 3.99 M | 192 / 4 (baseline) | 0.590 | 0.666 | 0.025 |
| **10.54 M** | 320 / 6 | **0.603** | **0.678** | 0.016 |
| 23.52 M | 448 / 8 | 0.598 | 0.675 | 0.016 |
| **classical ML** | — | **0.654** | **0.721** | — |

## Why this still does not rescue the model

Three reasons, in decreasing order of strength.

**1. The curve has already turned over.** 23.5 M is below 10.5 M at both budgets.
The difference (−0.005, −0.003) is inside the seed sd of 0.016, so the honest
reading is a **plateau from ~10 M onward**, not a decline — but it is certainly
not still climbing.

**2. The rate is hopeless even if it were still climbing.** +0.0077 per doubling
at k=10, +0.0093 at k=all. Against a residual gap of 0.056 and 0.046 at the best
model, that is 5–7 further doublings — 300 M to 3 B parameters — pretrained on
8,545 spectra from an effective 4.09 studies. Reporting that extrapolation as a
route would be dishonest twice over: the curve is flat by 23.5 M, and the corpus
cannot support such a model regardless.

**3. Best capacity still loses everywhere.** 0.678 against classical 0.721 at
k=all; 0.603 against 0.654 at k=10. No architecture in the sweep wins any cell.

## Verdict for the tree

The whiteboard's branch is *"if no — more parameters ... if no, then we hit a
limit on these tasks."* Capacity was measured, not argued. It helps by
+0.012 and plateaus, leaving a gap of 0.046.

**Terminal node reached: these tasks have a ceiling at this data scale.**

## What changes in the write-up

The consolidation previously justified *not* running this sweep with the
reasoning that the encoder already overfits so added capacity would hurt. That
reasoning is now refuted by measurement and has been removed. The conclusion it
supported is unchanged — capacity is not the lever — but it now rests on data
rather than on an inference that turned out to be wrong.

This is the difference between a judgement and a result, and it is why the sweep
was worth running.
