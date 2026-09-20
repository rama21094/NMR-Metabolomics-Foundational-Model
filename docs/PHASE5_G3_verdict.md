# Gate G3: synthetic data fails validation and is not trained on

## The rule this follows

Research plan §5.2: *"Synthetic data that fails these should not be trained on,
whatever it does to the metric."* It failed. No augmented pretraining was run, and
that is the finding, not an omission.

## What was built

GISSMO metabolite basis, rebuilt on the corpus ppm axis (the existing basis was
on a pre-rebuild axis and misplaced every peak by up to 1.0 ppm, commit 553c2b9):
87 serum-relevant metabolites, pinned by bmse ID, condition number 9.6, plus 7
lipid/macromolecule envelope components. Landmarks within 0.006 ppm of
literature.

Spectra were synthesised as non-negative linear combinations, with NNLS-fitted
coefficients from real spectra, random referencing offsets matched to the
measured 0.05 ppm between-study spread, and noise at the real corpus level. Two
sampling modes: `joint` (resample whole real coefficient vectors) and `marginal`
(sample each metabolite independently, which can produce combinations no real
study contained -- the property a trained generator cannot have).

## Why it failed

| test | synthetic | reference |
|---|---|---|
| discriminator AUC, real vs synthetic | **1.000** | 0.5 = indistinguishable |
| median per-bin Wasserstein / IQR | **9.72** | 0.053 within-corpus floor |
| bins within that floor | **0.0%** | — |
| lactate CH₃→CH separation | 2.781 | 2.780 expected |

The chemistry is right and the realism is not. Landmark positions are essentially
exact in `joint` mode, so the basis is correctly built and correctly placed.
(`marginal` mode shifted landmarks by ~0.1 ppm -- independent sampling produces
chemically impossible mixtures -- which is itself a limit on the one advantage
this route had over a generator.)

The cause is coverage, and it is measurable:

**NNLS fit of the 94-component basis to real spectra: median R² = 0.25**
(mean 0.28, range 0.11–0.60, none above 0.61).

Three quarters of a real spectrum is signal the basis cannot represent. The
synthetic spectra are consequently sparse where real ones are dense -- 54% of
bins have a near-zero median against 13% for real, and 233 of 468 bins carry
signal against 381 for real. A discriminator separates them perfectly on that
alone.

Adding the lipid basis changed nothing to three decimal places. That is expected
in hindsight and worth recording: **this corpus is CPMG-only**, and the CPMG
sequence exists to suppress broad macromolecule signal. Lipids were never the
missing component.

## The verdict

For the linear-combination route, G3 is **no**. Not because synthetic data failed
to help, but because it cannot be made realistic enough to honestly try: the
public metabolite library covers about a quarter of the signal in a real serum
CPMG spectrum.

## Why the generative route was not attempted

G2 established that transfer is limited by the number of independent studies, and
that this corpus holds an effective 4.09 of them. A VAE or diffusion model fitted
to the corpus can only resample the composition it was shown, so it cannot raise
that number -- it manufactures rows, which G2 showed do not help. The metabolite
basis was the only route with a mechanism for producing genuinely new
compositions, and that mechanism is capped at R² = 0.25.

This is a limit of the available basis, not a proof that no synthetic data could
work. A library covering the serum metabolome, or an experimentally measured
basis, would reopen the question. That is a recommendation, not a result.

## Position in the tree

All three gates are now answered:

- **G1** — classical ML wins, 24/24 cells, across all four transfer modes.
- **G2** — volume saturates below 2,000 spectra; diversity helps but this corpus
  cannot exceed an effective ~7 studies.
- **G3** — synthetic data cannot be validated, so it is not trained on.

The whiteboard's remaining branch after a G3 "no" is **more parameters** (Phase
6). The prior on that is low for the same reason G1 gave: a 1.7 M-parameter
encoder already overfits 40–142 samples, and fine-tuning made it worse, not
better. The honest terminal statement is the one the tree anticipated: *these
tasks have a ceiling at this data scale, and more public experimental data --
specifically more independent studies -- is what would move it.*
