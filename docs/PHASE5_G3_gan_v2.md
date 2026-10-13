# GAN v2 results, and what distinguishes GAN spectra from real ones

Follow-up agreed at the 2026-09-25 PI review: (1) finish the GAN v2 runs,
(2) use PCA or a statistical test rather than a neural classifier to find what
gives GAN output away.

## 1. GAN v2 results

| | GAN v1 | **GAN v2** | GAN v2 study-mixing | target |
|---|---|---|---|---|
| separability AUC (5.2 gate) | 0.945 | **0.832** | 0.922 | 0.5 |
| distance from real (Wasserstein / IQR) | 1.19 | **1.03** | 1.15 | 0.05 |
| lactate CH₃→CH (real 2.788) | 2.788 | **2.787** | 2.795 | |
| effective studies (corpus 4.09) | 4.71 | **4.38** | **8.38** | |
| trained to | epoch 299 (unstable) | best = epoch 249 | diverged at 175; best = epoch 79 | |

GAN v2 (gradient penalty, tanh output, EMA generator, best-checkpoint selection)
is the most realistic set so far, but still fails the gate. The study-mixing arm
never trained well and the divergence guard stopped it; its spectra come from the
best checkpoint it reached.

## 2. What gives GAN v2 away

`code/analysis/gan_vs_real_features.py`, `code/analysis/gan_noise_diagnostic.py`,
figure `docs/figures/fig_gan_vs_real.png`. Every test was also run on two random
halves of the real data, to calibrate how large a difference means anything when
there are thousands of spectra per side.

**The spectrum edges, overwhelmingly.** Separability with windows removed (same
linear test as the gate, kept windows re-normalised):

| windows | GAN v2 | study-mixing |
|---|---|---|
| all 468 | 0.776 | 0.923 |
| edges removed (0.2–9.4 ppm) | **0.589** | 0.883 |
| edges only (40 windows) | **0.940** | 0.852 |

In the 0.2–9.4 ppm region where the metabolites are, GAN v2 is close to
indistinguishable from real. Real spectra are flat at both ends; GAN output
wanders there:

| region | slow baseline wander, GAN / real | point-scale noise, GAN / real |
|---|---|---|
| −0.5 to −0.2 ppm | 11.7× | 23.3× |
| 9.5 to 10 ppm | **75.9×** | 16.4× |
| 8.2 to 9.0 ppm (empty aromatic) | 1.9× | 0.4× |
| TSP, −0.1 to +0.1 ppm | 1.0× | 1.6× |

**PCA (fitted on real spectra only).** GAN spectra sit inside the real cloud on
the main components -- 3.5% fall outside the real 95% envelope on the top 10
PCs, against 4.9% for held-out real spectra. But 81% of a GAN spectrum's
variation lies in directions the 20 real PCs do not capture, against 12% for real
spectra. The GAN reproduces the real variation well and adds variation real
spectra do not have -- consistent with the edge wander.

**Things that are NOT the giveaway:**
- Peak positions (landmarks within 0.001 ppm) and the creatine linewidth (Cliff's
  delta −0.14, small).
- TSP. Its peak height matches (0.0687 vs 0.0686 of spectrum max) and it is present
  in 99% of both sets. An early reading of the windows beside 0 ppm as "missing TSP"
  was wrong.

**A diagnosis that was tested and withdrawn.** The first feature run showed the
noise level in signal-free regions 3.6× higher for GAN output, and I read that as
excess random noise. Smoothing the GAN output to match real noise did not lower the
noise (it rose slightly) and did not move the separability (0.832 → 0.833). It
couldn't have: 0.02 ppm windows already average random noise away. The excess is
slow baseline structure at the edges, not noise.

**Reading the per-window plot (panel a).** Most windows show the GAN slightly
*lower* than real. That is largely a side effect of normalisation: each spectrum is
scaled to equal total area, so excess at the edges pushes every other window down.
It is why removing the edges and re-normalising drops separability to 0.59.

**Remaining chemical-region differences** are moderate (|delta| 0.4–0.5): the GAN is
higher near 1.83 and 2.63 ppm and lower at 3.33–3.35 ppm.

**Likely mechanism (a hypothesis, not tested).** Convolutions with zero padding
are a well-known source of artefacts at the borders of generated signals, and the
spike at the 10.0 ppm boundary in panel (d) is consistent with that.

## 3. The study-mixing arm's 8.38

This is the first synthetic set whose effective study count exceeds the corpus's,
which is what the conditional design aimed for. Two caveats keep it from being a
result yet:
- Its realism is poor throughout (0.88 even without the edges), and the diversity
  metric is only trustworthy for sets that look real.
- Uniform study sampling forces equal weight onto studies the GAN saw only 1–76
  spectra of, so part of that spread may be memorised copies of a few spectra.

## 4. Options, for discussion

1. **Blank the edges in preprocessing**, as already done for water. That is
   legitimate, since the edges carry no metabolite signal. But "edges" up to
   0.2 ppm includes **TSP at 0 ppm**, which the PI wants for concentration work, so
   the blanked range would be −0.5 to −0.1 and 9.4 to 10 ppm.
2. **Fix the generator's borders** (reflect/replicate padding) and retrain. About
   12 GPU-hours.
3. **Then the real G3 test:** pretrain on real + GAN v2 and measure transfer. The
   prior is still low -- GAN v2 resamples the corpus (effective 4.38) -- but with
   the edge problem understood it is now testable.
4. **Stabilise the study-mixing arm**, the only route with a mechanism for raising
   effective diversity.
