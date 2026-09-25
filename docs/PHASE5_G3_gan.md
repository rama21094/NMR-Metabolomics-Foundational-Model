# Gate G3, generative route: a GAN gets far closer, and still fails the gate

## Result

WGAN-GP, 300 epochs on the 8,545-spectrum training corpus, 4,000 spectra
generated. Put through the **same** Phase 5.2 gate as the GISSMO set.

| test | GISSMO basis | **GAN** | real / target |
|---|---|---|---|
| discriminator AUC, real vs synthetic | 1.000 | **0.945** (±0.067) | 0.5 |
| median per-bin Wasserstein / IQR | 9.72 | **1.19** | 0.053 floor |
| bins below 0.25 | 0.0% | **11.1%** | — |
| bins within the 0.053 floor | 0.0% | 0.0% | — |
| lactate CH₃→CH separation | 2.781 | **2.788** | 2.788 real |
| effective studies | (8.73, invalid) | **4.71** | 4.09 corpus |

**The GAN solves the problem GISSMO had.** Landmarks match the real corpus
exactly, and the distributional distance is 8× smaller than the metabolite basis
managed. Learning the whole signal rather than a 25%-coverage basis was the right
fix for the realism failure.

**It still fails 5.2.** A linear discriminator on binned spectra still separates
real from generated at AUC 0.945, and no bin reaches the within-corpus floor.
Under the plan's rule it is not trained on.

## The prediction held

I predicted a generator fitted to this corpus would resample its composition
rather than add to it. Effective studies of the generated set is **4.71**,
against 4.09 for the corpus and 4.26 for the same metric applied to real spectra.
Its composition mirrors the corpus: MTBLS798 takes 51% of generated spectra
against 57% of real ones.

This measurement is valid here in a way it was not for GISSMO. The GISSMO set's
8.73 was high because those spectra resembled no study. The GAN set is close
enough to real (AUC 0.945, not 1.000; landmarks exact) that nearest-centroid
assignment means something.

So even if the GAN had passed 5.2, it would not have addressed what G2 found to be
binding. It adds rows. G2 showed rows saturate below 2,000.

## A caveat on training stability

Training was stable to about epoch 210, then destabilised: generator loss swings
between roughly −260 and +90 from epoch 240 onward. The generated set comes from
the epoch-299 state, and the checkpoint was overwritten every 5 epochs, so no
stable-phase generator was kept.

This cuts in an unclear direction. Critic loss kept *falling* through the unstable
phase (−0.91 → −0.32), which means the critic's estimate of the real–generated
distance went down, not up. In WGAN training the generator loss is on the
critic's arbitrary output scale, and wide swings there do not necessarily mean
worse samples. Still, a generator saved at epoch ~200 could plausibly score
better on 5.2. **The verdict is "this GAN failed", not "no GAN can pass".**

Even a GAN that passed 5.2 would face the diversity result, which is a property
of what it was trained on rather than of how well it trained.

## Verdict for G3

Both routes named on the whiteboard were tested:

- **Linear combinations of metabolite spectra:** chemistry right, realism wrong.
  The basis explains R² = 0.25 of a real spectrum.
- **Generative model:** realism much closer, still detectable. Resamples the
  corpus composition (effective 4.71 vs 4.09).

Neither produces data that passes validation, and the one with a mechanism to
raise effective diversity (the basis) is the one that cannot be made realistic.
**G3: no.**
