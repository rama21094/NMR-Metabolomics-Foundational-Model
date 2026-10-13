# G3: does pretraining on real + GAN v2 spectra improve transfer?

**No. Adding 4,000 GAN v2 spectra made transfer slightly worse.**

Design: masking objective, baseline architecture, 3 seeds per arm, edges blanked
(ppm < -0.2 and > 9.4) in pretraining and cohort spectra for both arms.
Control = blanked real corpus; test = blanked real corpus + 4,000 GAN v2.
Evaluation = phase3_fixed_eval protocol (same splits), 24 cohort x budget cells.
Script: code/evaluation/g3_augment_eval.py; results: results/g3_augment/g3_augment_eval.json.

| Comparison (24 cells) | better | worse | level | mean diff |
|---|---|---|---|---|
| real+GAN vs real only | 2 | 14 | 8 | -0.011 |
| real+GAN vs standard ML | 0 | 22 | 2 | -0.078 |
| real only (blanked) vs standard ML | 0 | 22 | 2 | -0.067 |

The two cells where GAN helped are both small-sample: Barth k=5 (+0.033) and
MTBLS563 binary k=5. Barth LOOCV is +0.079 but the interval spans zero.

Confound: GAN-arm runs trained ~2x longer (best epochs 326-343 vs 95-221),
because synthetic spectra in the early-stopping set keep val loss falling. This
confound favours the GAN arm, so it does not weaken the "no".

Edge blanking itself did not change transfer meaningfully: blanked real-only is
level with the unblanked Phase 2 baseline on average (-0.067 vs about -0.06 against standard ML).
