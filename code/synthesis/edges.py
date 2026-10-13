"""Spectrum-edge blanking, applied identically to real, synthetic and cohort spectra.

The GAN's main giveaway is slow baseline wander at the two ends of the axis
(results/phase5/gan_auc_by_region.json). Real spectra carry no chemistry there,
so the ends are zeroed the same way the water region is excluded.

The upfield limit is -0.2, not 0.2: the TSP reference at 0 ppm has tails out to
about +/-0.1 ppm (the real median is still 3x baseline at -0.2..-0.1), and TSP is
kept because it may later anchor absolute concentrations.
"""
import numpy as np

BLANK_BELOW = -0.2   # ppm
BLANK_ABOVE = 9.4    # ppm


def edge_mask(ppm):
    ppm = np.asarray(ppm)
    return (ppm < BLANK_BELOW) | (ppm > BLANK_ABOVE)


def blank(M, ppm):
    """Return a float32 copy of M with the edge columns set to zero."""
    out = np.array(M, dtype=np.float32, copy=True)
    out[:, edge_mask(ppm)] = 0.0
    return out
