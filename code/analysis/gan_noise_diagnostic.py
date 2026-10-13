#!/usr/bin/env python3
"""Is excess noise what gives the GAN away? A direct test of that diagnosis.

gan_vs_real_features.py found the largest real-vs-GAN differences are not in
peaks but in noise: signal-free noise 3.6x higher and high-frequency texture 1.9x
higher in GAN output (Cliff's delta +0.68 and +0.67), while peak positions and
widths match. If that diagnosis is right, smoothing the GAN output until its
noise matches real spectra should sharply lower the 5.2 discriminator AUC. If
AUC barely moves, noise was a symptom and the giveaway lies elsewhere.

This is a DIAGNOSTIC, not a fix to adopt silently: smoothing to pass a test would
be tuning to the test. It answers "is noise the main tell?", which decides
where the generator needs work.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
q = (ppm > 9.5) | (ppm < -0.3)
X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
S = np.load(ROOT / "rebuild/pretrain/synthetic/gan_v2.npy", mmap_mode="r")
rng = np.random.default_rng(0)
R = np.asarray(X[rng.choice(X.shape[0], 200, replace=False)], dtype=np.float64)
noise = lambda M: np.median([y[q].std() / np.abs(y).max() for y in M])
target = noise(R)
probe = np.asarray(S[:200], dtype=np.float64)
print(f"real noise {target:.3e}, GAN noise {noise(probe):.3e}")
best = None
for sigma in (1, 2, 3, 4, 6, 8, 12, 16):
    n = noise(gaussian_filter1d(probe, sigma, axis=1))
    print(f"  sigma {sigma:>2} points ({sigma*abs(ppm[1]-ppm[0])*600:.2f} Hz)  noise {n:.3e}  ratio {n/target:.2f}")
    if best is None or abs(np.log(n / target)) < abs(np.log(best[1] / target)):
        best = (sigma, n)
sigma = best[0]
print(f"chosen sigma {sigma} points")
out = np.empty(S.shape, dtype=np.float32)
for i in range(0, S.shape[0], 250):
    out[i:i+250] = gaussian_filter1d(np.asarray(S[i:i+250], dtype=np.float32), sigma, axis=1)
np.save(ROOT / "rebuild/pretrain/synthetic/gan_v2_smoothed.npy", out)
print("wrote rebuild/pretrain/synthetic/gan_v2_smoothed.npy")
