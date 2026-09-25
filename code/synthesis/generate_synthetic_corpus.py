#!/usr/bin/env python3
"""Phase 5.1/5.2 -- synthesise spectra as linear combinations of GISSMO metabolites.

WHY THIS ROUTE AND NOT A GENERATIVE MODEL. G2 found that transfer is limited by
the number of independent studies, not the number of spectra, and that this
corpus holds an effective 4.09 studies. A VAE or diffusion model fitted to the
corpus can only resample the composition it was shown, so it cannot raise that
number -- it would manufacture rows, which G2 showed do not help. A metabolite
basis can, in principle, do something a generator cannot: produce concentration
combinations that appear in NO real study, because the mixing coefficients are
sampled rather than copied.

Whether that is enough is the G3 question, and this script is built so the answer
can be "no".

HOW CONCENTRATIONS ARE SAMPLED. Non-negative least squares fits the basis to real
corpus spectra, giving an empirical coefficient matrix. New spectra are then
drawn by sampling each metabolite's coefficient INDEPENDENTLY from its own
empirical marginal (--mode marginal). Independence is the point: it breaks the
between-metabolite correlations that define a particular study's physiology, so
the synthetic set covers combinations the corpus never contained. --mode joint
resamples whole real coefficient vectors instead, which preserves those
correlations and is the circular control -- if joint helps as much as marginal,
nothing was gained by the extra coverage.

REALISM CONTROLS. Each spectrum gets a random global referencing offset drawn to
match the between-study spread measured in the alignment QC (sd 0.05 ppm), random
per-metabolite line broadening, and additive noise at a signal-to-noise ratio
sampled from the real corpus. Without these the discriminator in 5.2 separates
real from synthetic trivially on linewidth alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
WATER = (4.45, 5.10)


def pool(A, f):
    """Mean-pool the last axis by a factor f (drops the ragged tail)."""
    n = (A.shape[-1] // f) * f
    return A[..., :n].reshape(*A.shape[:-1], n // f, f).mean(axis=-1)


def fit_coefficients(X, B, n_fit, rng, factor=32):
    """NNLS fit of the basis to a sample of real spectra.

    Fitted on a mean-pooled axis (131,072 -> ~4,096 points). NNLS costs
    O(n_points * n_basis^2) per iteration, so fitting at full resolution takes
    minutes PER SPECTRUM and gains nothing: the coefficients are concentrations,
    and pooling is a linear operator applied identically to both sides, so the
    least-squares solution is essentially unchanged while the fit is ~32x cheaper.
    """
    idx = rng.choice(X.shape[0], size=min(n_fit, X.shape[0]), replace=False)
    C = np.zeros((len(idx), B.shape[0]), dtype=np.float32)
    Bt = pool(B, factor).T
    for j, i in enumerate(sorted(idx)):
        y = pool(np.asarray(X[i], dtype=np.float64), factor)
        C[j], _ = nnls(Bt, y)
        if (j + 1) % 25 == 0:
            print(f"\r  nnls {j+1}/{len(idx)}", end="", flush=True)
    print()
    return C


def shift(y, d_ppm, ppm):
    """Shift a spectrum by d_ppm, by interpolation on the ppm axis.

    np.interp REQUIRES an increasing xp and does not check: on this project's
    descending axis (10.0 -> -0.5) it silently returned all zeros, even for a
    zero shift. Every synthetic spectrum generated before this fix was therefore
    pure added noise rescaled to full height, and the GISSMO 5.2 results computed
    from them (AUC 1.000, Wasserstein 9.72) measured noise, not metabolite
    mixtures. Interpolate on the reversed, ascending axis instead.
    """
    if ppm[0] > ppm[-1]:
        return np.interp(ppm[::-1], (ppm + d_ppm)[::-1], y[::-1],
                         left=0.0, right=0.0)[::-1]
    return np.interp(ppm, ppm + d_ppm, y, left=0.0, right=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="results/synthesis/rebuilt/basis_600MHz.npy")
    ap.add_argument("--lipid-basis", default="results/synthesis/rebuilt/lipid_basis.npy",
                    help="broad lipid/macromolecule envelope components. Real CPMG "
                         "serum carries a large broad background; without it the "
                         "5.2 discriminator separates real from synthetic trivially "
                         "on the baseline alone. 'none' to omit.")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-fit", type=int, default=150)
    ap.add_argument("--mode", choices=["marginal", "joint"], default="marginal")
    ap.add_argument("--shift-sd", type=float, default=0.05,
                    help="between-study referencing spread, ppm (alignment QC)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rebuild/pretrain/synthetic/synth_marginal.npy")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    B = np.load(ROOT / args.basis).astype(np.float64)
    if args.lipid_basis != "none":
        L = np.load(ROOT / args.lipid_basis).astype(np.float64)
        B = np.vstack([B, L])
        print(f"basis augmented with {L.shape[0]} lipid/envelope components")
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    print(f"basis {B.shape}, corpus {X.shape}, mode {args.mode}")

    C = fit_coefficients(X, B, args.n_fit, rng)
    nz = (C > 0).mean()
    print(f"fitted {C.shape[0]} spectra, {nz:.1%} of coefficients non-zero")

    # noise level of the real corpus, from a signal-free region
    sig = (ppm > 9.5) | (ppm < -0.3)
    real = np.asarray(X[rng.choice(X.shape[0], 100, replace=False)], dtype=np.float32)
    noise_sd = np.median(real[:, sig].std(axis=1))
    amp = np.median(np.abs(real).max(axis=1))
    print(f"real noise sd {noise_sd:.3e}, typical amplitude {amp:.3e}")

    out = np.zeros((args.n, B.shape[1]), dtype=np.float32)
    for i in range(args.n):
        if args.mode == "marginal":
            c = np.array([rng.choice(C[:, m]) for m in range(C.shape[1])])
        else:
            c = C[rng.integers(len(C))].astype(np.float64)
        y = c @ B
        y = shift(y, rng.normal(0.0, args.shift_sd), ppm)
        y = y + rng.normal(0.0, noise_sd, size=y.shape)
        y[(ppm <= WATER[1]) & (ppm >= WATER[0])] = 0.0
        m = np.abs(y).max()
        out[i] = (y / m * amp) if m > 0 else y
        if (i + 1) % 250 == 0:
            print(f"\r  generated {i+1}/{args.n}", end="", flush=True)
    print()

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, out)
    json.dump({"mode": args.mode, "n": args.n, "n_fit": args.n_fit,
               "basis": args.basis, "shift_sd": args.shift_sd,
               "frac_nonzero_coef": float(nz), "seed": args.seed},
              open(p.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {args.out}  {out.shape}")


if __name__ == "__main__":
    main()
