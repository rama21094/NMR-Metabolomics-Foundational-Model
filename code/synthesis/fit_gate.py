#!/usr/bin/env python3
"""Stage 0c: THE GATE. Does the LC forward model explain real serum spectra?

Everything downstream is conditional on this. If a GISSMO basis plus a smooth
macromolecule envelope plus a baseline can reproduce ~90% of a real spectrum's
variance, the forward model is adequate and synthetic spectra will resemble real
ones. If it reproduces 60%, components are missing, and synthetic data would be
systematically unlike the corpus -- while still looking perfectly plausible on
any plot, as the windowed-synthesis work already demonstrated.

THE MODEL
    y(x) = B(x) c_met  +  M(x) c_mm  +  P(x) c_base
  B      43 GISSMO metabolite basis vectors at 600 MHz          c_met >= 0
  M      broad Gaussian bumps tiled across the signal region,   c_mm  >= 0
         standing in for the lipoprotein / protein envelope that
         survives CPMG and is absent from any metabolite library
  P      low-order Legendre polynomials for baseline roll       c_base free

Non-negativity is imposed on the metabolite and macromolecule blocks because
concentrations and envelopes cannot be negative; the polynomial block is left
free because unit-area normalised spectra do contain small negative excursions.

WHY THE FIT IS DONE AT REDUCED RESOLUTION. 131,072 points is far more than the
linewidth requires: binning 8:1 gives 16,384 points, i.e. 0.44 Hz per point at
600 MHz, against natural linewidths of 1-2 Hz. Nothing that distinguishes
metabolites is lost, and the bounded least-squares solve becomes tractable.

WHAT IS REPORTED
  R2 for metabolites alone, envelope alone, and the full model -- so it is clear
  whether the metabolite basis is doing the work or the smooth envelope is
  absorbing everything. A high full-model R2 driven entirely by the envelope
  would be a false pass.
  Per-ppm residual, to show WHERE the model fails, which names the missing
  components.
  Per-metabolite usage, which is the evidence for pruning the panel.

STATUS 2026-09-02 -- THIS GATE HAS NOT BEEN PASSED, AND IS NOT YET A VERDICT.
-----------------------------------------------------------------------------
The trustworthy figure from this script is the UNCONSTRAINED UPPER BOUND, which
uses a plain `lstsq` and is therefore solver-independent: median R2 ~= 0.45-0.47.
That is well below the 0.90 bar, so the forward model as currently specified
cannot reproduce real serum spectra.

The constrained numbers are NOT yet trustworthy and should not be quoted. The
metabolite and envelope blocks are partially degenerate (a broadened multiplet
can be cancelled by a smooth bump), and the bounded solver has been observed to
return R2 = -2.7e7 (unbounded polynomials), R2 = -22 (bounded polynomials), and
R2 = 0.11 with a metabolite signal fraction of 0.004 (ridge = 1e-3, i.e.
over-shrunk). A per-block ridge, or an explicit orthogonalisation of the
envelope against the metabolite block, is the next fix.

THREE CAUSES OF THE LOW UPPER BOUND, in order of expected impact. Note that the
first two are limitations of THIS FIT, not of GISSMO or of the LC route:

1. NO PER-SPECTRUM OR PER-METABOLITE SHIFT FREEDOM. This corpus has no usable
   0 ppm reference resonance (the 0 ppm window holds a median 0.7% of each
   spectrum's global maximum), and it was aligned to its own rightmost peak, so
   absolute ppm is arbitrary and varies between the pooled source studies. A
   single global offset cannot align it -- the search here lands anywhere from
   -0.29 to +0.15 ppm depending on the probe set, which is itself the symptom.
   Every serious NMR quantification tool (BATMAN, rDolphin, Chenomx) fits a
   per-peak shift for exactly this reason. Implementing that is the single
   highest-value next step.

2. MISSING COMPONENTS. Residuals concentrate near 1.0, 1.6-1.7, 2.1-2.5, 2.9 and
   3.2-3.6 ppm. The 0.8-1.3 region is the lipoprotein CH2/CH3 envelope and the
   panel contains NO lipid components at all; 3.2-3.6 is the crowded sugar/
   choline region. 43 metabolites is also well short of the ~100 detectable in
   serum, and acetone, urea and mannose are absent from GISSMO entirely.

3. LINEWIDTH. GISSMO simulates near-ideal narrow lines. 2-4 Hz of Lorentzian
   broadening is physical and improves the fit; gains beyond ~8 Hz are the basis
   degenerating into smooth blobs that duplicate the envelope block, and must
   not be chased.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from numpy.polynomial import legendre
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[2]
_cn = importlib.util.spec_from_file_location("cn", ROOT / "code/analysis/corpus_normalisation.py")
cn = importlib.util.module_from_spec(_cn)
_cn.loader.exec_module(cn)

CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "WaterFixed_EDTAMagnitude_v4.npy")
PPM_AXIS = "data/mtbls326/MTBLS326_common_ppm_axis.npy"
WATER_LO, WATER_HI = 62500, 68000
SIGNAL_LO, SIGNAL_HI = 0.5, 9.5


def bin_rows(A, fold):
    n = (A.shape[-1] // fold) * fold
    return A[..., :n].reshape(*A.shape[:-1], n // fold, fold).mean(axis=-1)


def unconstrained_r2(A, y):
    """Upper bound on what a design can explain, ignoring sign constraints.

    Reported alongside the constrained fit so a low R2 cannot be blamed on the
    non-negativity constraints when the design itself is the limit.
    """
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ sol
    return 1.0 - float((res ** 2).sum() / ((y - y.mean()) ** 2).sum())


def macromolecule_basis(ppm_b, width_ppm, spacing_ppm):
    centres = np.arange(SIGNAL_LO, SIGNAL_HI + 1e-9, spacing_ppm)
    M = np.exp(-0.5 * ((ppm_b[None, :] - centres[:, None]) / width_ppm) ** 2)
    M /= np.abs(M).sum(axis=1, keepdims=True)
    return M, centres


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-spectra", type=int, default=60)
    ap.add_argument("--bin-fold", type=int, default=8)
    ap.add_argument("--mm-width-ppm", type=float, default=0.25)
    ap.add_argument("--mm-spacing-ppm", type=float, default=0.15)
    ap.add_argument("--poly-order", type=int, default=3,
                    help="high orders on a masked domain oscillate and destabilise the fit")
    ap.add_argument("--broadening-hz", type=float, default=2.0,
                    help="Lorentzian FWHM added to the GISSMO basis. GISSMO simulates "
                         "near-ideal narrow lines; real serum CPMG peaks are broader. "
                         "Values above ~8 Hz degrade the basis into smooth blobs that "
                         "merely duplicate the envelope block, so R2 gains there are "
                         "not metabolite identification -- do not chase them.")
    ap.add_argument("--shift-search-ppm", type=float, default=0.45,
                    help="search a GLOBAL ppm offset over +/- this range. Needed because "
                         "this corpus was aligned to its own rightmost peak, not to a 0 ppm "
                         "reference (there is no usable reference resonance in it), so its "
                         "ppm labels carry an arbitrary absolute offset.")
    ap.add_argument("--shift-steps", type=int, default=25)
    ap.add_argument("--envelope", choices=["bumps", "lipid", "both"], default="lipid",
                    help="'bumps' = generic Gaussians tiled across the spectrum (the "
                         "original, heavily degenerate with the metabolite block); "
                         "'lipid' = the empirical envelope basis from lipid_basis.py, "
                         "which is derived from what the metabolite basis CANNOT explain "
                         "and is therefore far less collinear with it "
                         "(mean |cos| 0.07 vs the bumps).")
    ap.add_argument("--ridge", type=float, default=1e-6,
                    help="Tikhonov penalty, as a fraction of the design's largest "
                         "singular value. The metabolite and envelope blocks are "
                         "partially degenerate -- a broadened multiplet and a smooth "
                         "bump can cancel -- and without a penalty the bounded solver "
                         "returns huge opposing amplitudes (observed R2 = -22 with "
                         "bounds, -2.7e7 without). A small ridge removes that null "
                         "space without materially biasing the fit.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/synthesis/fit_gate")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm = np.load(ROOT / PPM_AXIS)
    B = np.load(ROOT / "results/synthesis/basis_600MHz.npy").astype(np.float64)
    meta = pd.read_csv(ROOT / "results/synthesis/basis_600MHz_meta.csv")

    corpus = cn.open_corpus(ROOT / CORPUS, "unit_area", verbose=True)
    n_total, length = corpus.shape
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n_total, args.n_spectra, replace=False))
    Y = np.asarray(corpus[rows], dtype=np.float64)

    fold = args.bin_fold
    ppm_b = bin_rows(ppm, fold)
    Yb = bin_rows(Y, fold)
    Bb = bin_rows(B, fold)
    hz_per_pt = (ppm.max() - ppm.min()) * 600.0 / len(ppm_b)
    print(f"fitting at {len(ppm_b)} points ({hz_per_pt:.2f} Hz/point at 600 MHz)")

    water = np.zeros(len(ppm), bool)
    water[WATER_LO:WATER_HI] = True
    fitmask = (~bin_rows(water.astype(float), fold).astype(bool)) & \
              (ppm_b >= SIGNAL_LO) & (ppm_b <= SIGNAL_HI)
    print(f"fit region: {int(fitmask.sum())} points "
          f"({SIGNAL_LO}-{SIGNAL_HI} ppm, water window excluded)")

    blocks = []
    if args.envelope in ("bumps", "both"):
        Mb, _ = macromolecule_basis(ppm_b, args.mm_width_ppm, args.mm_spacing_ppm)
        blocks.append(Mb)
    if args.envelope in ("lipid", "both"):
        lp = ROOT / "results/synthesis/lipid_basis.npy"
        if not lp.exists():
            raise SystemExit(f"{lp} missing -- run code/synthesis/lipid_basis.py first")
        blocks.append(bin_rows(np.load(lp).astype(np.float64), fold))
    M = np.vstack(blocks)
    print(f"envelope block: {args.envelope} -> {len(M)} components")
    xs = np.linspace(-1, 1, len(ppm_b))
    P = np.stack([legendre.Legendre.basis(k)(xs) for k in range(args.poly_order + 1)])
    nB, nM, nP = len(Bb), len(M), len(P)
    print(f"design: {nB} metabolites + {nM} envelope bumps + {nP} polynomials "
          f"= {nB+nM+nP} columns\n")

    # ---- broaden the basis, then find the best GLOBAL ppm offset ----
    if args.broadening_hz > 0:
        pts_per_ppm = 1.0 / abs(ppm_b[1] - ppm_b[0])
        hw = max(args.broadening_hz / 600.0 * pts_per_ppm / 2.0, 1e-6)
        n = int(max(32, min(1500, 12 * hw)))
        xk = np.arange(-n, n + 1)
        k = 1.0 / (1.0 + (xk / hw) ** 2)
        k /= k.sum()
        Bb = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, Bb)
        print(f"basis broadened by {args.broadening_hz} Hz Lorentzian "
              f"({args.broadening_hz/600.0:.4f} ppm FWHM)")

    dppm = abs(ppm_b[1] - ppm_b[0])
    max_bins = int(args.shift_search_ppm / dppm)
    cand = np.linspace(-max_bins, max_bins, args.shift_steps).astype(int)
    probe = Yb[: min(6, len(Yb))]
    best_shift, best_val = 0, -np.inf
    for sb in cand:
        A = np.vstack([np.roll(Bb, sb, axis=1), M, P])[:, fitmask].T
        v = float(np.mean([unconstrained_r2(A, y[fitmask]) for y in probe]))
        if v > best_val:
            best_shift, best_val = int(sb), v
    print(f"best global ppm offset: {best_shift:+d} bins "
          f"({-best_shift*dppm:+.3f} ppm), unconstrained R2 {best_val:.4f} on "
          f"{len(probe)} probe spectra")
    Bb = np.roll(Bb, best_shift, axis=1)

    A_full = np.vstack([Bb, M, P])[:, fitmask].T
    yscale = float(np.abs(Yb[:, fitmask]).max())
    plim = 10.0 * yscale
    lo = np.concatenate([np.zeros(nB + nM), np.full(nP, -plim)])
    hi = np.concatenate([np.full(nB + nM, np.inf), np.full(nP, plim)])

    # The baseline block must be BOUNDED. With +/-inf bounds, trf on a masked
    # domain lets high-order polynomials grow without limit and trade off against
    # the metabolite block; an earlier version of this script returned solutions
    # with R2 = -2.7e7 and a metabolite "signal fraction" of 3.09, i.e. huge
    # cancelling amplitudes. Bounding the polynomials to a multiple of the data
    # scale fixes it.
    def fit(A, lb, ub, y):
        """Bounded least squares with a ridge penalty, returning residuals on the
        ORIGINAL rows (not the augmented ones) so R2 stays interpretable."""
        if args.ridge > 0:
            lam = args.ridge * np.linalg.norm(A, 2)
            Aa = np.vstack([A, lam * np.eye(A.shape[1])])
            ya = np.concatenate([y, np.zeros(A.shape[1])])
        else:
            Aa, ya = A, y
        r = lsq_linear(Aa, ya, bounds=(lb, ub), method="trf", tol=1e-10, max_iter=500)
        return r.x, y - A @ r.x


    def r2(y, resid):
        return 1.0 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum())

    recs, coefs, resid_stack = [], [], []
    A_met = np.vstack([Bb, P])[:, fitmask].T
    lo_met = np.concatenate([np.zeros(nB), np.full(nP, -plim)])
    hi_met = np.concatenate([np.full(nB, np.inf), np.full(nP, plim)])
    A_mm = np.vstack([M, P])[:, fitmask].T
    lo_mm = np.concatenate([np.zeros(nM), np.full(nP, -plim)])
    hi_mm = np.concatenate([np.full(nM, np.inf), np.full(nP, plim)])
    unc = []

    for i, y in enumerate(Yb):
        yf = y[fitmask]
        x_all, res_all = fit(A_full, lo, hi, yf)
        x_met, res_met = fit(A_met, lo_met, hi_met, yf)
        x_mm, res_mm = fit(A_mm, lo_mm, hi_mm, yf)
        unc.append(unconstrained_r2(A_full, yf))
        recs.append(dict(row=int(rows[i]),
                         r2_full=r2(yf, res_all), r2_metabolites_only=r2(yf, res_met),
                         r2_envelope_only=r2(yf, res_mm),
                         n_metabolites_used=int((x_all[:nB] > 1e-12).sum()),
                         r2_unconstrained_upper_bound=unc[-1],
                         metabolite_signal_frac=float(
                             np.abs(Bb[:, fitmask].T @ x_all[:nB]).sum() /
                             (np.abs(yf).sum() + 1e-30))))
        coefs.append(x_all[:nB])
        r = np.zeros(len(ppm_b)); r[fitmask] = res_all
        resid_stack.append(r)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1}/{len(Yb)}  R2 full={recs[-1]['r2_full']:.3f} "
                  f"met-only={recs[-1]['r2_metabolites_only']:.3f} "
                  f"env-only={recs[-1]['r2_envelope_only']:.3f}")

    df = pd.DataFrame(recs)
    C = np.stack(coefs)
    df.to_csv(out_dir / "fit_per_spectrum.csv", index=False)
    pd.DataFrame(C, columns=meta.name).assign(row=rows).to_csv(
        out_dir / "fitted_coefficients.csv", index=False)
    R = np.stack(resid_stack)
    np.save(out_dir / "residual_mean_abs.npy", np.abs(R).mean(0))
    np.save(out_dir / "ppm_binned.npy", ppm_b)

    print("\n" + "=" * 78)
    print("  THE GATE")
    print("=" * 78)
    for k, lab in [("r2_full", "full model (metabolites + envelope + baseline)"),
                   ("r2_metabolites_only", "metabolites + baseline only"),
                   ("r2_envelope_only", "envelope + baseline only")]:
        print(f"  R2 {lab:48s} median {df[k].median():.3f}  "
              f"IQR {df[k].quantile(.25):.3f}-{df[k].quantile(.75):.3f}")
    print(f"  R2 {'unconstrained upper bound (sign constraints off)':48s} median "
          f"{df.r2_unconstrained_upper_bound.median():.3f}")
    print(f"\n  metabolites with a non-zero coefficient: median "
          f"{df.n_metabolites_used.median():.0f}/{nB}")
    print(f"  fraction of |signal| explained by metabolite block: median "
          f"{df.metabolite_signal_frac.median():.3f}")

    used = (C > 1e-12).mean(0)
    share = np.abs(C).mean(0) / (np.abs(C).mean(0).sum() + 1e-30)
    tab = pd.DataFrame(dict(name=meta.name, frac_spectra_nonzero=used,
                            mean_abs_coef_share=share)).sort_values(
                                "mean_abs_coef_share", ascending=False)
    tab.to_csv(out_dir / "metabolite_usage.csv", index=False)
    print(f"\n  top 12 metabolites by fitted share:")
    for _, r in tab.head(12).iterrows():
        print(f"    {r['name']:20s} used in {100*r.frac_spectra_nonzero:5.1f}% of spectra, "
              f"share {100*r.mean_abs_coef_share:5.1f}%")
    print(f"\n  never used (candidates to prune): "
          f"{', '.join(tab[tab.frac_spectra_nonzero == 0]['name']) or 'none'}")

    med = df.r2_full.median()
    verdict = ("PASS" if med >= 0.90 else "MARGINAL" if med >= 0.75 else "FAIL")
    print(f"\n  VERDICT: median full-model R2 = {med:.3f}  ->  {verdict}")
    if verdict != "PASS":
        ra = np.abs(R).mean(0)
        top = np.argsort(-ra)[:400]
        pk = np.unique(np.round(ppm_b[top], 1))
        print(f"  Largest residuals concentrate near ppm: "
              f"{', '.join(f'{p:.1f}' for p in pk[:12])}")
        print(f"  Those regions name the missing components.")
    print(f"\nWrote {out_dir}/")


if __name__ == "__main__":
    main()
