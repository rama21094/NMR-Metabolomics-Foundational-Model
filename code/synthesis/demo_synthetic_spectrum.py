#!/usr/bin/env python3
"""Illustrative synthetic serum spectrum from the GISSMO basis — a DEMO, not a product.

This draws metabolite concentrations from literature serum reference ranges, mixes
the 43-metabolite GISSMO basis (code/synthesis/build_basis.py) with the empirical
lipid/macromolecule components (code/synthesis/lipid_basis.py), adds a smooth
baseline and measured-level noise, and plots the result beside a real corpus
spectrum.

READ THIS BEFORE SHOWING IT TO ANYONE
-------------------------------------
These spectra are NOT validated synthetic training data. The acceptance test in
docs/PI_outline.md 8a requires the forward model to explain >=0.90 of the variance
of a real spectrum; fit_gate.py currently reaches 0.45-0.51. Section 5h explains
why: the corpus is not on a common chemical-shift axis, so a ppm-referenced basis
cannot match it. Until that is fixed we cannot say a synthetic spectrum drawn this
way resembles real data in the ways that matter.

What the figure DOES legitimately show:
  - the basis is real, assembled, and on the corpus grid;
  - a linear combination of it looks like a 1D serum NMR spectrum;
  - which peaks each metabolite contributes, and where the lipid envelope sits.

What it does NOT show:
  - that the concentrations are jointly realistic (they are sampled
    independently here, which is exactly the dependence problem of 5b/5c);
  - that a model pretrained on these would transfer.

Usage:
    python code/synthesis/demo_synthetic_spectrum.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "results/synthesis"
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_"
          "Water_EDTA_Suppressed.npy")

NPT = 131_072
PPM_HI, PPM_LO = 14.82, -5.20          # MTBLS798 window, the corpus majority axis
WATER = (62500, 68000)
NOISE_SD = 0.045                       # measured corpus noise floor (SSL doc 15)

# Typical fasting serum concentrations, mmol/L, midpoint of commonly cited ranges.
# Rough by design: this is an illustration of shape, not a quantitative claim.
CONC_MM = {
    "glucose": 5.0, "L-lactate": 1.5, "L-alanine": 0.35, "L-valine": 0.22,
    "L-leucine": 0.13, "L-isoleucine": 0.06, "L-glutamine": 0.55,
    "L-glutamic-acid": 0.05, "glycine": 0.24, "L-threonine": 0.13,
    "L-lysine": 0.18, "L-histidine": 0.08, "L-phenylalanine": 0.06,
    "L-tyrosine": 0.06, "L-methionine": 0.03, "L-proline": 0.20,
    "L-serine": 0.12, "L-arginine": 0.08, "L-asparagine": 0.05,
    "L-tryptophan": 0.05, "citrate": 0.10, "pyruvate": 0.08,
    "creatine": 0.05, "creatinine": 0.08, "acetate": 0.05,
    "3-hydroxybutyrate": 0.10, "acetoacetate": 0.04, "formate": 0.03,
    "choline": 0.01, "betaine": 0.05, "myo-inositol": 0.03,
    "taurine": 0.06, "ornithine": 0.06, "glycerol": 0.10,
}
DEFAULT_MM = 0.03                      # for basis entries with no range above
CV = 0.30                              # between-subject coefficient of variation


def ppm_axis() -> np.ndarray:
    return np.linspace(PPM_HI, PPM_LO, NPT)


def synthesise(B, names, L, rng, lipid_scale=1.0):
    """One synthetic spectrum: metabolites + lipid envelope + baseline + noise."""
    conc = np.array([CONC_MM.get(n, DEFAULT_MM) for n in names])
    # Log-normal draw keeps concentrations positive with a realistic right tail.
    sigma = np.sqrt(np.log(1.0 + CV ** 2))
    draw = conc * rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=len(conc))

    y = draw @ np.asarray(B, dtype=np.float64)

    # Lipid / macromolecule envelope. Kept SMALL on purpose: the corpus is 99.7%
    # CPMG (docs/PI_outline.md 5h), and a CPMG filter suppresses exactly this broad
    # component. An envelope at tens of percent of the metabolite maximum -- which
    # is what my first draft of this script produced -- is a plain 1D serum
    # spectrum, not the CPMG spectra we actually have.
    w = np.abs(rng.normal(0, 1, size=min(3, L.shape[0])))
    env = np.abs(w @ np.asarray(L[: len(w)], dtype=np.float64))
    if env.max() > 0:
        env *= lipid_scale * 0.06 * y.max() / env.max()
    y = y + env

    # Smooth broad baseline, of the kind phase/baseline correction leaves behind.
    t = np.linspace(0, 1, NPT)
    base = sum(rng.normal(0, 0.012) * np.sin(np.pi * k * t) for k in (1, 2, 3))
    y = y + base * y.max()

    y = y + rng.normal(0, NOISE_SD * 0.02 * y.max(), size=NPT)
    y[WATER[0]:WATER[1]] = 0.0
    return y, draw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="synthetic spectra to draw")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/figures/fig_gissmo_demo.png")
    args = ap.parse_args()

    B = np.load(SYN / "basis_600MHz.npy")
    meta = pd.read_csv(SYN / "basis_600MHz_meta.csv")
    L = np.load(SYN / "lipid_basis.npy")
    names = meta["name"].tolist()
    rng = np.random.default_rng(args.seed)

    syn = [synthesise(B, names, L, rng) for _ in range(args.n)]

    X = np.load(ROOT / CORPUS, mmap_mode="r")
    real = np.array(X[0], dtype=np.float64)

    ppm = ppm_axis()
    band = (ppm >= 0.5) & (ppm <= 4.6)          # the informative serum region

    fig, axes = plt.subplots(args.n + 1, 1, figsize=(12, 2.35 * (args.n + 1)),
                             sharex=True)
    INK, ACC, WARN = "#1f2933", "#2f6f8f", "#b04a3a"

    for i, (y, draw) in enumerate(syn):
        ax = axes[i]
        v = y[band]
        ax.plot(ppm[band], v / v.max(), lw=0.6, color=ACC)
        ax.set_ylabel("synthetic", fontsize=9)
        ax.set_title(f"Synthetic #{i + 1} — linear combination of 43 GISSMO "
                     f"metabolites + empirical lipid envelope",
                     fontsize=10, loc="left", color=ACC)

    ax = axes[-1]
    v = real[band]
    ax.plot(ppm[band], v / v.max(), lw=0.6, color=INK)
    ax.set_ylabel("real", fontsize=9)
    ax.set_title("Real corpus spectrum (row 0) — for shape comparison only; the "
                 "forward model explains 0.45–0.51 of variance, not 0.90",
                 fontsize=10, loc="left", color=WARN)
    ax.set_xlabel("chemical shift (ppm)")

    for a in axes:
        a.invert_xaxis()
        a.set_yticks([])
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)

    fig.suptitle("GISSMO forward model — ILLUSTRATIVE ONLY, has not passed the "
                 "acceptance gate", fontsize=12, y=1.0, color=WARN)
    fig.tight_layout()
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=165, bbox_inches="tight")
    print(f"wrote {p}")

    # Which metabolites actually carry the signal, for the caption.
    y, draw = syn[0]
    area = draw * np.abs(np.asarray(B, dtype=np.float64)).sum(axis=1)
    top = np.argsort(-area)[:8]
    print("\ntop contributors in synthetic #1 (by integrated area):")
    for j in top:
        print(f"   {names[j]:<20s} {draw[j]:7.3f} mM  "
              f"{100 * area[j] / area.sum():5.1f}% of area")


if __name__ == "__main__":
    main()
