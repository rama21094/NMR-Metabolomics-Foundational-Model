#!/usr/bin/env python3
"""Check that a rebuilt corpus really is on one ppm axis. Run this before using it.

The whole point of the rebuild is that spectra acquired at different spectral
widths end up on a common axis. That claim is checkable, and it must be checked:
the old corpus looked fine by every casual measure and was not.

THE TEST THAT MATTERS (docs/PI_outline.md 6a). Group the rows by the spectral
width they were ACQUIRED at, take each group's median, and scan the scale factor
that best maps it onto the median of the largest group. If the rebuild worked,
every group peaks at scale 1.00 with a sharp, dominant maximum.

That distinction is the one that caught seven false positives in the old corpus:

    correct   : best scale 1.00, peak 0.844, runner-up 0.507   <- sharp, dominant
    incorrect : best scale 1.90, peak 0.527, runner-up 0.482   <- broad plateau

A high correlation on its own proves nothing. What proves it is the optimum
landing on 1.00 AND standing clear of its runner-up.

Also reported:
  * absolute position of known metabolite landmarks -- the rebuild keeps real ppm,
    so lactate must appear at 1.33 and not merely at a consistent index;
  * per-group displacement, which should now be a few points rather than thousands.

Usage:
    python verify_corpus.py plasma_spectra.npy plasma_ppm_axis.npy plasma_provenance.csv
"""
from __future__ import annotations

import argparse
import collections
import csv

import numpy as np

# (name, ppm, tolerance) -- singlets/multiplets that are unambiguous in serum
LANDMARKS = [("lactate CH3", 1.33, 0.06),
             ("alanine CH3", 1.47, 0.05),
             ("creatine CH3", 3.03, 0.05),
             ("glucose H1(a)", 5.22, 0.05)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spectra"); ap.add_argument("axis"); ap.add_argument("provenance")
    ap.add_argument("--band", type=float, nargs=2, default=(0.6, 4.2),
                    help="ppm window used for the comparisons")
    ap.add_argument("--cap", type=int, default=120)
    args = ap.parse_args()

    X = np.load(args.spectra, mmap_mode="r")
    ppm = np.load(args.axis)
    prov = list(csv.DictReader(open(args.provenance)))
    n, npt = X.shape
    print(f"{n:,} spectra x {npt:,} points")
    print(f"axis {ppm[0]:+.3f} .. {ppm[-1]:+.3f} ppm, "
          f"{(ppm[0] - ppm[-1]) / (npt - 1):.4e} ppm/point")
    if len(prov) != n:
        print(f"  !! provenance has {len(prov)} rows, array has {n} -- STOP")
        return

    groups = collections.defaultdict(list)
    for i, r in enumerate(prov):
        try:
            groups[round(float(r["sw_ppm"]), 3)].append(i)
        except (KeyError, ValueError):
            pass
    print("\nacquisition spectral widths present:")
    for w in sorted(groups):
        print(f"   {w:<10} n={len(groups[w]):,}")

    lo = int(np.argmin(np.abs(ppm - args.band[1])))
    hi = int(np.argmin(np.abs(ppm - args.band[0])))
    cols = np.arange(lo, hi)
    pivot = (lo + hi) // 2

    def med(rows):
        return np.median(np.array(X[sorted(rows)[:args.cap]], dtype=np.float64), axis=0)

    big = max(groups, key=lambda k: len(groups[k]))
    ref = med(groups[big])
    rr = ref[cols] - ref[cols].mean()
    rr /= np.linalg.norm(rr) + 1e-12
    m = len(cols); nf = 1 << (2 * m - 1).bit_length()
    Rf = np.conj(np.fft.rfft(rr, nf))
    grid = np.arange(npt, dtype=np.float64)

    def peak(y, s):
        if s != 1.0:
            y = np.interp(pivot + (grid - pivot) / s, grid, y, left=0.0, right=0.0)
        z = y[cols] - y[cols].mean(); nz = np.linalg.norm(z)
        if nz < 1e-12:
            return -1.0, 0
        cc = np.fft.irfft(np.fft.rfft(z / nz, nf) * Rf, nf)
        cc = np.concatenate([cc[-(m - 1):], cc[:m]])
        k = int(cc.argmax())
        return float(cc[k]), k - (m - 1)

    scales = np.round(np.arange(0.60, 2.01, 0.05), 2)
    print(f"\nscale scan against the largest group (SW {big}), "
          f"{args.band[0]}-{args.band[1]} ppm:")
    print(f"{'SW':<10}{'n':>6}{'best':>8}{'peak':>8}{'runner':>8}{'disp':>7}  verdict")
    allgood = True
    for w in sorted(groups):
        y = med(groups[w])
        vals = np.array([peak(y, s)[0] for s in scales])
        order = np.argsort(-vals)
        bs, bp = scales[order[0]], vals[order[0]]
        runner = max(vals[i] for i in order[1:] if abs(scales[i] - bs) > 0.12) \
            if len(order) > 1 else 0.0
        _, disp = peak(y, 1.0)
        sharp = bp - runner
        good = abs(bs - 1.0) < 1e-9 and sharp > 0.15
        allgood &= good
        verdict = ("OK" if good else
                   f"FAIL -- optimum at {bs:.2f}, not 1.00" if abs(bs - 1.0) > 1e-9
                   else f"WEAK -- peak only {sharp:+.2f} above runner-up")
        print(f"{w:<10}{len(groups[w]):>6}{bs:>8.2f}{bp:>8.3f}"
              f"{runner:>8.3f}{disp:>7d}  {verdict}")

    print("\nabsolute landmark positions in the pooled median "
          "(the rebuild keeps real ppm, so these must be right):")
    allm = med(list(range(min(n, 400))))
    for name, want, tol in LANDMARKS:
        a = int(np.argmin(np.abs(ppm - (want + tol))))
        b = int(np.argmin(np.abs(ppm - (want - tol))))
        if b <= a:
            continue
        got = ppm[a + int(np.argmax(allm[a:b]))]
        d = got - want
        print(f"   {name:<16} expected {want:+.3f}  found {got:+.3f}  "
              f"({d:+.3f} ppm){'' if abs(d) <= tol * 0.8 else '   <-- off'}")

    print("\n" + ("PASS: every width group peaks at scale 1.00."
                  if allgood else
                  "FAIL: at least one group does not. Do NOT use this corpus; "
                  "the axis is not common."))


if __name__ == "__main__":
    main()
