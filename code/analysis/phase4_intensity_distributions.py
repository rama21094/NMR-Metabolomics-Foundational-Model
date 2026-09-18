#!/usr/bin/env python3
"""Phase 4.2 / 4.3 -- the peak-intensity distributions, and whether p(y|x) is defined.

The PI's whiteboard asks, under "is data limiting":

    distributions of peak intensities
      - how they change w/ increasing data
      - have we defined experimental y-distribution at all x
          if yes, use synthetic data (generative AI or LC of metabolite spectra)
          if no,  we end -- more publicly available exptl data needed

This is the gate that decides the branch, so the measurement has to be one that
can come back "no".

METHOD. At each ppm position x, p(y|x) is the distribution of intensities across
spectra. "Defined" means: we have enough spectra that this distribution has
stopped moving as we add more. Measuring that against the full-corpus estimate
would be circular -- the full corpus is itself a finite sample. So instead we
split the available spectra into two DISJOINT halves of size N, estimate p(y|x)
independently in each, and measure the Wasserstein-1 distance between them.

  * If two independent samples of size N already agree, p(y|x) is pinned down at
    that N, and more data would not change it.
  * If they still disagree, the distribution is not yet determined, and any
    synthetic generator trained on it would be inventing the parts we have not
    measured.

The distance is reported RELATIVE to the distribution's own interquartile range,
so it is comparable across ppm regions with wildly different intensity scales.

Halves are split BY STUDY wherever possible: two halves drawn at random from the
same studies are not independent samples of the underlying population, because
within-study nearest-neighbour correlation is 0.989. A row-wise split would
report convergence that is really just duplication.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
WATER = (4.45, 5.10)


def bin_spectra(X, ppm, width, chunk=512):
    """Mean-pool every spectrum into `width`-ppm bins, reading the array ONCE.

    The obvious implementation -- loop over bins, slice `X[rows][:, mask]` -- is
    catastrophic on a memmap: each iteration materialises the whole 4.5 GB array
    again. This reads row-chunks once and reduces all bins in a single pass with
    np.add.reduceat over the contiguous in-band columns.
    """
    keep = ~((ppm <= WATER[1]) & (ppm >= WATER[0]))
    keep &= (ppm >= -0.5) & (ppm <= 9.5)
    cols = np.where(keep)[0]
    kppm = ppm[cols]
    edges = np.arange(kppm.min(), kppm.max(), width)
    which = np.digitize(kppm, edges)          # bin index per kept column
    order = np.argsort(which, kind="stable")
    cols_s, which_s = cols[order], which[order]
    starts = np.searchsorted(which_s, np.unique(which_s))
    labels = np.unique(which_s)
    counts = np.diff(np.append(starts, len(which_s)))
    centres = np.array([kppm[order][a:a + c].mean()
                        for a, c in zip(starts, counts)])

    n = X.shape[0]
    out = np.zeros((n, len(labels)), dtype=np.float32)
    for i in range(0, n, chunk):
        blk = np.asarray(X[i:i + chunk][:, cols_s], dtype=np.float32)
        out[i:i + blk.shape[0]] = np.add.reduceat(blk, starts, axis=1) / counts
        if (i // chunk) % 4 == 0:
            print(f"\r  binning {min(i + chunk, n)}/{n}", end="", flush=True)
    print()
    a = np.abs(out).sum(axis=1, keepdims=True); a[a == 0] = 1
    return out / a, centres


def halves_distance(B, ia, ib):
    """Per-bin Wasserstein-1 between two disjoint samples, scaled by pooled IQR."""
    d = np.zeros(B.shape[1]); scale = np.zeros(B.shape[1])
    for j in range(B.shape[1]):
        a, b = B[ia, j], B[ib, j]
        pooled = np.concatenate([a, b])
        q75, q25 = np.percentile(pooled, [75, 25])
        iqr = q75 - q25
        scale[j] = iqr
        d[j] = wasserstein_distance(a, b)
    rel = np.where(scale > 0, d / np.maximum(scale, 1e-12), np.nan)
    return d, rel, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-ppm", type=float, default=0.02)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 250, 500, 1000, 2000, 4000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/phase4/intensity_distributions.json")
    args = ap.parse_args()

    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "train"]
    studies = np.array([r["study"] for r in split])
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    print(f"{X.shape[0]:,} spectra, {len(set(studies))} studies, "
          f"bin {args.bin_ppm} ppm")

    B, centres = bin_spectra(X, ppm, args.bin_ppm)
    print(f"binned to {B.shape[1]} positions\n")

    uniq = sorted(set(studies))
    rng = np.random.default_rng(0)
    out = {"bin_ppm": args.bin_ppm, "centres": centres.tolist(),
           "n_spectra": int(X.shape[0]), "convergence": [], "per_bin_at_max": None}

    print("4.2  does p(y|x) stop moving as data is added?")
    print("     (Wasserstein between two DISJOINT study-split samples, "
          "relative to pooled IQR)")
    print(f"{'N per half':>11}{'median rel.':>13}{'p90 rel.':>10}{'bins<0.10':>11}"
          f"{'bins<0.25':>11}")
    for N in args.sizes:
        rels = []
        for rep in range(args.repeats):
            perm = list(rng.permutation(uniq))
            halfA = set(perm[:len(perm) // 2]); halfB = set(perm[len(perm) // 2:])
            poolA = np.where(np.isin(studies, list(halfA)))[0]
            poolB = np.where(np.isin(studies, list(halfB)))[0]
            if len(poolA) < N or len(poolB) < N:
                continue
            ia = rng.choice(poolA, size=N, replace=False)
            ib = rng.choice(poolB, size=N, replace=False)
            _, rel, _ = halves_distance(B, ia, ib)
            rels.append(rel)
        if not rels:
            print(f"{N:>11}   (insufficient spectra on one side of a study split)")
            continue
        R = np.nanmedian(np.vstack(rels), axis=0)
        row = {"n_per_half": N, "median_rel": float(np.nanmedian(R)),
               "p90_rel": float(np.nanpercentile(R, 90)),
               "frac_bins_below_0.10": float(np.nanmean(R < 0.10)),
               "frac_bins_below_0.25": float(np.nanmean(R < 0.25)),
               "n_repeats": len(rels)}
        out["convergence"].append(row)
        print(f"{N:>11}{row['median_rel']:>13.3f}{row['p90_rel']:>10.3f}"
              f"{row['frac_bins_below_0.10']:>11.2%}{row['frac_bins_below_0.25']:>11.2%}",
              flush=True)
        out["per_bin_at_max"] = R.tolist()
        json.dump(out, open(ROOT / args.out, "w"), indent=1)

    R = np.asarray(out["per_bin_at_max"], dtype=float)
    print("\n4.3  where is p(y|x) NOT defined, at the largest N tested?")
    bad = np.where(~(R < 0.25))[0]
    print(f"     {len(bad)} of {len(R)} bins ({len(bad)/len(R):.1%}) still "
          f"disagree by >25% of their IQR")
    if len(bad):
        runs, start = [], bad[0]
        for a, b in zip(bad, bad[1:]):
            if b != a + 1:
                runs.append((start, a)); start = b
        runs.append((start, bad[-1]))
        runs = sorted(runs, key=lambda r: r[0] - r[1])
        print("     widest unresolved regions (ppm):")
        for lo, hi in runs[:8]:
            print(f"       {centres[hi]:+.2f} .. {centres[lo]:+.2f}   "
                  f"({hi - lo + 1} bins, median rel. dist "
                  f"{np.nanmedian(R[lo:hi+1]):.2f})")

    json.dump(out, open(ROOT / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
