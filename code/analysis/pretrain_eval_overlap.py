#!/usr/bin/env python3
"""Experiment #20: how close are the evaluation cohorts to the pretraining corpus?

This answers two questions that the next-steps slide was silently assuming
answers to.

Q1 -- LEAKAGE. Are any evaluation spectra actually *in* the pretraining corpus?
If they were, every downstream number in this document would be optimistic and
"held-out" would be a misnomer. §19 found the corpus contains near-duplicates of
itself (1.1% at r > 0.9999), so the same machinery has to be pointed at the
train/eval boundary before any of it can be trusted.

Q2 -- COVERAGE. Assuming no leakage, how far outside the pretraining
distribution do the evaluation cohorts sit? §19 established that within the
corpus a spectrum's nearest neighbour is at r ~= 0.99, i.e. the corpus is nearly
self-redundant. If the evaluation cohorts' nearest corpus neighbour is far
below that, then the backbone is being asked to generalise to distributions it
saw nothing like, and "the corpus is too small" (#17) is the wrong diagnosis --
the corpus is too NARROW.

Method: bin to 2048 points, mean-centre and L2-normalise each spectrum, then for
every evaluation spectrum take the maximum cosine similarity (= Pearson r on the
binned vector) against all corpus rows. Reports the distribution of that best
match per cohort, plus a within-corpus reference computed the same way.

CAVEAT ON NORMALISATION. Cohorts are compared using their v4 rowMinMax arrays
where those exist, so they match the corpus's own rowMinMax preprocessing. The
TBI Tirupati array has not been through the v4 pipeline (no rowMinMax stage), so
its number mixes a genuine distribution gap with a preprocessing difference and
is reported separately as indicative only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_"
          "Suppressed_rowMinMax_v4.npy")

# (label, path, comparable_preprocessing). Each of these is a DISTINCT cohort
# with a DISTINCT prediction task -- Barth syndrome, IP3R expression, a 3-class
# diagnosis, cancer status, diabetes status. No two share a label space, which
# is why supervised cross-cohort transfer is not executable here (see §20).
COHORTS = [
    ("Barth", "data/Barth/aligned_128K_Workbench_Barth_Syndrome_"
              "WS625to680Zero_EDTASuppressed_rowMinMax_v4.npy", True),
    ("MTBLS326", "data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy", True),
    ("MTBLS563", "data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy", True),
    ("BrC-T2D", "data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_rowMinMax_v4.npy", True),
    ("TBI Tirupati", "data/tbi_tirupati/aligned_128K_TBI_Tirupati_WS625to680Zero.npy", False),
]


def normalise(arr: np.ndarray, n_bins: int) -> np.ndarray:
    fold = arr.shape[1] // n_bins
    binned = np.asarray(arr[:, :n_bins * fold], dtype=np.float32)
    binned = binned.reshape(arr.shape[0], n_bins, fold).mean(axis=2)
    z = binned - binned.mean(1, keepdims=True)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bins", type=int, default=2048)
    ap.add_argument("--corpus-sample", type=int, default=1500,
                    help="rows used for the within-corpus reference")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="results/analysis/pretrain_eval_overlap")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = np.load(ROOT / CORPUS, mmap_mode="r")
    print(f"pretraining corpus: {corpus.shape[0]} x {corpus.shape[1]}")
    cz = normalise(corpus, args.n_bins)

    rows = []
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cz), min(args.corpus_sample, len(cz)), replace=False)
    sim = cz[idx] @ cz.T
    for i, j in enumerate(idx):
        sim[i, j] = -1.0
    ref = sim.max(1)
    rows.append(dict(cohort="(within-corpus reference)", n=len(idx),
                     comparable_preprocessing=True,
                     median_best_r=float(np.median(ref)),
                     p10_best_r=float(np.percentile(ref, 10)),
                     n_gt_0_9999=int((ref > 0.9999).sum()),
                     n_gt_0_99=int((ref > 0.99).sum())))
    print(f"\n  {'cohort':<26} {'n':>5}  {'median best r':>13}  {'>0.9999':>8}  {'>0.99':>6}")
    print(f"  {'(within-corpus reference)':<26} {len(idx):5d}  {np.median(ref):13.4f}  "
          f"{(ref > 0.9999).sum():8d}  {(ref > 0.99).sum():6d}")

    for name, rel, comparable in COHORTS:
        path = ROOT / rel
        if not path.exists():
            print(f"  {name:<26} MISSING: {rel}")
            continue
        arr = np.load(path, mmap_mode="r")
        if arr.shape[1] != corpus.shape[1]:
            print(f"  {name:<26} incompatible length {arr.shape[1]}; skipped")
            continue
        best = (normalise(arr, args.n_bins) @ cz.T).max(1)
        tag = "" if comparable else "   (preprocessing NOT matched -- indicative only)"
        print(f"  {name:<26} {arr.shape[0]:5d}  {np.median(best):13.4f}  "
              f"{(best > 0.9999).sum():8d}  {(best > 0.99).sum():6d}{tag}")
        rows.append(dict(cohort=name, n=int(arr.shape[0]),
                         comparable_preprocessing=comparable,
                         median_best_r=float(np.median(best)),
                         p10_best_r=float(np.percentile(best, 10)),
                         n_gt_0_9999=int((best > 0.9999).sum()),
                         n_gt_0_99=int((best > 0.99).sum())))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pretrain_eval_overlap.csv", index=False)

    leaked = df[(df.cohort != "(within-corpus reference)") & (df.n_gt_0_9999 > 0)]
    print("\n" + "=" * 74)
    if leaked.empty:
        print("  Q1 LEAKAGE: none. No evaluation spectrum has a near-duplicate (r > 0.9999)")
        print("     anywhere in the pretraining corpus. 'Held-out' is accurate.")
    else:
        print(f"  Q1 LEAKAGE: *** {int(leaked.n_gt_0_9999.sum())} near-duplicate(s) found *** "
              f"in {', '.join(leaked.cohort)}")
    ref_med = float(df[df.cohort == "(within-corpus reference)"].median_best_r.iloc[0])
    comp = df[(df.cohort != "(within-corpus reference)") & df.comparable_preprocessing]
    print(f"\n  Q2 COVERAGE: within the corpus, median best match is {ref_med:.4f}.")
    print(f"     For the evaluation cohorts it is "
          f"{comp.median_best_r.min():.3f}-{comp.median_best_r.max():.3f}.")
    print("     The corpus does not cover the evaluation distribution: the problem is")
    print("     that it is NARROW, not (only) that it is small. See §20.")
    print(f"\nWrote {out_dir}/pretrain_eval_overlap.csv")


if __name__ == "__main__":
    main()
