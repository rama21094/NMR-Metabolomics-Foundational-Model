#!/usr/bin/env python3
"""Phase 5.2 -- is the synthetic data good enough to train on?

Three tests, all of which the synthetic set can fail:

1. DISCRIMINATOR. Logistic regression on binned spectra, real vs synthetic,
   study-level CV on the real side. AUC near 0.5 means indistinguishable; AUC
   near 1.0 means a network will learn "is this synthetic?" instead of chemistry.
2. LANDMARKS. Lactate CH3 and the CH3->CH separation, the same test the corpus
   rebuild was verified with.
3. INTENSITY DISTRIBUTIONS. Per-bin Wasserstein-1 against real, scaled by the
   real pooled IQR -- the Phase 4 metric, so the numbers are comparable to the
   0.053 within-corpus floor measured there.
"""
import argparse, csv
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WATER = (4.45, 5.10)


def binned(M, ppm, width=0.02):
    keep = ~((ppm <= WATER[1]) & (ppm >= WATER[0])) & (ppm >= -0.5) & (ppm <= 9.5)
    cols = np.where(keep)[0]; kppm = ppm[cols]
    edges = np.arange(kppm.min(), kppm.max(), width)
    which = np.digitize(kppm, edges); order = np.argsort(which, kind="stable")
    cols_s, which_s = cols[order], which[order]
    labels = np.unique(which_s); starts = np.searchsorted(which_s, labels)
    counts = np.diff(np.append(starts, len(which_s)))
    out = np.add.reduceat(np.asarray(M[:, cols_s], dtype=np.float32), starts, axis=1) / counts
    a = np.abs(out).sum(axis=1, keepdims=True); a[a == 0] = 1
    centres = np.array([kppm[order][s:s+c].mean() for s, c in zip(starts, counts)])
    return out / a, centres


def peak(y, ppm, lo, hi):
    m = np.where((ppm <= hi) & (ppm >= lo))[0]
    return ppm[m[np.argmax(y[m])]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--n-real", type=int, default=1500)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "train"]
    studies = np.array([r["study"] for r in split])
    S = np.load(ROOT / args.synth)
    ri = np.sort(rng.choice(X.shape[0], min(args.n_real, X.shape[0]), replace=False))
    R = np.asarray(X[ri], dtype=np.float32)
    print(f"real {R.shape}, synthetic {S.shape}\n")

    Rb, centres = binned(R, ppm)
    Sb, _ = binned(S, ppm)

    # 1. discriminator
    Z = np.vstack([Rb, Sb]); y = np.r_[np.zeros(len(Rb)), np.ones(len(Sb))]
    g = np.r_[studies[ri], np.array([f"SYN{i%5}" for i in range(len(Sb))])]
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(Z, y, g):
        sc = StandardScaler().fit(Z[tr])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Z[tr]), y[tr])
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], m.predict_proba(sc.transform(Z[te]))[:, 1]))
    print(f"1. discriminator real-vs-synthetic AUC = {np.mean(aucs):.3f} "
          f"(+/- {np.std(aucs):.3f})   0.5 = indistinguishable")

    # 2. landmarks
    rm, sm = np.median(R, axis=0), np.median(S, axis=0)
    print("\n2. landmarks (ppm)")
    for nm, lo, hi in [("lactate CH3", 1.24, 1.42), ("lactate CH", 4.02, 4.20),
                       ("creatine CH3", 2.95, 3.10)]:
        print(f"   {nm:<14} real {peak(rm,ppm,lo,hi):+.3f}   synthetic {peak(sm,ppm,lo,hi):+.3f}")
    sep_r = peak(rm,ppm,4.02,4.20) - peak(rm,ppm,1.24,1.42)
    sep_s = peak(sm,ppm,4.02,4.20) - peak(sm,ppm,1.24,1.42)
    print(f"   CH3->CH separation  real {sep_r:.3f}   synthetic {sep_s:.3f}   expected 2.780")

    # 3. intensity distributions
    n = min(len(Rb), len(Sb))
    d = []
    for j in range(Rb.shape[1]):
        a, b = Rb[:n, j], Sb[:n, j]
        q75, q25 = np.percentile(a, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            d.append(wasserstein_distance(a, b) / iqr)
    d = np.array(d)
    print(f"\n3. intensity distributions, {len(d)} bins")
    print(f"   median rel. Wasserstein {np.median(d):.3f}   p90 {np.percentile(d,90):.3f}")
    print(f"   bins within the 0.053 within-corpus floor: {np.mean(d<0.053):.1%}")
    print(f"   bins below 0.25: {np.mean(d<0.25):.1%}")


if __name__ == "__main__":
    main()
