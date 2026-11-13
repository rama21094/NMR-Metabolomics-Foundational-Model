#!/usr/bin/env python3
"""Final experiment 1 (Shyam meeting 2026-10-23): classifiers on the top PCA
components vs on the full binned spectrum.

Question it answers: does the diagnostic signal live in the main directions of
variation of NMR spectra, or in low-variance directions? If a classifier on the
top 40-50 components matches binning, the signal is in the high-variance
subspace; if it falls short, the signal hides in directions that a
reconstruction objective (which is dominated by high-variance structure) has
little reason to learn.

PCA is fitted on the binned PRETRAINING corpus (8,545 spectra, no labels), the
same unlabelled data the encoders saw, so it is a label-free transform and
composes with the cross-validated probe without leakage. The probe, splits,
budgets and intervals are exactly those of phase3_fixed_eval.py.
"""
import argparse, json, sys, zlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import target_defs, load_cohort, binned  # noqa: E402
from phase3_fixed_eval import (BUDGETS, fit_predict, exact_episodes, cv_splits,  # noqa: E402
                               paired_ci, bootstrap_ci)

KS = [5, 10, 20, 40, 50, 100]


def corpus_binned(ppm):
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    return np.vstack([binned(np.asarray(X[i:i + 500], dtype=np.float32), ppm, 0.02)
                      for i in range(0, X.shape[0], 500)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results/final/pca_vs_binning.json")
    a = ap.parse_args()
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    C = corpus_binned(ppm)
    pca = PCA(n_components=max(KS), random_state=0).fit(C)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"corpus binned {C.shape}; variance explained: " +
          ", ".join(f"{k}: {cum[k-1]:.3f}" for k in KS), flush=True)
    out = {"explained": {k: float(cum[k - 1]) for k in KS}, "results": []}

    for tid, cohort, labfn in target_defs():
        M, cppm, y = load_cohort(cohort, labfn)
        assert np.allclose(cppm, ppm), f"{cohort} axis differs from corpus axis"
        y = np.asarray(y); seed = zlib.crc32(tid.encode()) % 2**31
        B = binned(M, cppm, 0.02)
        feats = {f"pca{k}": pca.transform(B)[:, :k] for k in KS}
        print(f"\n{tid} n={len(y)}", flush=True)

        def emit(design, k, nspl, cis, scores):
            for name in feats:
                d, lo, hi = cis(name)
                out["results"].append({"target": tid, "design": design, "k_per_class": k,
                                       "n_splits": nspl, "features": name,
                                       "acc": scores[name], "binned": scores["binned"],
                                       "diff": d, "lo": lo, "hi": hi})
                print(f"  {design:<8} {str(k):<4} {name:<6} {scores[name]:.3f} vs binned "
                      f"{scores['binned']:.3f}  {d:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)

        allf = {"binned": B, **feats}
        for k in BUDGETS:
            eps = exact_episodes(y, k, a.episodes, seed)
            if eps is None:
                continue
            per = {n: [balanced_accuracy_score(y[q], fit_predict(F, y, s, q)) for s, q in eps]
                   for n, F in allf.items()}
            emit("few-shot", k, len(eps), lambda n: paired_ci(per[n], per["binned"]),
                 {n: float(np.mean(v)) for n, v in per.items()})
        design, reps = cv_splits(y, seed)
        preds = {}
        for n, F in allf.items():
            preds[n] = []
            for folds in reps:
                pr = np.empty(len(y), dtype=y.dtype)
                for tr, te in folds:
                    pr[te] = fit_predict(F, y, tr, te)
                preds[n].append(pr)
        if design == "loocv":
            sc = {n: float(balanced_accuracy_score(y, p[0])) for n, p in preds.items()}
            emit(design, None, len(y), lambda n: bootstrap_ci(y, preds[n][0], preds["binned"][0]), sc)
        else:
            per = {n: [balanced_accuracy_score(y, p) for p in ps] for n, ps in preds.items()}
            emit(design, None, len(reps), lambda n: paired_ci(per[n], per["binned"]),
                 {n: float(np.mean(v)) for n, v in per.items()})
        json.dump(out, open(ROOT / a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
