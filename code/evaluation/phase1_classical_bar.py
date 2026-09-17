#!/usr/bin/env python3
"""Phase 1 -- re-establish the classical-ML bar on the rebuilt cohorts.

Whiteboard node: "ML is better - sets bar, can we better it". Nothing in Phase 3
means anything until this number exists on the corrected data, so this runs first
and its output is the reference every SSL result is measured against.

Protocol is fixed in docs/PHASE0_data_state.md and is not adjusted after seeing
results:
  * repeated stratified 10-fold x 10 repeats where n >= 100
  * LOOCV x 10 seeds where n < 100
  * hyperparameters chosen ONLY in a nested inner loop
  * balanced accuracy primary; mean with 95% CI across repeats
  * a difference below the measured single-run sd of 0.045 is not a result

MTBLS326 is run but tagged confounded=True: SSL_vs_classical_analysis.md §11
showed its labels are collinear with acquisition order by study design. It is
reported for completeness and excluded from conclusions.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

COH = Path("rebuild/cohorts")
WATER = (4.45, 5.10)
NOISE_FLOOR = 0.045


# ----------------------------------------------------------------- targets ---
def load(name):
    X = np.load(COH / f"{name}_spectra.npy", mmap_mode="r")
    ppm = np.load(COH / f"{name}_ppm_axis.npy")
    rows = list(csv.DictReader(open(COH / f"{name}_labels.csv")))
    return X, ppm, rows


def targets():
    """(id, cohort, label_fn -> y or None to drop the row, confounded)"""
    def m563_bin(r):
        lab = r["label"]
        if lab == "unknown":
            return None
        return "control" if lab == "surgery (control)" else "infection"

    def m563_3(r):
        lab = r["label"]
        return None if lab == "unknown" else lab

    def barth(r):
        lab = r["label"]
        return None if lab == "Pool" else lab          # pooled QC are not samples

    return [
        ("mtbls563_binary",  "mtbls563", m563_bin, False),
        ("mtbls563_3class",  "mtbls563", m563_3,   False),
        ("brc_t2d_cancer",   "brc_t2d",  lambda r: r["Group"].split("-")[0], False),
        ("brc_t2d_diabetes", "brc_t2d",  lambda r: r["Group"].split("-")[1], False),
        ("brc_t2d_4class",   "brc_t2d",  lambda r: r["Group"], False),
        ("barth_case_control", "barth",  barth, False),
        ("mtbls326_ip3r",    "mtbls326", lambda r: r["label"], True),
    ]


# ---------------------------------------------------------------- features ---
def binned(X, ppm, width, rows_idx):
    """Mean-pool into `width`-ppm bins, water excluded, then unit total area."""
    keep = ~((ppm <= WATER[1]) & (ppm >= WATER[0]))
    keep &= (ppm >= -0.5) & (ppm <= 9.5)
    edges = np.arange(ppm[keep].min(), ppm[keep].max(), width)
    idx = np.digitize(ppm, edges)
    M = np.array(X[rows_idx], dtype=np.float32)
    M[:, ~keep] = 0.0
    out = np.zeros((M.shape[0], len(edges)), dtype=np.float32)
    for b in range(len(edges)):
        sel = (idx == b) & keep
        if sel.any():
            out[:, b] = M[:, sel].mean(axis=1)
    area = np.abs(out).sum(axis=1, keepdims=True)
    area[area == 0] = 1
    return out / area


# ------------------------------------------------------------------ models ---
class PLSDA(PLSRegression):
    """PLS-DA: the standard metabolomics classifier, wrapped to a classifier API."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1
        if len(self.classes_) == 2:
            Y = Y[:, 1:]
        super().fit(X, Y)
        return self

    def predict(self, X):
        s = super().predict(X)
        if s.ndim == 1 or s.shape[1] == 1:
            return self.classes_[(s.ravel() > 0.5).astype(int)]
        return self.classes_[np.argmax(s, axis=1)]

    def decision_function(self, X):
        s = super().predict(X)
        return s.ravel() if (s.ndim == 1 or s.shape[1] == 1) else s


def model_grid(n_feat):
    sc = lambda est: Pipeline([("sc", StandardScaler()), ("m", est)])
    return {
        "logreg_l2": (sc(LogisticRegression(max_iter=5000, class_weight="balanced")),
                      {"m__C": [0.01, 0.1, 1, 10]}),
        "logreg_l1": (sc(LogisticRegression(max_iter=5000, penalty="l1", solver="liblinear",
                                            class_weight="balanced")),
                      {"m__C": [0.01, 0.1, 1, 10]}),
        "linear_svm": (sc(LinearSVC(max_iter=20000, class_weight="balanced")),
                       {"m__C": [0.01, 0.1, 1, 10]}),
        "rbf_svm": (sc(SVC(class_weight="balanced")),
                    {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=0, n_jobs=1),
                          {"max_depth": [None, 6]}),
        "hist_gbm": (HistGradientBoostingClassifier(random_state=0),
                     {"max_leaf_nodes": [15, 31]}),
        "pls_da": (sc(PLSDA()), {"m__n_components": [2, 4, 8]}),
        "lda": (sc(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")), {}),
    }


def score(yt, yp, sdec, classes):
    out = {"balanced_accuracy": balanced_accuracy_score(yt, yp),
           "macro_f1": f1_score(yt, yp, average="macro")}
    try:
        if len(classes) == 2 and sdec is not None:
            out["auc"] = roc_auc_score((yt == classes[1]).astype(int), sdec)
    except Exception:
        pass
    return out


def one_split(Xtr, ytr, Xte, yte, est, grid, classes, seed):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    if grid:
        gs = GridSearchCV(est, grid, scoring="balanced_accuracy", cv=inner, n_jobs=1)
        gs.fit(Xtr, ytr); best = gs.best_estimator_
    else:
        best = est.fit(Xtr, ytr)
    yp = best.predict(Xte)
    sdec = None
    try:
        sdec = best.decision_function(Xte)
        if np.ndim(sdec) > 1:
            sdec = None
    except Exception:
        try:
            sdec = best.predict_proba(Xte)[:, 1]
        except Exception:
            pass
    return yte, yp, sdec


def run_target(tid, cohort, labfn, confounded, bin_width, n_jobs):
    X, ppm, rows = load(cohort)
    idx, y = [], []
    for r in rows:
        lab = labfn(r)
        if lab is None or lab == "":
            continue
        idx.append(int(r["row"])); y.append(lab)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return []
    F = binned(X, ppm, bin_width, idx)
    classes = np.unique(y)
    n = len(y)
    small = n < 100
    results = []

    import time
    for mname, (est, grid) in model_grid(F.shape[1]).items():
        t0 = time.time()
        reps = []
        if small:
            for seed in range(10):
                cv = LeaveOneOut()
                yt_all, yp_all = [], []
                for tr, te in cv.split(F):
                    yt, yp, _ = one_split(F[tr], y[tr], F[te], y[te], est, grid, classes, seed)
                    yt_all.append(yt[0]); yp_all.append(yp[0])
                reps.append(score(np.array(yt_all), np.array(yp_all), None, classes))
        else:
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
            fold = Parallel(n_jobs=n_jobs)(
                delayed(one_split)(F[tr], y[tr], F[te], y[te], est, grid, classes, 0)
                for tr, te in cv.split(F, y))
            for k in range(10):                       # regroup folds into repeats
                sl = fold[k * 10:(k + 1) * 10]
                yt = np.concatenate([a for a, _, _ in sl])
                yp = np.concatenate([b for _, b, _ in sl])
                sd = [c for _, _, c in sl]
                sdec = np.concatenate(sd) if all(s is not None for s in sd) else None
                reps.append(score(yt, yp, sdec, classes))

        ba = np.mean([r["balanced_accuracy"] for r in reps])
        print(f"    {tid:<20} bin {bin_width:<5} {mname:<14} "
              f"bal.acc {ba:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        for metric in ("balanced_accuracy", "macro_f1", "auc"):
            vals = [r[metric] for r in reps if metric in r]
            if not vals:
                continue
            v = np.asarray(vals)
            results.append({
                "target": tid, "cohort": cohort, "model": mname, "metric": metric,
                "bin_ppm": bin_width, "n": int(n), "n_features": int(F.shape[1]),
                "classes": list(map(str, classes)), "confounded": confounded,
                "protocol": "loocv x10 seeds" if small else "stratified 10-fold x10 repeats",
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "ci95_lo": float(np.percentile(v, 2.5)), "ci95_hi": float(np.percentile(v, 97.5)),
                "n_reps": len(v)})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=float, nargs="+", default=[0.02, 0.04])
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--out", default="results/phase1_classical_bar.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    allres = []
    for tid, cohort, labfn, conf in targets():
        for bw in args.bins:
            r = run_target(tid, cohort, labfn, conf, bw, args.n_jobs)
            allres += r
            if r:
                best = max((x for x in r if x["metric"] == "balanced_accuracy"),
                           key=lambda x: x["mean"])
                print(f"{tid:<20} bin {bw:<5} n={best['n']:<4} best {best['model']:<14} "
                      f"bal.acc {best['mean']:.3f} +/- {best['sd']:.3f}"
                      f"{'   [CONFOUNDED]' if conf else ''}", flush=True)
            json.dump(allres, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}  ({len(allres)} rows)")


if __name__ == "__main__":
    main()
