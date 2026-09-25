#!/usr/bin/env python3
"""Gate G1, re-evaluated with the three defects of the Phase 3 design fixed.

1. BUDGETS ARE EXACT. Phase 3 capped each class at (size - 1), so for the small
   cohorts the "20", "40" and "all" labels did not mean what they said: on
   Barth, "40" trained on 35 of 37 patients and tested on 2, and on three
   cohorts "40" trained on MORE patients than "all". A budget k is now used only
   if EVERY class keeps at least MIN_TEST patients for testing; otherwise it is
   dropped for that cohort, not silently capped.

2. FULL-DATA EVALUATION MATCHES PHASE 1. The classical bar was set with LOOCV on
   small cohorts and repeated stratified CV on large ones, but the pretrained
   models were only ever evaluated with few-shot episodes. Here both arms get
   the same protocol: LOOCV when n < 100 (Barth), 10 x stratified 5-fold CV
   otherwise.

3. RESULTS ARE PAIRED DIFFERENCES WITH INTERVALS. Averaging balanced accuracy
   across targets mixes chance levels of 0.50, 0.33 and 0.25, so the pooled
   number means nothing clinically. Every result is reported per target as
   (pretrained - standard ML) on identical splits, with a 95% interval:
     episodes / CV repeats   mean paired difference +/- 1.96 standard errors
     LOOCV                   paired bootstrap over patients (2,000 resamples)
   Seeds are averaged per split first, so the interval reflects split-to-split
   variation of the seed-averaged model, not seed noise alone.

Both arms use the identical readout (StandardScaler + balanced logistic
regression), as in Phase 3, so a difference cannot come from the classifier.
"""
from __future__ import annotations

import argparse, json, sys, zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import target_defs, load_cohort, binned, find_ckpts  # noqa: E402

MIN_TEST = 5
BUDGETS = [5, 10, 20, 40]


def fit_predict(F, y, tr, te):
    sc = StandardScaler().fit(F[tr])
    m = LogisticRegression(max_iter=3000, class_weight="balanced")
    m.fit(sc.transform(F[tr]), y[tr])
    return m.predict(sc.transform(F[te]))


def exact_episodes(y, k, n_ep, seed):
    """k per class exactly, or None if any class would keep < MIN_TEST."""
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < k + MIN_TEST:
        return None
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_ep):
        sup = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False)
                                      for c in classes]))
        out.append((sup, np.setdiff1d(np.arange(len(y)), sup)))
    return out


def cv_splits(y, seed):
    """LOOCV for n < 100, else 10 x stratified 5-fold. Returns list of repeats,
    each a list of (train, test) that together cover every patient once."""
    if len(y) < 100:
        return "loocv", [list(LeaveOneOut().split(np.zeros(len(y))))]
    reps = []
    for r in range(10):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed + r)
        reps.append(list(skf.split(np.zeros(len(y)), y)))
    return "cv10x5", reps


def paired_ci(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return float(d.mean()), float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)


def bootstrap_ci(y, pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    base = balanced_accuracy_score(y, pa) - balanced_accuracy_score(y, pb)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < len(np.unique(y)):
            continue
        ds.append(balanced_accuracy_score(y[i], pa[i]) - balanced_accuracy_score(y[i], pb[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(base), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results/phase3_fixed/g1_fixed.json")
    args = ap.parse_args()
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    from linear_probe_frozen_embeddings import embed_masking, embed_jigsaw, embed_joint
    EMB = {"masking": embed_masking, "jigsaw": embed_jigsaw, "joint": embed_joint}

    C = find_ckpts()
    print(f"{len(C)} checkpoints: " + ", ".join(sorted({o for o, _, _ in C})), flush=True)
    results = []
    for tid, cohort, labfn in target_defs():
        M, ppm, y = load_cohort(cohort, labfn)
        y = np.asarray(y)
        seed = zlib.crc32(tid.encode()) % 2**31
        base = binned(M, ppm, 0.02)
        embs = defaultdict(list)
        for obj, s, ck in C:
            try:
                embs[obj].append(np.asarray(EMB[obj](str(ck), M, args.device), dtype=np.float32))
            except Exception as e:
                print(f"  embed fail {obj} s{s}: {type(e).__name__}: {e}", flush=True)
        classes, counts = np.unique(y, return_counts=True)
        print(f"\n{tid}  n={len(y)}  classes {dict(zip(classes, counts))}", flush=True)

        # ---- few-shot, exact budgets --------------------------------------
        for k in BUDGETS:
            eps = exact_episodes(y, k, args.episodes, seed)
            if eps is None:
                print(f"  k={k:<3} dropped: smallest class {counts.min()} < {k}+{MIN_TEST}")
                continue
            cls = [balanced_accuracy_score(y[q], fit_predict(base, y, s, q)) for s, q in eps]
            for obj, Fs in embs.items():
                ssl = [np.mean([balanced_accuracy_score(y[q], fit_predict(F, y, s, q)) for F in Fs])
                       for s, q in eps]
                d, lo, hi = paired_ci(ssl, cls)
                results.append({"target": tid, "objective": obj, "design": "few-shot",
                                "k_per_class": k, "n_splits": len(eps), "n_seeds": len(Fs),
                                "ssl": float(np.mean(ssl)), "classical": float(np.mean(cls)),
                                "diff": d, "lo": lo, "hi": hi})
                print(f"  k={k:<3} {obj:<8} ssl {np.mean(ssl):.3f}  classical {np.mean(cls):.3f}"
                      f"  diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)

        # ---- full data: LOOCV or repeated CV -------------------------------
        design, reps = cv_splits(y, seed)
        def run(F):
            out = []
            for folds in reps:
                pred = np.empty(len(y), dtype=y.dtype)
                for tr, te in folds:
                    pred[te] = fit_predict(F, y, tr, te)
                out.append(pred)
            return out
        cls_p = run(base)
        for obj, Fs in embs.items():
            per_seed = [run(F) for F in Fs]
            if design == "loocv":
                # seed-average by majority vote per patient, then paired bootstrap
                stack = np.stack([ps[0] for ps in per_seed])
                vote = np.array([max(set(col), key=list(col).count) for col in stack.T])
                d, lo, hi = bootstrap_ci(y, vote, cls_p[0])
                ssl_v = float(np.mean([balanced_accuracy_score(y, ps[0]) for ps in per_seed]))
                cls_v = float(balanced_accuracy_score(y, cls_p[0]))
                nspl = len(y)
            else:
                ssl_r = [np.mean([balanced_accuracy_score(y, ps[r]) for ps in per_seed])
                         for r in range(len(reps))]
                cls_r = [balanced_accuracy_score(y, p) for p in cls_p]
                d, lo, hi = paired_ci(ssl_r, cls_r)
                ssl_v, cls_v, nspl = float(np.mean(ssl_r)), float(np.mean(cls_r)), len(reps)
            results.append({"target": tid, "objective": obj, "design": design,
                            "k_per_class": None, "n_splits": nspl, "n_seeds": len(Fs),
                            "ssl": ssl_v, "classical": cls_v, "diff": d, "lo": lo, "hi": hi})
            print(f"  {design:<7} {obj:<8} ssl {ssl_v:.3f}  classical {cls_v:.3f}"
                  f"  diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
        json.dump(results, open(ROOT / args.out, "w"), indent=1)
    json.dump(results, open(ROOT / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
