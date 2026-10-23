#!/usr/bin/env python3
"""Final experiment 4 (Shyam meeting 2026-10-23): better readouts of the frozen
masking encoder -- learned attention pooling and a CLS token -- against standard
ML, on the G1 protocol (same splits, budgets and intervals as phase3_fixed_eval).

Readouts per checkpoint:
  mean        mean over the 128 tokens, then standardise + balanced LR  (G1 default)
  regional16  mean within 16 groups of 8 tokens, then the same LR
  attention   a learned query scores each (standardised) token; softmax weights
              pool the tokens; a linear softmax classifier on the pooled vector.
              Query and classifier are trained together on the training split
              ONLY (balanced cross-entropy, L2), so nothing leaks from the test
              patients. The query starts at zero, i.e. AT mean pooling, so the
              optimiser can only move away from mean pooling if the labels pay.
  cls         the final state of the CLS token (only for --ckpts with one),
              then the same LR

Standard ML is binned spectra + the same LR, as in G1.
"""
from __future__ import annotations

import argparse, glob, json, re, sys, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import target_defs, load_cohort, binned, find_ckpts  # noqa: E402
from phase3_fixed_eval import (BUDGETS, fit_predict, exact_episodes, cv_splits,  # noqa: E402
                               paired_ci, bootstrap_ci)
from linear_probe_frozen_embeddings import embed_masking  # noqa: E402

N_TOK, D = 128, 192


def attn_fit_predict(T, y, tr, te, steps=300, lr=1e-2, wd=1e-2, seed=0):
    """T: (n, N_TOK, D) tokens. Train query + classifier on tr, predict te."""
    torch.manual_seed(seed)
    classes = np.unique(y[tr]); yi = np.searchsorted(classes, y[tr])
    mu = T[tr].mean(axis=(0, 1), keepdims=True); sd = T[tr].std(axis=(0, 1), keepdims=True) + 1e-6
    Xtr = torch.from_numpy((T[tr] - mu) / sd).float(); Xte = torch.from_numpy((T[te] - mu) / sd).float()
    q = nn.Parameter(torch.zeros(D)); clf = nn.Linear(D, len(classes))
    w = torch.tensor(len(yi) / (len(classes) * np.bincount(yi, minlength=len(classes))), dtype=torch.float32)
    opt = torch.optim.Adam([q, *clf.parameters()], lr=lr)
    yt = torch.from_numpy(yi).long()
    for _ in range(steps):
        a = torch.softmax(Xtr @ q, dim=1)                      # (n, N_TOK)
        z = (a.unsqueeze(-1) * Xtr).sum(1)
        loss = nn.functional.cross_entropy(clf(z), yt, weight=w) + wd * clf.weight.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a = torch.softmax(Xte @ q, dim=1)
        return classes[clf((a.unsqueeze(-1) * Xte).sum(1)).argmax(1).numpy()]


def ckpt_list(pattern):
    if pattern is None:
        return [(s, p) for o, s, p in find_ckpts() if o == "masking"]
    out = [(int(re.search(r"_seed(\d+)_", Path(p).name).group(1)), Path(p))
           for p in sorted(glob.glob(str(ROOT / pattern)))]
    seeds = [s for s, _ in out]
    assert out and len(seeds) == len(set(seeds)), f"bad checkpoint set for {pattern}: {seeds}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--ckpts", default=None, help="glob under ROOT; default = Phase 2 baseline masking")
    ap.add_argument("--readouts", nargs="+", default=["mean", "regional16", "attention"])
    ap.add_argument("--out", default="results/final/pooling_eval_baseline.json")
    a = ap.parse_args()
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    C = ckpt_list(a.ckpts)
    print("checkpoints:", [(s, p.name) for s, p in C], flush=True)
    results = []
    for tid, cohort, labfn in target_defs():
        M, ppm, y = load_cohort(cohort, labfn)
        y = np.asarray(y); seed = zlib.crc32(tid.encode()) % 2**31
        base = binned(M, ppm, 0.02)
        F = {r: [] for r in a.readouts}
        for _, p in C:
            tok = embed_masking(str(p), M, a.device, pooling="flatten").reshape(len(M), N_TOK, D)
            if "mean" in F: F["mean"].append(tok.mean(1))
            if "regional16" in F: F["regional16"].append(tok.reshape(len(M), 16, 8, D).mean(2).reshape(len(M), -1))
            if "attention" in F: F["attention"].append(tok)
            if "cls" in F: F["cls"].append(embed_masking(str(p), M, a.device, pooling="cls"))
        fit = {r: (attn_fit_predict if r == "attention" else fit_predict) for r in F}
        print(f"\n{tid} n={len(y)}", flush=True)

        def emit(design, k, nspl, cis, sc):
            for r in F:
                d, lo, hi = cis(r)
                results.append({"target": tid, "design": design, "k_per_class": k, "n_splits": nspl,
                                "readout": r, "ssl": sc[r], "classical": sc["classical"],
                                "diff": d, "lo": lo, "hi": hi})
                print(f"  {design:<8} {str(k):<4} {r:<10} {sc[r]:.3f} vs {sc['classical']:.3f}"
                      f"  {d:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)

        for k in BUDGETS:
            eps = exact_episodes(y, k, a.episodes, seed)
            if eps is None:
                continue
            per = {"classical": [balanced_accuracy_score(y[q], fit_predict(base, y, s, q)) for s, q in eps]}
            for r, Fs in F.items():
                per[r] = [np.mean([balanced_accuracy_score(y[q], fit[r](X, y, s, q)) for X in Fs])
                          for s, q in eps]
            emit("few-shot", k, len(eps), lambda r: paired_ci(per[r], per["classical"]),
                 {r: float(np.mean(v)) for r, v in per.items()})

        design, reps = cv_splits(y, seed)

        def run(fn, X):
            out = []
            for folds in reps:
                pr = np.empty(len(y), dtype=y.dtype)
                for tr, te in folds:
                    pr[te] = fn(X, y, tr, te)
                out.append(pr)
            return out
        cls_p = run(fit_predict, base)
        preds = {r: [run(fit[r], X) for X in Fs] for r, Fs in F.items()}
        if design == "loocv":
            def vote(ps):
                st = np.stack([p[0] for p in ps]); return np.array([max(set(c), key=list(c).count) for c in st.T])
            V = {r: vote(ps) for r, ps in preds.items()}
            sc = {r: float(np.mean([balanced_accuracy_score(y, p[0]) for p in ps])) for r, ps in preds.items()}
            sc["classical"] = float(balanced_accuracy_score(y, cls_p[0]))
            emit(design, None, len(y), lambda r: bootstrap_ci(y, V[r], cls_p[0]), sc)
        else:
            per = {r: [np.mean([balanced_accuracy_score(y, p[i]) for p in ps]) for i in range(len(reps))]
                   for r, ps in preds.items()}
            per["classical"] = [balanced_accuracy_score(y, p) for p in cls_p]
            emit(design, None, len(reps), lambda r: paired_ci(per[r], per["classical"]),
                 {r: float(np.mean(v)) for r, v in per.items()})
        json.dump(results, open(ROOT / a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
