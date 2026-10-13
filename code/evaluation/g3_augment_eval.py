#!/usr/bin/env python3
"""G3 -- does adding GAN v2 spectra to pretraining improve transfer?

Arms (masking objective, baseline architecture, 3 seeds each):
  realblank   real corpus, edges blanked                    (the control)
  realgan     real corpus + 4,000 GAN v2, edges blanked
  baseline    Phase 2 masking, unblanked -- shown for reference only; it differs
              from both arms in the edges, so it is not the control.

The G3 answer is the paired difference realgan - realblank. Each arm is also
compared with standard ML. The protocol, budgets, episodes and intervals are
exactly those of phase3_fixed_eval.py (same seeds, so the same splits); cohort
spectra are edge-blanked before being embedded by the blanked arms.
"""
from __future__ import annotations

import argparse, json, re, sys, zlib
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis", "code/synthesis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import target_defs, load_cohort, binned, find_ckpts  # noqa: E402
from phase3_fixed_eval import (BUDGETS, fit_predict, exact_episodes, cv_splits,  # noqa: E402
                               paired_ci, bootstrap_ci)
from edges import blank  # noqa: E402

PAIRS = [("realgan", "realblank"), ("realgan", "classical"),
         ("realblank", "classical"), ("baseline", "classical")]


def g3_ckpts():
    out = {"realblank": [], "realgan": []}
    for p in sorted((ROOT / "models/masked_ssl").glob("g3real_blank_*_seed*_best.pth")):
        arm = "realgan" if "_merged2_" in p.name else "realblank"
        out[arm].append((int(re.search(r"_seed(\d+)_", p.name).group(1)), p))
    for arm, v in out.items():
        seeds = [s for s, _ in v]
        assert len(seeds) == len(set(seeds)), f"duplicate seeds in {arm}: {seeds}"
    out["baseline"] = [(s, p) for o, s, p in find_ckpts() if o == "masking"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results/g3_augment/g3_augment_eval.json")
    args = ap.parse_args()
    from linear_probe_frozen_embeddings import embed_masking

    C = g3_ckpts()
    print({a: [s for s, _ in v] for a, v in C.items()}, flush=True)
    results = []
    for tid, cohort, labfn in target_defs():
        M, ppm, y = load_cohort(cohort, labfn)
        y = np.asarray(y)
        seed = zlib.crc32(tid.encode()) % 2**31
        feats = {"classical": [binned(M, ppm, 0.02)]}
        Mb = blank(M, ppm)
        for arm, v in C.items():
            feats[arm] = [np.asarray(embed_masking(str(p), Mb if arm != "baseline" else M,
                                                   args.device), dtype=np.float32) for _, p in v]
        print(f"\n{tid}  n={len(y)}", flush=True)

        def emit(design, k, nspl, scores, cis):
            for a, b in PAIRS:
                d, lo, hi = cis(a, b)
                results.append({"target": tid, "design": design, "k_per_class": k,
                                "n_splits": nspl, "a": a, "b": b,
                                "acc_a": scores[a], "acc_b": scores[b],
                                "diff": d, "lo": lo, "hi": hi})
                print(f"  {design:<8} {str(k):<4} {a:>9} - {b:<9} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                      flush=True)

        for k in BUDGETS:
            eps = exact_episodes(y, k, args.episodes, seed)
            if eps is None:
                continue
            per = {a: [np.mean([balanced_accuracy_score(y[q], fit_predict(F, y, s, q)) for F in Fs])
                       for s, q in eps] for a, Fs in feats.items()}
            emit("few-shot", k, len(eps), {a: float(np.mean(v)) for a, v in per.items()},
                 lambda a, b: paired_ci(per[a], per[b]))

        design, reps = cv_splits(y, seed)

        def run(F):
            out = []
            for folds in reps:
                pred = np.empty(len(y), dtype=y.dtype)
                for tr, te in folds:
                    pred[te] = fit_predict(F, y, tr, te)
                out.append(pred)
            return out
        preds = {a: [run(F) for F in Fs] for a, Fs in feats.items()}
        if design == "loocv":
            def vote(ps):
                st = np.stack([p[0] for p in ps])
                return np.array([max(set(c), key=list(c).count) for c in st.T])
            V = {a: vote(ps) for a, ps in preds.items()}
            sc = {a: float(np.mean([balanced_accuracy_score(y, p[0]) for p in ps]))
                  for a, ps in preds.items()}
            emit(design, None, len(y), sc, lambda a, b: bootstrap_ci(y, V[a], V[b]))
        else:
            per = {a: [np.mean([balanced_accuracy_score(y, p[r]) for p in ps])
                       for r in range(len(reps))] for a, ps in preds.items()}
            emit(design, None, len(reps), {a: float(np.mean(v)) for a, v in per.items()},
                 lambda a, b: paired_ci(per[a], per[b]))
        json.dump(results, open(ROOT / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
