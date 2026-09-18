#!/usr/bin/env python3
"""Phase 3 -- gate G1: does pretraining beat the classical bar?

Whiteboard node: "ML is better - sets bar, can we better it".

The claim under test is NOT "SSL wins at full cohort size". It is the project's
actual premise: real clinical NMR tasks have 20-150 labelled samples, so the
question is whether corpus-scale pretraining helps *when labels are scarce*.
A single full-data number cannot answer that, so every target is evaluated on a
label-budget curve and compared against classical ML fitted on the same episodes.

Design decisions that matter:

* Frozen embeddings, not fine-tuning, for this first pass. Fine-tuning a 192-d
  backbone on 20 samples has enormous variance, and mixing it in here would make
  the comparison about optimisation rather than representation.
* The classical baseline is refit inside every episode on the SAME support set.
  Comparing a few-shot SSL number against Phase 1's full-data classical number
  would be meaningless.
* A random-init backbone of identical architecture is run as a control, so
  "pretraining helped" is separable from "this architecture's inductive bias
  helped".
* Every checkpoint seed is evaluated on the SAME episodes, so the SSL-vs-classical
  difference is paired and the seed variance is visible rather than averaged away.

Emits one JSON row per (target, representation, k, episode) so all aggregation
happens downstream and nothing is baked in.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))

COH = ROOT / "rebuild/cohorts"
WATER = (4.45, 5.10)
BUDGETS = [5, 10, 20, 40, 0]          # 0 = all available labels
N_EPISODES = 50


# --------------------------------------------------------------- targets ---
def target_defs():
    def m563_bin(r):
        return None if r["label"] == "unknown" else (
            "control" if r["label"] == "surgery (control)" else "infection")

    def m563_3(r):
        return None if r["label"] == "unknown" else r["label"]

    def barth(r):
        return None if r["label"] == "Pool" else r["label"]

    return [
        ("mtbls563_binary", "mtbls563", m563_bin),
        ("mtbls563_3class", "mtbls563", m563_3),
        ("brc_t2d_cancer", "brc_t2d", lambda r: r["Group"].split("-")[0]),
        ("brc_t2d_diabetes", "brc_t2d", lambda r: r["Group"].split("-")[1]),
        ("brc_t2d_4class", "brc_t2d", lambda r: r["Group"]),
        ("barth_case_control", "barth", barth),
    ]


def load_cohort(name, labfn):
    X = np.load(COH / f"{name}_spectra.npy", mmap_mode="r")
    ppm = np.load(COH / f"{name}_ppm_axis.npy")
    rows = list(csv.DictReader(open(COH / f"{name}_labels.csv")))
    idx, y = [], []
    for r in rows:
        lab = labfn(r)
        if lab:
            idx.append(int(r["row"])); y.append(lab)
    return np.array(X[idx], dtype=np.float32), ppm, np.asarray(y)


def binned(M, ppm, width=0.02):
    keep = ~((ppm <= WATER[1]) & (ppm >= WATER[0]))
    keep &= (ppm >= -0.5) & (ppm <= 9.5)
    edges = np.arange(ppm[keep].min(), ppm[keep].max(), width)
    idx = np.digitize(ppm, edges)
    out = np.zeros((M.shape[0], len(edges)), dtype=np.float32)
    for b in range(len(edges)):
        s = (idx == b) & keep
        if s.any():
            out[:, b] = M[:, s].mean(axis=1)
    a = np.abs(out).sum(axis=1, keepdims=True); a[a == 0] = 1
    return out / a


# -------------------------------------------------------------- episodes ---
def episodes(y, k, n_ep, seed=0):
    """n_ep stratified support/query splits. k per class; k=0 means 80/20."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    out = []
    for _ in range(n_ep):
        sup = []
        for c in classes:
            pool = np.where(y == c)[0]
            take = max(1, int(round(0.8 * len(pool)))) if k == 0 else min(k, len(pool) - 1)
            sup += list(rng.choice(pool, size=take, replace=False))
        sup = np.array(sorted(sup))
        qry = np.setdiff1d(np.arange(len(y)), sup)
        if len(np.unique(y[sup])) < 2 or len(qry) == 0:
            continue
        out.append((sup, qry))
    return out


def probe(F, y, sup, qry):
    Z = StandardScaler().fit(F[sup])
    m = LogisticRegression(max_iter=3000, class_weight="balanced")
    m.fit(Z.transform(F[sup]), y[sup])
    return balanced_accuracy_score(y[qry], m.predict(Z.transform(F[qry])))


def find_ckpts():
    out = []
    for p in sorted((ROOT / "models/masked_ssl").glob("corpus_train_*_seed*_best.pth")):
        out.append(("masking", int(p.name.split("_seed")[-1].split("_")[0]), p))
    for obj in ("jigsaw", "joint"):
        for d in sorted((ROOT / "models/phase2").glob(f"{obj}_seed*")):
            best = sorted(d.rglob("*_best.pth"))
            if best:
                out.append((obj, int(d.name.split("seed")[-1]), best[-1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--objectives", nargs="+", default=["masking", "jigsaw", "joint"])
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--out", default="results/phase3/transfer.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from linear_probe_frozen_embeddings import embed_masking, embed_jigsaw, embed_joint
    EMB = {"masking": embed_masking, "jigsaw": embed_jigsaw, "joint": embed_joint}

    ckpts = [c for c in find_ckpts() if c[0] in args.objectives]
    print(f"{len(ckpts)} checkpoints, {len(target_defs())} targets, "
          f"budgets {args.budgets}, {args.episodes} episodes each\n")

    rows = []
    for tid, cohort, labfn in target_defs():
        M, ppm, y = load_cohort(cohort, labfn)
        reps = {"classical_binned0.02": binned(M, ppm, 0.02)}

        for obj, seed, ck in ckpts:
            try:
                reps[f"{obj}_seed{seed}"] = np.asarray(
                    EMB[obj](str(ck), M, args.device), dtype=np.float32)
            except Exception as e:
                print(f"  embed FAILED {obj} seed{seed} on {tid}: "
                      f"{type(e).__name__}: {e}", flush=True)
        if args.random_init:
            for obj in args.objectives:
                ck = next((c for o, s, c in ckpts if o == obj), None)
                if ck is not None:
                    reps[f"{obj}_randinit"] = np.asarray(
                        EMB[obj](str(ck), M, args.device, random_init=True), dtype=np.float32)

        for k in args.budgets:
            eps = episodes(y, k, args.episodes, seed=hash(tid) % 2**31)
            if not eps:
                continue
            for rname, F in reps.items():
                sc = [probe(F, y, s, q) for s, q in eps]
                v = np.asarray(sc)
                rows.append({"target": tid, "cohort": cohort, "representation": rname,
                             "k_per_class": k, "n_total": int(len(y)),
                             "n_episodes": len(sc), "dim": int(F.shape[1]),
                             "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                             "ci95_lo": float(np.percentile(v, 2.5)),
                             "ci95_hi": float(np.percentile(v, 97.5)),
                             "scores": [float(x) for x in v]})
            base = next(r for r in rows if r["target"] == tid and r["k_per_class"] == k
                        and r["representation"] == "classical_binned0.02")
            best = max((r for r in rows if r["target"] == tid and r["k_per_class"] == k
                        and r["representation"] != "classical_binned0.02"),
                       key=lambda r: r["mean"], default=None)
            kl = "all" if k == 0 else str(k)
            msg = (f"{tid:<20} k={kl:<4} classical {base['mean']:.3f}")
            if best:
                msg += (f"   best SSL {best['representation']:<18} {best['mean']:.3f}"
                        f"   delta {best['mean'] - base['mean']:+.3f}")
            print(msg, flush=True)
            json.dump(rows, open(args.out, "w"), indent=1)
        print(flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
