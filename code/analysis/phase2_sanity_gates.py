#!/usr/bin/env python3
"""Phase 2 sanity gates -- run BEFORE any checkpoint is used downstream.

Two ways a pretrained backbone can look healthy and be useless:

  COLLAPSE   the embedding occupies a handful of directions, so every spectrum
             maps to nearly the same vector. Reconstruction loss can still fall.

  STUDY CODE the embedding encodes which study a spectrum came from. This scores
             beautifully on any within-corpus diagnostic and fails transfer
             completely, because the evaluation cohorts are studies the model has
             never seen. With 51% of our corpus from MTBLS798 and a within-study
             nearest-neighbour correlation of 0.989, this is the live risk here.

Both are measured on the HELD-OUT studies -- seven studies withheld from
pretraining -- not on training data.

  effective rank   exp(entropy of the normalised eigenvalue spectrum). The
                   meaningful reference is NOT the embedding dimension but the
                   INTRINSIC rank of the data itself, measured the same way on
                   the same held-out spectra:

                       raw spectra (116,716 pts in-band)   15.3
                       binned 0.02 ppm (classical input)   10.5

                   An embedding below ~10 has discarded structure the classical
                   pipeline keeps; one near 15 has preserved what is there.
                   Judging against the embedding dimension instead makes every
                   representation look collapsed, including the data.
  study probe      balanced accuracy of a logistic regression predicting which
                   held-out study a spectrum came from, 5-fold. Chance is 1/7 =
                   0.143. NOTE: the raw spectra themselves probe at 0.989 and a
                   random-init backbone at 0.973-0.988, so study identity is a
                   property of this data, not something pretraining introduces.
                   The gate cannot condemn a checkpoint on its own -- it is
                   reported because trivially decodable study identity is a
                   standing confound risk for every downstream result.

Neither gate has a pass/fail threshold baked in: they are reported alongside the
random-init control, which is the only meaningful reference. An untrained
backbone already separates studies to some degree, because spectra from one
instrument genuinely resemble each other.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))


def effective_rank(E):
    X = E - E.mean(0)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def study_probe(E, studies, seed=0):
    y = np.asarray(studies)
    keep = np.isin(y, [s for s in np.unique(y) if (y == s).sum() >= 5])
    E, y = E[keep], y[keep]
    if len(np.unique(y)) < 2:
        return float("nan")
    Z = StandardScaler().fit_transform(E)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    preds = np.empty(len(y), dtype=object)
    for tr, te in cv.split(Z, y):
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Z[tr], y[tr])
        preds[te] = m.predict(Z[te])
    return float(balanced_accuracy_score(y, preds))


def find_ckpts():
    """(objective, seed, path) for every completed Phase 2 run."""
    out = []
    for p in sorted((ROOT / "models/masked_ssl").glob("corpus_train_*_seed*_best.pth")):
        seed = int(p.name.split("_seed")[-1].split("_")[0])
        out.append(("masking", seed, p))
    for obj in ("jigsaw", "joint"):
        for d in sorted((ROOT / "models/phase2").glob(f"{obj}_seed*")):
            seed = int(d.name.split("seed")[-1])
            best = sorted(d.rglob("*_best.pth"))
            if best:
                out.append((obj, seed, best[-1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="results/phase2/sanity_gates.json")
    ap.add_argument("--random-init", action="store_true",
                    help="also embed with an untrained backbone as the control")
    args = ap.parse_args()

    from linear_probe_frozen_embeddings import embed_masking, embed_jigsaw, embed_joint
    EMB = {"masking": embed_masking, "jigsaw": embed_jigsaw, "joint": embed_joint}

    X = np.load(ROOT / "rebuild/pretrain/corpus_heldout.npy")
    split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "heldout"]
    studies = [r["study"] for r in split]
    assert len(studies) == X.shape[0]
    print(f"held-out: {X.shape[0]} spectra, {len(set(studies))} studies\n")
    print("reference effective rank on the same spectra: raw 15.3, binned-0.02 10.5\n")
    print(f"{'objective':<10}{'seed':>5}{'dim':>7}{'eff.rank':>10}{'vs raw':>9}{'study probe':>13}")

    rows = []
    for obj, seed, ck in find_ckpts():
        try:
            E = EMB[obj](str(ck), X, args.device)
            E = np.asarray(E, dtype=np.float64)
            er = effective_rank(E)
            sp = study_probe(E, studies)
            rows.append({"objective": obj, "seed": seed, "checkpoint": str(ck),
                         "dim": int(E.shape[1]), "effective_rank": er,
                         "eff_rank_frac": er / E.shape[1], "study_probe_balacc": sp,
                         "random_init": False})
            print(f"{obj:<10}{seed:>5}{E.shape[1]:>7}{er:>10.1f}"
                  f"{er / 15.3:>9.2f}{sp:>13.3f}", flush=True)
        except Exception as e:
            print(f"{obj:<10}{seed:>5}   FAILED: {type(e).__name__}: {e}", flush=True)

    if args.random_init:
        print("\nrandom-init controls:")
        for obj in ("masking", "jigsaw", "joint"):
            ck = next((c for o, s, c in find_ckpts() if o == obj), None)
            if ck is None:
                continue
            E = np.asarray(EMB[obj](str(ck), X, args.device, random_init=True), dtype=np.float64)
            er, sp = effective_rank(E), study_probe(E, studies)
            rows.append({"objective": obj, "seed": -1, "checkpoint": str(ck),
                         "dim": int(E.shape[1]), "effective_rank": er,
                         "eff_rank_frac": er / E.shape[1], "study_probe_balacc": sp,
                         "random_init": True})
            print(f"{obj:<10}{'rnd':>5}{E.shape[1]:>7}{er:>10.1f}"
                  f"{er / 15.3:>9.2f}{sp:>13.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
