#!/usr/bin/env python3
"""Final experiment 3 (Shyam meeting 2026-10-23): do different seeds converge
to the same model?

Raw weights cannot answer this: hidden units can be permuted (and attention
heads reordered) without changing the function, so two identical solutions can
have uncorrelated weight matrices. We therefore report both:

1. RAW last-layer weight correlation between seeds, after the best one-to-one
   matching of output units (Hungarian, on |correlation|). Shown to make the
   point; low values are expected even for equivalent models.
2. LINEAR CKA between seeds' embeddings of the same spectra (Kornblith et al.
   2019), which is invariant to rotations and permutations of the embedding
   space. Calibrated against the same network with random weights (5 random
   initialisations), so "high" has a reference.

Spectra: the 935 held-out corpus spectra (7 studies never used in pretraining).
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/training", "code/evaluation", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
from phase3_transfer import find_ckpts  # noqa: E402
from linear_probe_frozen_embeddings import embed_masking  # noqa: E402


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X, "fro") ** 2 /
                 (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def last_layer(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    # Last block's first feed-forward matrix, (hidden 768 x width 192): its rows are
    # hidden units, the units that can be freely permuted between seeds.
    keys = [k for k in sd if k.endswith("linear1.weight") and ".layers." in k]
    last = max(int(k.split(".layers.")[1].split(".")[0]) for k in keys)
    k = [k for k in keys if f".layers.{last}." in k][0]
    return k, sd[k].numpy()


def matched_corr(A, B):
    C = np.corrcoef(A, B)[:len(A), len(A):]
    r, c = linear_sum_assignment(-np.abs(C))
    return float(np.mean(np.abs(C[r, c])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="results/final/seed_similarity.json")
    a = ap.parse_args()
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    S = np.load(ROOT / "rebuild/pretrain/corpus_heldout.npy").astype(np.float32)
    C = [(s, p) for o, s, p in find_ckpts() if o == "masking"]
    out = {"n_spectra": int(len(S)), "seeds": [s for s, _ in C]}

    k, _ = last_layer(C[0][1]); out["weight_tensor"] = k
    W = {s: last_layer(p)[1] for s, p in C}
    out["weight_matched_abs_corr"] = {f"{i}-{j}": matched_corr(W[i], W[j])
                                      for i, j in itertools.combinations(W, 2)}
    rng = np.random.default_rng(0)
    shp = next(iter(W.values())).shape
    out["weight_matched_abs_corr_random_baseline"] = float(np.mean(
        [matched_corr(rng.standard_normal(shp), rng.standard_normal(shp)) for _ in range(3)]))
    print("last-layer weights", k, "matched |r|:", out["weight_matched_abs_corr"],
          "random-matrix baseline", out["weight_matched_abs_corr_random_baseline"], flush=True)

    for pool in ("mean_pool", "regional:16"):
        E = {s: np.asarray(embed_masking(str(p), S, a.device, pooling=pool)) for s, p in C}
        R = {i: np.asarray(embed_masking(str(C[0][1]), S, a.device, pooling=pool,
                                         random_init=True, seed=100 + i)) for i in range(5)}
        # Raw embeddings are dominated by one shared direction (overall intensity),
        # which makes linear CKA high for ANY two networks, random ones included.
        # The probe standardises each feature before fitting, so CKA on z-scored
        # features is the comparison that matches what the classifier sees.
        z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-8)
        for tag, f in (("raw", lambda X: X), ("standardised", z)):
            tr = [cka(f(E[i]), f(E[j])) for i, j in itertools.combinations(E, 2)]
            rr = [cka(f(R[i]), f(R[j])) for i, j in itertools.combinations(R, 2)]
            tx = [cka(f(E[i]), f(R[j])) for i in E for j in R]
            out[f"{pool}|{tag}"] = {"cka_trained_vs_trained": tr, "cka_random_vs_random": rr,
                                    "cka_trained_vs_random": tx}
            print(f"{pool} {tag}: CKA trained-trained {np.mean(tr):.3f} (min {np.min(tr):.3f}); "
                  f"random-random {np.mean(rr):.3f}; trained-random {np.mean(tx):.3f}", flush=True)
    json.dump(out, open(ROOT / a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
