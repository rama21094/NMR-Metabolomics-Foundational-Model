#!/usr/bin/env python3
"""Separability (AUC) of GAN v2 vs real, by region, over the FULL axis.

The earlier region test (gan_auc_by_region.json) binned only up to 9.5 ppm and
lumped the TSP neighbourhood (|ppm| < 0.2) in with the edges. This one keeps
the far edges, the TSP area and the metabolite region apart. Same classifier
and study-grouped CV as validate_synthetic.py.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
WATER = (4.45, 5.10)
rng = np.random.default_rng(0)
ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
st = np.array([r["study"] for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
               if r["split"] == "train"])
ri = np.sort(rng.choice(X.shape[0], 1500, replace=False))
R = np.asarray(X[ri], dtype=np.float32)
S = np.asarray(np.load(ROOT / "rebuild/pretrain/synthetic/gan_v2.npy", mmap_mode="r"), dtype=np.float32)

keep = ~((ppm <= WATER[1]) & (ppm >= WATER[0]))
edges = np.arange(-0.5, 10.0 + 1e-9, 0.02); which = np.digitize(ppm, edges)
bins = [b for b in np.unique(which[keep])]
cen = np.array([ppm[keep & (which == b)].mean() for b in bins])
def binm(M):
    return np.stack([M[:, keep & (which == b)].mean(1) for b in bins], 1)
Rb, Sb = binm(R), binm(S)

def auc(cols):
    Z = np.vstack([Rb[:, cols], Sb[:, cols]])
    Z = Z / np.maximum(np.abs(Z).sum(1, keepdims=True), 1e-12)
    y = np.r_[np.zeros(len(Rb)), np.ones(len(Sb))]
    g = np.r_[st[ri], [f"S{i%5}" for i in range(len(Sb))]]
    a = []
    for tr, te in GroupKFold(5).split(Z, y, g):
        if len(np.unique(y[te])) < 2:      # fold with one class only, as validate_synthetic skips
            continue
        sc = StandardScaler().fit(Z[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Z[tr]), y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(sc.transform(Z[te]))[:, 1]))
    return float(np.mean(a))

regions = {
    "whole spectrum": np.ones(len(cen), bool),
    "far edges only (<-0.2 or >9.4)": (cen < -0.2) | (cen > 9.4),
    "TSP area only (-0.2 to 0.2)": (cen >= -0.2) & (cen < 0.2),
    "metabolite region only (0.2 to 9.4)": (cen >= 0.2) & (cen <= 9.4),
}
out = {k: auc(m) for k, m in regions.items()}
for k, v in out.items():
    print(f"{k:<40} AUC {v:.3f}")
json.dump(out, open(ROOT / "results/phase5/gan_auc_by_region_full.json", "w"), indent=1)
