#!/usr/bin/env python3
"""Realism test with stratified (not study-grouped) cross-validation.

validate_synthetic.py groups its folds by study. That is right for a generator
that is far from the data, but it breaks for one that reproduces each study
closely: synthetic copies of study A sit in the training folds labelled
"synthetic" while study A's real spectra sit in the test fold, so the classifier
calls them synthetic and the AUC falls BELOW 0.5 (0.24 for the diffusion model;
0.12 for autoencoder reconstructions). Here real and synthetic spectra are
shuffled across folds, so 0.5 again means indistinguishable. Real spectra are a
fresh random draw, disjoint from the 600 used for checkpoint selection.
Same features (0.02 ppm bins) and classifier as validate_synthetic.py.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/synthesis"))
from validate_synthetic import binned  # noqa: E402
from edges import blank  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--synth", nargs="+", required=True)
ap.add_argument("--n", type=int, default=1500); ap.add_argument("--out", default="results/final/realism_stratified.json")
a = ap.parse_args()
ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
sel = set(np.random.default_rng(12345).choice(X.shape[0], 600, replace=False).tolist())
pool = np.array([i for i in range(X.shape[0]) if i not in sel])
ri = np.sort(np.random.default_rng(99).choice(pool, a.n, replace=False))
R = np.asarray(X[ri], dtype=np.float32)


def auc(Rb, Sb):
    Z = np.vstack([Rb, Sb]); y = np.r_[np.zeros(len(Rb)), np.ones(len(Sb))]; v = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Z[tr]), y[tr])
        v.append(roc_auc_score(y[te], m.predict_proba(sc.transform(Z[te]))[:, 1]))
    return float(np.mean(v)), float(np.std(v))


out = {}
for s in a.synth:
    S = np.load(ROOT / s, mmap_mode="r")
    S = np.asarray(S[np.sort(np.random.default_rng(1).choice(len(S), min(a.n, len(S)), replace=False))], dtype=np.float32)
    r = {"auc": auc(binned(R, ppm)[0], binned(S, ppm)[0]),
         "auc_edges_blanked": auc(binned(blank(R, ppm), ppm)[0], binned(blank(S, ppm), ppm)[0])}
    out[Path(s).stem] = r
    print(f"{Path(s).stem:<24} AUC {r['auc'][0]:.3f} (+/-{r['auc'][1]:.3f})   edges blanked "
          f"{r['auc_edges_blanked'][0]:.3f}", flush=True)
json.dump(out, open(ROOT / a.out, "w"), indent=1)
