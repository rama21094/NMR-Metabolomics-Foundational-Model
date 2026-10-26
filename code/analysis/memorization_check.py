#!/usr/bin/env python3
"""Does a generator copy its training spectra? (Somepalli et al. 2023 style.)

For each generated spectrum: its highest Pearson correlation with any training
spectrum (binned 0.02 ppm, same features as the realism test). Reference: the
same statistic for real held-out spectra (studies never used in training), which
shows how close a genuinely new spectrum normally comes to the training set.
A generator that memorises shows a distribution shifted to ~1.0.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/synthesis"))
from validate_synthetic import binned  # noqa: E402


def zrows(B):
    B = B - B.mean(1, keepdims=True)
    return B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)


def nn_corr(Q, Tz):
    return np.concatenate([(zrows(Q[i:i + 500]) @ Tz.T).max(1) for i in range(0, len(Q), 500)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--out", default="results/final/memorization.json"); a = ap.parse_args()
    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    T = np.vstack([binned(np.asarray(X[i:i + 500], dtype=np.float32), ppm)[0] for i in range(0, len(X), 500)])
    Tz = zrows(T)
    H = binned(np.load(ROOT / "rebuild/pretrain/corpus_heldout.npy").astype(np.float32), ppm)[0]
    out = {}
    def summ(name, v):
        out[name] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                     "share_above_0.999": float(np.mean(v > 0.999))}
        print(f"{name:<34} median {out[name]['median']:.4f}  p95 {out[name]['p95']:.4f}  "
              f">0.999: {out[name]['share_above_0.999']:.1%}", flush=True)
    summ("real held-out (reference)", nn_corr(H, Tz))
    # Fairer reference: a real TRAINING spectrum's nearest OTHER training
    # spectrum. Held-out spectra come from unseen studies and so sit further away
    # than a new spectrum from a seen study would.
    q = np.random.default_rng(0).choice(len(Tz), 2000, replace=False)
    C = Tz[q] @ Tz.T; C[np.arange(len(q)), q] = -1
    summ("real train, nearest other (ref.)", C.max(1))
    for s in a.synth:
        S = np.load(ROOT / s, mmap_mode="r")
        B = np.vstack([binned(np.asarray(S[i:i + 500], dtype=np.float32), ppm)[0] for i in range(0, len(S), 500)])
        summ(Path(s).stem, nn_corr(B, Tz))
    json.dump(out, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
