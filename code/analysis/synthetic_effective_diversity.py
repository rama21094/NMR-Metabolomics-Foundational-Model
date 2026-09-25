#!/usr/bin/env python3
"""Does a synthetic set carry more study diversity than the corpus it came from?

This is the measurement that decides whether the generative route can address
gate G2's finding. G2 established that transfer is limited by effective study
count -- exp(entropy of the study composition) -- and that this corpus holds
4.09. A generator that merely resamples what it was shown cannot raise that
number; one that interpolates between studies might.

METHOD. Build a centroid per real study in a binned, normalised space. Assign
each synthetic spectrum to its nearest centroid. The entropy of the resulting
assignment gives the synthetic set's effective study count, on the same scale as
the corpus's own.

WHAT THE NUMBER CANNOT TELL YOU. Nearest-centroid assignment measures how the
synthetic set SPREADS OVER the existing studies, not whether it occupies genuinely
new regions. A generator inventing a thirteenth study would show up here only as
an even spread over the twelve it knows. So a high value is necessary, not
sufficient, evidence of added diversity; a LOW value is conclusive evidence of
mode collapse. It is reported as a falsifier, not a certificate.

WORKED EXAMPLE OF THE TRAP. The GISSMO linear-combination set scores 7.10 here --
apparently far above the corpus's 4.09 -- while failing the 5.2 realism gate at
discriminator AUC 1.000. (An earlier run scored 8.73 on spectra that were pure
noise, from a since-fixed np.interp bug; the trap holds either way.) It resembles no real study, so nearest-centroid
assignment is close to arbitrary and the entropy is high for the worst possible
reason. THIS NUMBER IS ONLY MEANINGFUL FOR A SET THAT HAS ALREADY PASSED 5.2.
Run validate_synthetic.py first; if the discriminator separates the set, stop.
"""
import argparse, csv
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WATER = (4.45, 5.10)


def binned(M, ppm, width=0.02):
    keep = ~((ppm <= WATER[1]) & (ppm >= WATER[0])) & (ppm >= -0.5) & (ppm <= 9.5)
    cols = np.where(keep)[0]; kppm = ppm[cols]
    edges = np.arange(kppm.min(), kppm.max(), width)
    which = np.digitize(kppm, edges); order = np.argsort(which, kind="stable")
    cols_s, which_s = cols[order], which[order]
    labels = np.unique(which_s); starts = np.searchsorted(which_s, labels)
    counts = np.diff(np.append(starts, len(which_s)))
    out = np.add.reduceat(np.asarray(M[:, cols_s], dtype=np.float32), starts, axis=1) / counts
    a = np.abs(out).sum(axis=1, keepdims=True); a[a == 0] = 1
    return out / a


def eff(counts):
    p = np.array(list(counts), float); p = p / p.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--n-real", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    split = [r for r in csv.DictReader(open(ROOT / "rebuild/pretrain/corpus_split.csv"))
             if r["split"] == "train"]
    studies = np.array([r["study"] for r in split])
    idx = np.sort(rng.choice(X.shape[0], min(args.n_real, X.shape[0]), replace=False))
    R = binned(np.asarray(X[idx], dtype=np.float32), ppm)
    rs = studies[idx]

    uniq = sorted(set(rs))
    C = np.vstack([R[rs == s].mean(axis=0) for s in uniq])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12

    print(f"real corpus effective studies: {eff(Counter(studies).values()):.2f} "
          f"(full training set, {len(set(studies))} nominal)")
    # sanity: the same assignment applied to REAL spectra, as a reference point
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    assign_r = np.argmax(Rn @ C.T, axis=1)
    print(f"  same metric on real spectra:  {eff(Counter(assign_r.tolist()).values()):.2f}"
          f"   (assignment accuracy {np.mean([uniq[a] == b for a, b in zip(assign_r, rs)]):.1%})")

    S = binned(np.load(ROOT / args.synth), ppm)
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    assign_s = np.argmax(Sn @ C.T, axis=1)
    cnt = Counter(assign_s.tolist())
    e = eff(cnt.values())
    print(f"\nsynthetic set: {S.shape[0]} spectra, effective studies {e:.2f}")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"    {uniq[k]:<14} {v:>5}  ({v/len(assign_s):.1%})")
    print("\nverdict:")
    if e < 2.0:
        print("  MODE COLLAPSE -- the generated set concentrates on few studies.")
    elif e < 4.09:
        print("  resampling, and narrower than the corpus. Cannot address G2.")
    elif e < 5.0:
        print("  resampling the corpus composition, as predicted. Cannot address G2.")
    else:
        print("  SPREADS WIDER than the corpus. The prediction that a generator")
        print("  cannot raise effective diversity is not supported -- follow up.")


if __name__ == "__main__":
    main()
