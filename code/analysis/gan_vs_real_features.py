#!/usr/bin/env python3
"""What distinguishes GAN-generated spectra from real ones? (PI request, 2026-09-25)

A discriminator AUC (0.832 for GAN v2) says the two CAN be told apart, not WHERE.
This answers "where" with transparent tools instead of a neural classifier:

  1. PER-WINDOW TEST. 468 windows of 0.02 ppm, area-normalised exactly as the
     5.2 gate does. Mann-Whitney U in every window, Benjamini-Hochberg FDR, and
     Cliff's delta as the effect size (-1..+1; +ve = synthetic higher). Delta,
     not p, decides what matters: with thousands of spectra per side almost every
     window is "significant".
  2. PCA. Fitted on REAL spectra only, so the components describe genuine
     variation. Synthetic spectra are projected in; each component's real-vs-
     synthetic separation is measured with Cliff's delta, and the loadings of the
     most separating component say which ppm regions drive it.
  3. WHOLE-SPECTRUM PROPERTIES that 0.02 ppm binning averages away: noise level
     in signal-free regions, baseline offset, high-frequency texture, and the
     width of the creatine CH3 singlet.

CALIBRATION. Every test is also run on two random halves of the REAL data. That
null tells us how large a delta arises between two samples of genuinely the same
distribution, so a synthetic effect is only reported as a real difference when it
clearly exceeds it.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/synthesis"))
from validate_synthetic import binned  # noqa: E402  (identical binning to the gate)


def cliffs_delta(a, b):
    """P(b > a) - P(b < a), via U. +ve means b (synthetic) tends to be larger."""
    u = mannwhitneyu(b, a, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def window_tests(A, B):
    d = np.array([cliffs_delta(A[:, j], B[:, j]) for j in range(A.shape[1])])
    p = np.array([mannwhitneyu(A[:, j], B[:, j]).pvalue for j in range(A.shape[1])])
    return d, bh(p)


def fwhm(y, ppm, lo, hi):
    m = np.where((ppm <= hi) & (ppm >= lo))[0]
    seg = y[m]; i = np.argmax(seg); h = seg[i]
    if h <= 0: return np.nan
    half = h / 2; l = i
    while l > 0 and seg[l] > half: l -= 1
    r = i
    while r < len(seg) - 1 and seg[r] > half: r += 1
    return abs(ppm[m[l]] - ppm[m[r]])


def properties(S, ppm):
    q = (ppm > 9.5) | (ppm < -0.3)
    sig = (ppm < 9.0) & (ppm > 0.5)
    out = {"noise_sd": [], "baseline": [], "hf_ratio": [], "creatine_fwhm_ppm": []}
    for y in S:
        y = np.asarray(y, dtype=np.float64)
        out["noise_sd"].append(y[q].std() / (np.abs(y).max() + 1e-12))
        out["baseline"].append(np.median(y[sig]) / (np.abs(y).max() + 1e-12))
        dy = np.diff(y[sig])
        out["hf_ratio"].append(np.sum(dy ** 2) / (np.sum(y[sig] ** 2) + 1e-12))
        out["creatine_fwhm_ppm"].append(fwhm(y, ppm, 2.98, 3.08))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="rebuild/pretrain/synthetic/gan_v2.npy")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/phase5/gan_vs_real_features.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    ppm = np.load(ROOT / "rebuild/pretrain/ppm_axis.npy")
    X = np.load(ROOT / "rebuild/pretrain/corpus_train.npy", mmap_mode="r")
    idx = np.sort(rng.choice(X.shape[0], 2 * args.n, replace=False))
    R = np.asarray(X[idx], dtype=np.float32)
    R1, R2 = R[0::2], R[1::2]                     # real-vs-real null
    S = np.load(ROOT / args.synth, mmap_mode="r")
    S = np.asarray(S[np.sort(rng.choice(S.shape[0], min(args.n, S.shape[0]), replace=False))],
                   dtype=np.float32)
    print(f"real A {R1.shape[0]}, real B {R2.shape[0]}, synthetic {S.shape[0]}\n")

    B1, centres = binned(R1, ppm); B2, _ = binned(R2, ppm); BS, _ = binned(S, ppm)

    # ---- 1. per-window tests ----------------------------------------------
    d_syn, q_syn = window_tests(B1, BS)
    d_nul, q_nul = window_tests(B1, B2)
    thr = float(np.percentile(np.abs(d_nul), 99))
    print("1. PER-WINDOW (468 windows of 0.02 ppm)")
    print(f"   real-vs-real null:  |delta| median {np.median(np.abs(d_nul)):.3f}, "
          f"99th pct {thr:.3f};  'significant' (q<0.05) {np.mean(q_nul<0.05):.1%}")
    print(f"   real-vs-synthetic:  |delta| median {np.median(np.abs(d_syn)):.3f};  "
          f"'significant' {np.mean(q_syn<0.05):.1%};  beyond null 99th pct "
          f"{np.mean(np.abs(d_syn)>thr):.1%}")
    big = np.where(np.abs(d_syn) > 0.5)[0]
    print(f"   windows with a LARGE effect (|delta| > 0.5): {len(big)}")
    print(f"     synthetic higher: {np.sum(d_syn[big] > 0)}   synthetic lower: {np.sum(d_syn[big] < 0)}")
    print("   strongest 12 windows:")
    for j in np.argsort(-np.abs(d_syn))[:12]:
        print(f"     {centres[j]:+6.2f} ppm   delta {d_syn[j]:+.3f}   real median "
              f"{np.median(B1[:,j]):.2e}  synth median {np.median(BS[:,j]):.2e}")

    # ---- 2. PCA on real ---------------------------------------------------
    sc = StandardScaler().fit(B1)
    pca = PCA(n_components=20, random_state=0).fit(sc.transform(B1))
    Z1, Z2, ZS = (pca.transform(sc.transform(M)) for M in (B1, B2, BS))
    print("\n2. PCA (fitted on real only)")
    pcs = []
    for c in range(10):
        dn = cliffs_delta(Z1[:, c], Z2[:, c]); ds = cliffs_delta(Z1[:, c], ZS[:, c])
        pcs.append({"pc": c + 1, "var": float(pca.explained_variance_ratio_[c]),
                    "delta_syn": float(ds), "delta_null": float(dn)})
        print(f"   PC{c+1:<2} {pca.explained_variance_ratio_[c]:6.1%} of real variance   "
              f"delta synthetic {ds:+.3f}   (null {dn:+.3f})")
    # distance from the real cloud: Mahalanobis in the top 10 PCs
    var = pca.explained_variance_[:10]
    m1 = np.sqrt(((Z1[:, :10] ** 2) / var).sum(1))
    ms = np.sqrt(((ZS[:, :10] ** 2) / var).sum(1))
    m2 = np.sqrt(((Z2[:, :10] ** 2) / var).sum(1))
    env = np.percentile(m1, 95)
    print(f"   outside the real 95% envelope (top-10 PCs): real-B {np.mean(m2>env):.1%}, "
          f"synthetic {np.mean(ms>env):.1%}")
    # unseen variation: how much of each spectrum the real PCs fail to reconstruct
    def resid(M):
        Zt = sc.transform(M); rec = pca.inverse_transform(pca.transform(Zt))
        return ((Zt - rec) ** 2).sum(1) / (Zt ** 2).sum(1)
    r2, rs = resid(B2), resid(BS)
    print(f"   variance NOT captured by the 20 real PCs: real-B median {np.median(r2):.1%}, "
          f"synthetic {np.median(rs):.1%}")
    top = max(pcs, key=lambda r: abs(r["delta_syn"]))
    load = pca.components_[top["pc"] - 1] / sc.scale_
    print(f"   most separating component PC{top['pc']}; its strongest loadings:")
    for j in np.argsort(-np.abs(load))[:6]:
        print(f"     {centres[j]:+6.2f} ppm   loading {load[j]:+.2e}")

    # ---- 3. whole-spectrum properties ------------------------------------
    P1, P2, PS = properties(R1[:600], ppm), properties(R2[:600], ppm), properties(S[:600], ppm)
    print("\n3. WHOLE-SPECTRUM PROPERTIES (600 spectra each)")
    props = {}
    for k in P1:
        a, b, c = P1[k], P2[k], PS[k]
        ok = ~np.isnan(a); okb = ~np.isnan(b); okc = ~np.isnan(c)
        ds = cliffs_delta(a[ok], c[okc]); dn = cliffs_delta(a[ok], b[okb])
        props[k] = {"real_median": float(np.median(a[ok])), "synth_median": float(np.median(c[okc])),
                    "delta_syn": float(ds), "delta_null": float(dn)}
        print(f"   {k:<18} real {np.median(a[ok]):.3e}   synthetic {np.median(c[okc]):.3e}   "
              f"delta {ds:+.3f}  (null {dn:+.3f})")

    json.dump({"synth": args.synth, "centres": centres.tolist(),
               "delta_syn": d_syn.tolist(), "delta_null": d_nul.tolist(),
               "q_syn": q_syn.tolist(), "null_99": thr, "pcs": pcs,
               "pca_top": top["pc"], "pca_top_loading": load.tolist(),
               "envelope_out": {"real": float(np.mean(m2 > env)), "synth": float(np.mean(ms > env))},
               "unexplained": {"real": float(np.median(r2)), "synth": float(np.median(rs))},
               "properties": props, "real_median_binned": np.median(B1, 0).tolist(),
               "pc_scores": {"real": Z1[:, :2].tolist(), "synth": ZS[:, :2].tolist()}},
              open(ROOT / args.out, "w"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
