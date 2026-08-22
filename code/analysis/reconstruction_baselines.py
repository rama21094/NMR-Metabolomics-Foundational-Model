#!/usr/bin/env python3
"""Experiment #19: is masked reconstruction actually a hard task on this corpus?

Every slide and every earlier note that praised the pretext task ("overall R2
0.98", "r = 0.999", "the pretext task itself works very well") reported the
model's reconstruction quality WITHOUT A BASELINE. That is the same mistake as
reporting a classifier's accuracy without the majority-class rate.

This script supplies the missing baselines. On the SAME spectra with the SAME
masks, it scores four predictors that involve no learned metabolite structure
at all:

  corpus_mean   -- predict the corpus mean spectrum at the masked bins.
                   Tests "all serum CPMG spectra look alike".
  linear_interp -- linearly interpolate the masked bins from the surrounding
                   visible bins. Tests pure local smoothness / lineshape
                   autocorrelation.
  pca{K}        -- least-squares fit of a K-component corpus PCA basis to the
                   VISIBLE bins only, then read off the masked bins. Tests
                   "the corpus is low-rank", using no information about the
                   held-out region beyond the population subspace.
  nn_copy       -- find the most similar OTHER spectrum (by visible bins) and
                   copy its masked bins verbatim. Tests inter-spectrum
                   redundancy. This is the strongest non-learned baseline and
                   the one that matters: if it ties the network, the network's
                   reconstruction skill is not evidence of chemical knowledge.

WHY THE MASK RATIO HAS TO BE MATCHED
------------------------------------
The reconstruction figure in plot_groupmeeting_figures.py hides 25% of patches
(`int(0.25 * npatch)`), while the model was trained at mr 0.20-0.60. Comparing
the model at 25% against baselines at 60% flatters the model by a wide margin;
an earlier pass at this analysis did exactly that and drew the wrong
conclusion. Every predictor here is scored on identical masks, and the whole
sweep is repeated at each ratio in --mask-ratios.

nn_copy AND ROW ADJACENCY
-------------------------
The corpus is ordered by source study, so a test spectrum's nearest neighbour
is often from the same study -- sometimes a near-duplicate. That is a property
of the corpus, not a bug in the baseline, but it inflates nn_copy relative to
what it would score against a genuinely unrelated cohort. We therefore also
report nn_copy restricted to matches whose row index is far from the test row
(--row-gap), as a study-disjointness proxy, and record the near-duplicate rate
separately in the redundancy table.

PEAK vs BASELINE BINS
---------------------
Whole-spectrum correlation is dominated by the many low-intensity bins. Each
masked region is therefore also scored separately over its high-intensity
(peak) bins and its low-intensity (baseline) bins, to check whether the
headline r is carried by predicting emptiness.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for sub in ("code/training", "code/evaluation"):
    sys.path.insert(0, str(ROOT / sub))

# The v3 ps1024 reference checkpoint -- the same one every §4/§5 number and the
# slide-8 reconstruction demo are measured on.
CKPT = ("models/masked_ssl/combine_unique_MetaboLights_Workbench_Water_EDTA_"
        "Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth")
CORPUS = ("data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_"
          "Suppressed_rowMinMax_v4.npy")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def load_model(spectrum_length: int):
    from trainer_revised import NMRMaskedAutoencoder
    from barth_all_models_loocv import infer_mae_config

    state = torch.load(ROOT / CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    # nhead=4 is the value this checkpoint actually trained with. It is not
    # recoverable from tensor shapes, so it must be passed explicitly; see the
    # §4a/§5c nhead note in docs/SSL_vs_classical_analysis.md.
    model = NMRMaskedAutoencoder(spectrum_length=spectrum_length,
                                 **infer_mae_config(state, 4, 0.0))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def corpus_redundancy(corpus, n_sample: int, n_bins: int, seed: int) -> pd.DataFrame:
    """How self-similar is the corpus? PCA rank + near-duplicate rate."""
    rng = np.random.default_rng(seed)
    n, length = corpus.shape
    idx = np.sort(rng.choice(n, min(n_sample, n), replace=False))
    fold = length // n_bins
    binned = np.asarray(corpus[idx], dtype=np.float32)[:, :n_bins * fold]
    binned = binned.reshape(len(idx), n_bins, fold).mean(axis=2)

    centred = binned - binned.mean(0)
    sv = np.linalg.svd(centred, compute_uv=False)
    frac = np.cumsum(sv ** 2) / (sv ** 2).sum()

    z = binned - binned.mean(1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    corr = z @ z.T
    np.fill_diagonal(corr, -1.0)
    best = corr.max(1)

    rows = [dict(metric=f"pca_cum_var_{k}pc", value=float(frac[k - 1]))
            for k in (1, 2, 5, 10, 20, 50, 100) if k <= len(frac)]
    rows += [dict(metric=f"frac_rows_with_neighbour_r_gt_{t}", value=float((best > t).mean()))
             for t in (0.9999, 0.999, 0.99, 0.95)]
    rows.append(dict(metric="median_best_match_r", value=float(np.median(best))))
    rows.append(dict(metric="n_sampled", value=float(len(idx))))
    rows.append(dict(metric="n_bins", value=float(n_bins)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-ratios", type=float, nargs="+", default=[0.25, 0.40, 0.60])
    ap.add_argument("--n-eval", type=int, default=50, help="test spectra per ratio")
    ap.add_argument("--n-train", type=int, default=500, help="reference pool for PCA / nn_copy")
    ap.add_argument("--pca-k", type=int, default=50)
    ap.add_argument("--row-gap", type=int, default=500,
                    help="nn_copy matches closer than this in row index are "
                         "flagged as possibly same-study")
    ap.add_argument("--peak-quantile", type=float, default=0.95)
    ap.add_argument("--redundancy-sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/analysis/reconstruction_baselines")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = np.load(ROOT / CORPUS, mmap_mode="r")
    n, length = corpus.shape
    model = load_model(length)
    patch = model.patch_size
    n_tokens = length // patch
    print(f"corpus {n} x {length}; patch {patch} -> {n_tokens} tokens")

    rng0 = np.random.default_rng(args.seed)
    train_rows = np.sort(rng0.choice(n, args.n_train, replace=False))
    train = np.asarray(corpus[train_rows], dtype=np.float32)
    mu = train.mean(0)
    print(f"building {args.pca_k}-component PCA basis from {args.n_train} spectra ...")
    basis = np.linalg.svd(train - mu, full_matrices=False)[2][:args.pca_k]

    eval_rows = np.linspace(0, n - 1, args.n_eval, dtype=int)
    grid = np.arange(length)
    pca_name = f"pca{args.pca_k}_from_visible"
    per_spectrum, summary = [], []

    for ratio in args.mask_ratios:
        n_hide = int(round(ratio * n_tokens))
        print(f"\nmask ratio {ratio:.0%}  ({n_hide}/{n_tokens} patches)")
        for j, row in enumerate(eval_rows):
            x = np.asarray(corpus[int(row)], dtype=np.float32)
            rng = np.random.default_rng(7 + j)
            tok = np.zeros(n_tokens, dtype=bool)
            tok[rng.choice(n_tokens, n_hide, replace=False)] = True
            hid = np.repeat(tok, patch)[:length]
            if hid.size < length:                       # length not a multiple of patch
                hid = np.concatenate([hid, np.zeros(length - hid.size, dtype=bool)])
            vis = ~hid

            fed = x.copy()
            fed[hid] = 0.0
            with torch.no_grad():
                rec, _ = model(torch.from_numpy(fed).unsqueeze(0),
                               mask=torch.from_numpy(tok).unsqueeze(0))
            rec = rec.squeeze(0).numpy().reshape(-1)[:length]

            coef, *_ = np.linalg.lstsq(basis[:, vis].T, (x - mu)[vis], rcond=None)
            nn_i = int(np.argmin(((train[:, vis] - x[vis]) ** 2).sum(1)))
            row_gap = abs(int(train_rows[nn_i]) - int(row))

            preds = {
                "dnn": rec,
                "corpus_mean": mu,
                "linear_interp": np.interp(grid, grid[vis], x[vis]),
                pca_name: mu + coef @ basis,
                "nn_copy": train[nn_i],
            }
            thr = np.quantile(x[hid], args.peak_quantile)
            peak, base = hid & (x > thr), hid & (x <= thr)
            for name, p in preds.items():
                per_spectrum.append(dict(
                    mask_ratio=ratio, row=int(row), predictor=name,
                    r_masked=pearson(x[hid], p[hid]),
                    r_whole=pearson(x, p),
                    r_masked_peak=pearson(x[peak], p[peak]) if peak.sum() > 10 else np.nan,
                    r_masked_baseline=pearson(x[base], p[base]) if base.sum() > 10 else np.nan,
                    nn_row_gap=row_gap if name == "nn_copy" else np.nan))

        block = pd.DataFrame(per_spectrum)
        block = block[block.mask_ratio == ratio]
        for name, g in block.groupby("predictor", sort=False):
            far = g[g.nn_row_gap > args.row_gap] if name == "nn_copy" else g
            summary.append(dict(
                mask_ratio=ratio, predictor=name, n=len(g),
                r_masked_mean=g.r_masked.mean(), r_masked_sd=g.r_masked.std(ddof=1),
                r_whole_mean=g.r_whole.mean(),
                r_masked_peak_mean=g.r_masked_peak.mean(),
                r_masked_baseline_mean=g.r_masked_baseline.mean(),
                r_masked_mean_rowgap_filtered=far.r_masked.mean() if len(far) else np.nan,
                n_rowgap_filtered=len(far) if name == "nn_copy" else np.nan))
            print(f"  {name:<22} r_masked = {g.r_masked.mean():.3f} "
                  f"+- {g.r_masked.std(ddof=1):.3f}"
                  + (f"   (row-gap>{args.row_gap}: {far.r_masked.mean():.3f}, n={len(far)})"
                     if name == "nn_copy" else ""))

    per_df = pd.DataFrame(per_spectrum)
    sum_df = pd.DataFrame(summary)
    per_df.to_csv(out_dir / "recon_baselines_per_spectrum.csv", index=False)
    sum_df.to_csv(out_dir / "recon_baselines_summary.csv", index=False)

    print("\ncorpus redundancy ...")
    red = corpus_redundancy(corpus, args.redundancy_sample, 2048, args.seed + 1)
    red.to_csv(out_dir / "corpus_redundancy.csv", index=False)
    print(red.to_string(index=False))

    print("\n" + "=" * 74)
    print("  DNN margin over the best non-learned baseline")
    print("=" * 74)
    for ratio, g in sum_df.groupby("mask_ratio"):
        dnn = float(g[g.predictor == "dnn"].r_masked_mean.iloc[0])
        others = g[g.predictor != "dnn"]
        best = others.loc[others.r_masked_mean.idxmax()]
        print(f"  mask {ratio:.0%}:  dnn {dnn:.3f}  vs  {best.predictor} "
              f"{best.r_masked_mean:.3f}   margin {dnn - best.r_masked_mean:+.3f}")
    print(f"\nWrote {out_dir}/recon_baselines_summary.csv")
    print(f"Wrote {out_dir}/recon_baselines_per_spectrum.csv")
    print(f"Wrote {out_dir}/corpus_redundancy.csv")


if __name__ == "__main__":
    main()
