"""Find outlier spectra in the pretraining corpus.

All 9670+ spectra are human blood serum/plasma, so they should share a broadly
similar metabolite envelope; a spectrum wildly unlike the rest (corrupted
acquisition, mislabeled sample type, a missed suppression artifact, etc.)
could disproportionately hurt VAE/SSL training.

Two complementary, independent views are combined, since a real outlier
should show up under more than one:

1. Peak-space multivariate outlier detection, using the 60 canonical peak
   intensities already extracted by peak_extraction.py
   (results/analysis/peak_saturation_full_nw/peak_values.npy) as a compact,
   domain-meaningful feature vector per spectrum. Three standard, orthogonal
   methods vote:
     - Robust Mahalanobis distance (MinCovDet): parametric, assumes roughly
       elliptical structure, robust to a minority of contaminating outliers.
     - Isolation Forest: nonparametric, catches non-elliptical structure.
     - Local Outlier Factor: density-based, catches local outliers that
       aren't extreme in any single global sense.
   Missing (undetected) peak values are median-imputed per peak; the number
   of missing peaks for a spectrum is itself reported as a secondary signal
   (a spectrum where most canonical peaks failed to match may be atypical or
   just low SNR, not necessarily biologically anomalous).

2. Whole-spectrum shape correlation to the corpus median reference spectrum
   -- independent of the 60-peak extraction, so it also catches gross shape
   anomalies (baseline problems, misalignment) the peak features might miss.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA = "data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax.npy"
DEFAULT_PEAKS_DIR = "results/analysis/peak_saturation_full_nw"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--peaks-dir", default=DEFAULT_PEAKS_DIR)
    p.add_argument("--contamination", type=float, default=0.02,
                   help="expected outlier fraction, passed to IsolationForest/LOF")
    p.add_argument("--top-n", type=int, default=20, help="how many worst outliers to plot/report")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/analysis/outlier_detection")
    return p.parse_args()


def build_feature_matrix(peaks_dir):
    values = np.load(Path(peaks_dir) / "peak_values.npy")  # (n_spectra, n_peaks), NaN = undetected
    n_missing = np.isnan(values).sum(axis=1)
    col_median = np.nanmedian(values, axis=0)
    imputed = np.where(np.isnan(values), col_median[None, :], values)
    return imputed, n_missing


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    features, n_missing = build_feature_matrix(args.peaks_dir)
    n_spectra, n_peaks = features.shape
    print(f"Feature matrix: {features.shape}  (median-imputed; missing-peak counts range "
          f"{n_missing.min()}-{n_missing.max()} of {n_peaks})")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # 1. Robust Mahalanobis distance
    mcd = MinCovDet(random_state=args.seed).fit(X)
    mahal_dist = np.sqrt(mcd.mahalanobis(X))

    # 2. Isolation Forest (higher score = more outlying; sklearn's decision_function is inverted)
    iso = IsolationForest(contamination=args.contamination, random_state=args.seed, n_estimators=300)
    iso.fit(X)
    iso_score = -iso.decision_function(X)  # flip sign: larger = more anomalous

    # 3. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=args.contamination)
    lof.fit_predict(X)
    lof_score = -lof.negative_outlier_factor_  # larger = more anomalous

    n_flag = max(1, int(round(args.contamination * n_spectra)))

    def top_mask(score, n):
        idx = np.argsort(score)[-n:]
        mask = np.zeros(len(score), dtype=bool)
        mask[idx] = True
        return mask

    mahal_flag = top_mask(mahal_dist, n_flag)
    iso_flag = top_mask(iso_score, n_flag)
    lof_flag = top_mask(lof_score, n_flag)
    vote_count = mahal_flag.astype(int) + iso_flag.astype(int) + lof_flag.astype(int)

    # 4. Whole-spectrum shape correlation to the corpus median reference
    print("Computing corpus median reference spectrum...")
    data = np.load(args.data, mmap_mode="r")
    ref = np.median(np.asarray(data), axis=0)
    ref_centered = ref - ref.mean()
    ref_norm = np.linalg.norm(ref_centered)

    corrs = np.empty(n_spectra)
    chunk = 500
    for start in range(0, n_spectra, chunk):
        end = min(start + chunk, n_spectra)
        rows = np.asarray(data[start:end], dtype=np.float64)
        rows = np.nan_to_num(rows, nan=0.0)
        rows_centered = rows - rows.mean(axis=1, keepdims=True)
        rows_norm = np.linalg.norm(rows_centered, axis=1)
        denom = np.where(rows_norm * ref_norm > 0, rows_norm * ref_norm, 1e-12)
        corrs[start:end] = (rows_centered @ ref_centered) / denom

    df = pd.DataFrame({
        "row_index": np.arange(n_spectra),
        "n_missing_peaks": n_missing,
        "mahalanobis_dist": mahal_dist,
        "isolation_forest_score": iso_score,
        "lof_score": lof_score,
        "vote_count": vote_count,
        "shape_correlation_to_median": corrs,
    })
    df = df.sort_values("vote_count", ascending=False)
    df.to_csv(out_dir / "outlier_scores.csv", index=False)

    consensus = df[df["vote_count"] >= 2]
    print(f"\n{len(consensus)} spectra flagged by >=2/3 peak-space methods (out of {n_flag} flagged by each individually).")
    print(f"Correlation-to-median-reference: p1={np.percentile(corrs, 1):.4f}, "
          f"p50={np.percentile(corrs, 50):.4f}, min={corrs.min():.4f}")
    low_corr_threshold = np.percentile(corrs, 1)
    low_corr_rows = df[df["shape_correlation_to_median"] <= low_corr_threshold]
    overlap = set(consensus["row_index"]) & set(low_corr_rows["row_index"])
    print(f"{len(low_corr_rows)} spectra in the bottom 1% of shape-correlation to the median reference.")
    print(f"Overlap between peak-space consensus outliers and low-shape-correlation outliers: {len(overlap)}")

    # PCA scatter, colored by vote_count
    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=df.sort_values("row_index")["vote_count"] if False else vote_count,
                     cmap="viridis", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="outlier method vote count (0-3)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of 60-peak feature space, colored by outlier consensus")
    fig.tight_layout()
    fig.savefig(out_dir / "outlier_pca_scatter.png", dpi=150)
    plt.close(fig)

    # Overlay plot of the worst consensus outliers vs. the reference spectrum
    top_rows = df.head(args.top_n)["row_index"].values
    n_plot = min(args.top_n, len(top_rows), 8)
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.0 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for ax, ridx in zip(axes, top_rows[:n_plot]):
        row = np.asarray(data[ridx], dtype=np.float64)
        ax.plot(ref, color="#999999", linewidth=0.7, label="corpus median" if ridx == top_rows[0] else None)
        ax.plot(row, color="#e34948", linewidth=0.7, alpha=0.9,
                label=f"row {ridx} (votes={df[df.row_index==ridx].vote_count.values[0]}, "
                      f"corr={df[df.row_index==ridx].shape_correlation_to_median.values[0]:.3f})")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "top_outliers_vs_reference.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_spectra": int(n_spectra),
        "n_peaks_used": int(n_peaks),
        "contamination": args.contamination,
        "n_flagged_per_method": n_flag,
        "n_consensus_2of3": int(len(consensus)),
        "n_consensus_3of3": int((df["vote_count"] == 3).sum()),
        "top_consensus_rows": consensus.head(args.top_n)["row_index"].tolist(),
        "shape_corr_p1": float(np.percentile(corrs, 1)),
        "shape_corr_min": float(corrs.min()),
        "shape_corr_median": float(np.percentile(corrs, 50)),
    }
    with open(out_dir / "outlier_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
