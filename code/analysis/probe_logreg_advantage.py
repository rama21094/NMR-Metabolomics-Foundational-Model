#!/usr/bin/env python3
"""Diagnose *why* logistic regression on binned spectra beats the SSL backbones.

The headline v4 result is that LogReg on 1024-bin integrated absolute area
outperforms all three SSL families on every dataset, by up to 18 points of
balanced accuracy. Two very different explanations predict that same outcome:

  (A) REPRESENTATION bottleneck -- the SSL embedding discards the
      discriminative signal that the binned-area features retain. Then any
      classifier on the embedding underperforms, including LogReg itself.
  (B) CLASSIFIER/HEAD bottleneck -- the embedding retains the signal but the
      MLP head can't extract it from ~70 training samples. Then LogReg on the
      *embedding* should recover most of the binned-feature performance.

The two are distinguished by running the SAME LogReg pipeline
(StandardScaler + LogisticRegression(C=1, class_weight="balanced")) on both
representations over the SAME cross-validation folds. That is the core of
this script.

Also included, because each rules out a distinct trivial explanation:
  * bin-count sweep (16..4096) -- is the advantage from fine spectral detail,
    or would a handful of coarse regions do just as well?
  * raw 131072-point spectrum -- is binning itself necessary?
  * single global scalars (total absolute area, row std) -- a 1-feature
    control. If total area alone predicts the label, the "signal" is a
    global intensity/dilution confound rather than metabolite chemistry.
  * label-permutation null -- with n<=142 samples and >=1024 features, the
    honest question for a perfect/near-perfect score is what the null
    distribution looks like under the identical CV protocol.

Fold construction mirrors the real evaluation exactly
(StratifiedKFold(n_splits, shuffle=True, random_state=seed)) so numbers here
are directly comparable to the committed summary.csv files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(ROOT / "code" / "training"))

from brc_t2d_common import binned_abs_area  # noqa: E402


def logreg_pipeline(seed: int) -> Pipeline:
    """Exactly the classical pipeline used by the real evaluation scripts."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, C=1.0, random_state=seed, class_weight="balanced")),
    ])


def cv_balanced_accuracy(features: np.ndarray, labels: np.ndarray, n_splits, seed: int) -> float:
    """Pooled out-of-fold balanced accuracy -- same aggregation as summary.csv.

    n_splits="loo" reproduces the LOOCV datasets (Barth, MTBLS326); an int
    reproduces the stratified k-fold datasets (BrC-T2D, MTBLS563)."""
    if n_splits == "loo":
        from sklearn.model_selection import LeaveOneOut
        splitter = LeaveOneOut()
        split_iter = splitter.split(features)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(features, labels)
    oof = np.empty_like(labels)
    for train_idx, test_idx in split_iter:
        model = logreg_pipeline(seed)
        model.fit(features[train_idx], labels[train_idx])
        oof[test_idx] = model.predict(features[test_idx])
    return float(balanced_accuracy_score(labels, oof))


def load_masking_backbone(checkpoint_path: Path, spectrum_length: int, device: torch.device):
    """Rebuild the masked-SSL backbone exactly as the real evaluation does, by
    reusing that script's own infer_mae_config so the architecture can't drift
    from what produced the committed results."""
    from trainer_revised import NMRMaskedAutoencoder
    from barth_all_models_loocv import infer_mae_config

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    config = infer_mae_config(state, 8, 0.0)
    model = NMRMaskedAutoencoder(spectrum_length=spectrum_length, **config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, config


def extract_embeddings(model, spectra: np.ndarray, device: torch.device, strategy: str, batch_size: int = 8) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, spectra.shape[0], batch_size):
            batch = np.asarray(spectra[start:start + batch_size], dtype=np.float32)
            xb = torch.from_numpy(batch).to(device)
            _, encoded = model(xb, mask=None)
            emb = encoded.mean(dim=1) if strategy == "mean_pool" else encoded.reshape(encoded.shape[0], -1)
            out.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(out)


DATASETS = {
    "brc_t2d_cancer": dict(
        data="data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/BrC_T2D/BC_T2D_newlabels_metadata_mapping.csv",
        label_column="cancer_status", loader="brc_t2d", n_splits=10,
    ),
    "brc_t2d_diabetes": dict(
        data="data/BrC_T2D/BC_T2D_newlabels_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/BrC_T2D/BC_T2D_newlabels_metadata_mapping.csv",
        label_column="diabetes_status", loader="brc_t2d", n_splits=10,
    ),
    "mtbls563": dict(
        data="data/mtbls563/MTBLS563_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/mtbls563/MTBLS563_metadata_mapping.csv",
        label_column="Factor Value[Diagnosis]", loader="generic", n_splits=10,
        exclude_labels=["unknown"],
    ),
    # MTBLS326's 'label' column is Yes/No = IP3R high+low vs control (27/15).
    # Evaluated by LOOCV in the real pipeline, so probed that way here too.
    "mtbls326": dict(
        data="data/mtbls326/MTBLS326_aligned_spectra_WS625to680Zero_rowMinMax_v4.npy",
        metadata="data/mtbls326/MTBLS326_metadata_mapping.csv",
        label_column="label", loader="generic", n_splits="loo",
    ),
    # "Pool" rows are pooled-QC samples, excluded by the real Barth run (n=37).
    "barth": dict(
        data="data/Barth/aligned_128K_Workbench_Barth_Syndrome_WS625to680Zero_EDTASuppressed_rowMinMax_v4.npy",
        metadata="data/Barth/Workbench_Barth_Syndrome_metadata.csv",
        label_column="label", loader="generic", n_splits="loo",
        exclude_labels=["Pool"],
    ),
}


def load_generic(data_path, metadata_path, label_column, exclude_labels=()):
    spectra = np.load(data_path).astype(np.float32)
    meta = pd.read_csv(metadata_path)
    if label_column not in meta.columns:
        raise KeyError(f"{label_column!r} not in {metadata_path}; have {list(meta.columns)}")
    labels_raw = meta[label_column].astype(str).str.strip()
    keep = ~labels_raw.str.lower().isin({str(x).lower() for x in exclude_labels})
    keep &= labels_raw.ne("")
    meta, labels_raw = meta[keep], labels_raw[keep]
    row_col = "npy_row" if "npy_row" in meta.columns else None
    idx = meta[row_col].astype(int).to_numpy() if row_col else np.arange(len(meta))
    names = sorted(labels_raw.unique())
    mapping = {n: i for i, n in enumerate(names)}
    return spectra[idx], labels_raw.map(mapping).to_numpy(), names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["brc_t2d_cancer", "brc_t2d_diabetes", "mtbls563"])
    parser.add_argument("--masking-checkpoint", default="models/masked_ssl/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth")
    parser.add_argument("--bin-sweep", nargs="+", type=int, default=[16, 64, 256, 1024, 4096])
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="results/analysis/logreg_advantage_probe")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []

    for name in args.datasets:
        cfg = DATASETS[name]
        print(f"\n=== {name} ===", flush=True)
        if cfg["loader"] == "brc_t2d":
            from brc_t2d_common import load_brc_t2d
            spectra, labels, _, label_names = load_brc_t2d(cfg["data"], cfg["metadata"], cfg["label_column"])
        else:
            spectra, labels, label_names = load_generic(
                cfg["data"], cfg["metadata"], cfg["label_column"], cfg.get("exclude_labels", ()),
            )
        n_splits = cfg["n_splits"]
        print(f"  n={len(labels)}, classes={label_names}, counts={np.bincount(labels).tolist()}", flush=True)

        def record(rep, detail, n_feat, score):
            rows.append(dict(dataset=name, representation=rep, detail=detail, n_features=n_feat,
                             balanced_accuracy=score, n_samples=len(labels), n_classes=len(label_names)))
            print(f"  {rep:28s} {detail:14s} d={n_feat:<7d} bal_acc={score:.4f}", flush=True)

        # 1. Bin-count sweep on the classical representation.
        for n_bins in args.bin_sweep:
            feats = binned_abs_area(spectra, n_bins)
            record("binned_abs_area", f"{n_bins} bins", feats.shape[1],
                   cv_balanced_accuracy(feats, labels, n_splits, args.seed))

        # 2. Raw spectrum, no binning.
        record("raw_spectrum", "131072 pts", spectra.shape[1],
               cv_balanced_accuracy(spectra, labels, n_splits, args.seed))

        # 3. Single-scalar controls -- is this just a global intensity confound?
        total_area = np.abs(spectra).sum(axis=1, keepdims=True)
        record("global_scalar", "total_abs_area", 1,
               cv_balanced_accuracy(total_area, labels, n_splits, args.seed))
        row_std = spectra.std(axis=1, keepdims=True)
        record("global_scalar", "row_std", 1,
               cv_balanced_accuracy(row_std, labels, n_splits, args.seed))

        # 4. Frozen masked-SSL embeddings, same LogReg, same folds.
        #    This is the representation-vs-head discriminator.
        model, info = load_masking_backbone(Path(args.masking_checkpoint), spectra.shape[1], device)
        print(f"  backbone: {info}", flush=True)
        for strategy in ["mean_pool", "flatten"]:
            emb = extract_embeddings(model, spectra, device, strategy)
            record("masked_ssl_embedding", strategy, emb.shape[1],
                   cv_balanced_accuracy(emb, labels, n_splits, args.seed))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 5. Permutation null for the 1024-bin classical representation.
        feats_1024 = binned_abs_area(spectra, 1024)
        observed = cv_balanced_accuracy(feats_1024, labels, n_splits, args.seed)
        rng = np.random.default_rng(args.seed)
        null = []
        for i in range(args.n_permutations):
            null.append(cv_balanced_accuracy(feats_1024, rng.permutation(labels), n_splits, args.seed))
        null = np.array(null)
        p_value = float((null >= observed).sum() + 1) / (len(null) + 1)
        print(f"  permutation null (1024 bins, {len(null)} perms): "
              f"mean={null.mean():.4f} p95={np.percentile(null,95):.4f} max={null.max():.4f} "
              f"| observed={observed:.4f} p={p_value:.4g}", flush=True)
        rows.append(dict(dataset=name, representation="permutation_null", detail="1024 bins",
                         n_features=feats_1024.shape[1], balanced_accuracy=float(null.mean()),
                         n_samples=len(labels), n_classes=len(label_names),
                         null_p95=float(np.percentile(null, 95)), null_max=float(null.max()),
                         observed=observed, p_value=p_value))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "probe_results.csv", index=False)
    print(f"\nWrote {out_dir / 'probe_results.csv'}")
    print("\n" + df[["dataset", "representation", "detail", "n_features", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
