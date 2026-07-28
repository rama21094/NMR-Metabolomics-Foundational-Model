#!/usr/bin/env python3
"""Experiment #2: swap the SSL fine-tuning head for a plain linear probe.

Every SSL family in this project ends in the same shape of classifier:

    pooled embedding -> LayerNorm -> Dropout -> Linear(d, n_classes)

trained with Adam for ~50 epochs with early stopping on a handful of samples.
That is a linear classifier fitted by SGD. sklearn's LogisticRegression on the
same pooled embedding is *also* a linear classifier, but fitted by L-BFGS to
convergence with an explicit L2 penalty and standardized inputs.

So running LogReg on the frozen pooled embedding, over the same CV folds,
isolates the optimization/regularization quality of the head from everything
else: identical backbone, identical pooling, identical features, identical
folds. Any difference is attributable to how the linear map is fitted, not to
what the backbone learned.

Pooling replicates each family's real classifier exactly:
  masking  -- encoded.mean(dim=1)                                    (128-d)
  jigsaw   -- per-bin-size transformer pass, mean-pool, concat 4     (768-d)
  joint    -- encode_spectrum(bin_sizes, include_masked_task=True)   (960-d)

Input normalization follows what each checkpoint records. For these v4
checkpoints on already row-min-max-normalized [0,1] data, both jigsaw ("auto"
-> False, since min>=-1e-4 and max<=1.5) and joint ("checkpoint" ->
normalize_resolved=False) resolve to no additional normalization, matching the
committed evaluation runs.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for sub in ("code/evaluation", "code/training", "code/analysis"):
    sys.path.insert(0, str(ROOT / sub))

from probe_logreg_advantage import DATASETS, load_generic, logreg_pipeline  # noqa: E402

CHECKPOINTS = {
    "masking": "models/masked_ssl/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth",
    "jigsaw": "models/jigsaw/multibin/20260725_085608/multibin_20260725_085608_best.pth",
    "joint": "models/joint_ssl/joint_ssl_20260725_085627/joint_ssl_20260725_085627_best.pth",
}

# Where the officially reported head numbers live, per (dataset, family).
OFFICIAL = {
    ("brc_t2d_cancer", "masking"): ("results/cv10/brc_t2d_newlabels_v4/cancer_status/summary.csv", "masking"),
    ("brc_t2d_cancer", "jigsaw"): ("results/cv10/brc_t2d_newlabels_v4/cancer_status/summary.csv", "jigsaw"),
    ("brc_t2d_cancer", "joint"): ("results/cv10/brc_t2d_newlabels_v4_joint/cancer_status/summary.csv", "joint_ssl"),
    ("brc_t2d_diabetes", "masking"): ("results/cv10/brc_t2d_newlabels_v4/diabetes_status/summary.csv", "masking"),
    ("brc_t2d_diabetes", "jigsaw"): ("results/cv10/brc_t2d_newlabels_v4/diabetes_status/summary.csv", "jigsaw"),
    ("brc_t2d_diabetes", "joint"): ("results/cv10/brc_t2d_newlabels_v4_joint/diabetes_status/summary.csv", "joint_ssl"),
    ("mtbls563", "masking"): ("results/loocv/mtbls563_all_models_v4/summary.csv", "masking"),
    ("mtbls563", "jigsaw"): ("results/loocv/mtbls563_all_models_v4/summary.csv", "jigsaw"),
    ("mtbls563", "joint"): ("results/loocv/mtbls563_all_models_v4/summary.csv", "joint_ssl"),
    ("mtbls326", "masking"): ("results/loocv/mtbls326_masking_v4/summary.csv", "foundation"),
    ("mtbls326", "jigsaw"): ("results/loocv/mtbls326_jigsaw_v4/summary.csv", "jigsaw"),
    ("mtbls326", "joint"): ("results/loocv/mtbls326_joint_ssl_v4/summary.csv", "joint_ssl"),
    ("barth", "masking"): ("results/loocv/barth_all_models_v4/summary.csv", "masking"),
    ("barth", "jigsaw"): ("results/loocv/barth_all_models_v4/summary.csv", "jigsaw"),
    ("barth", "joint"): ("results/loocv/barth_all_models_v4/summary.csv", "joint_ssl"),
}


def official_head(dataset: str, family: str) -> tuple[float | None, float | None]:
    key = (dataset, family)
    if key not in OFFICIAL:
        return None, None
    path, fam = OFFICIAL[key]
    if not Path(path).exists():
        return None, None
    df = pd.read_csv(path)
    sub = df[df["family"] == fam]
    if not len(sub):
        return None, None
    best = float(sub["balanced_accuracy"].max())
    frozen = sub[sub["model"].astype(str).str.endswith("frozen")]
    return best, (float(frozen["balanced_accuracy"].iloc[0]) if len(frozen) else None)


# --------------------------------------------------------------------------
# Embedding extraction, one function per family, each replicating the real
# classifier's own pooling.
# --------------------------------------------------------------------------
def embed_masking(ckpt_path, spectra, device, batch_size=8):
    from trainer_revised import NMRMaskedAutoencoder
    from barth_all_models_loocv import infer_mae_config

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    model = NMRMaskedAutoencoder(spectrum_length=spectra.shape[1], **infer_mae_config(state, 8, 0.0))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            _, enc = model(x, mask=None)
            out.append(enc.mean(dim=1).cpu().numpy())
    del model
    return np.vstack(out).astype(np.float32)


def embed_jigsaw(ckpt_path, spectra, device, batch_size=4):
    from train_jigsaw_spectra import JigsawNMRModel

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyperparameters", {})
    bin_sizes = [int(b) for b in ck["bin_sizes"]]
    model = JigsawNMRModel(
        spectrum_length=int(ck["spectrum_length"]), bin_sizes=bin_sizes,
        d_model=int(hp.get("d_model", 192)), nhead=int(hp.get("nhead", 6)),
        num_layers=int(hp.get("num_layers", 4)),
        dim_feedforward=int(hp.get("dim_feedforward", 768)),
        dropout=float(hp.get("dropout", 0.15)),
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            per_bin = []
            for bs in bin_sizes:
                usable = (x.shape[1] // bs) * bs
                bins = x[:, :usable].reshape(x.shape[0], usable // bs, bs)
                e = model.input_projections[str(int(bs))](bins)
                pos = torch.arange(e.shape[1], device=e.device)
                e = e + model.slot_embedding(pos).unsqueeze(0)
                e = model.transformer(e)
                per_bin.append(e.mean(dim=1))
            out.append(torch.cat(per_bin, dim=1).cpu().numpy())
    del model
    return np.vstack(out).astype(np.float32)


def embed_joint(ckpt_path, spectra, device, batch_size=4):
    from train_joint_ssl import build_joint_model_from_loaded_checkpoint

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_joint_model_from_loaded_checkpoint(ck, device)
    model.eval()
    bin_sizes = [int(b) for b in ck.get("jigsaw_bin_sizes", model.jigsaw_bin_sizes)]
    out = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            out.append(model.encode_spectrum(x, bin_sizes, include_masked_task=True).cpu().numpy())
    del model
    return np.vstack(out).astype(np.float32)


EMBEDDERS = {"masking": embed_masking, "jigsaw": embed_jigsaw, "joint": embed_joint}


def cv_scores(features, labels, n_splits, seed):
    """Pooled OOF balanced accuracy and ROC-AUC (macro-OVR when multiclass)."""
    n_classes = len(np.unique(labels))
    if n_splits == "loo":
        split_iter = LeaveOneOut().split(features)
    else:
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(features, labels)
    oof = np.empty_like(labels)
    proba = np.zeros((len(labels), n_classes), dtype=float)
    for tr, te in split_iter:
        model = logreg_pipeline(seed)
        model.fit(features[tr], labels[tr])
        oof[te] = model.predict(features[te])
        p = model.predict_proba(features[te])
        # Map fold-local classes back onto the global column order.
        for j, cls in enumerate(model.classes_):
            proba[te, cls] = p[:, j]
    bal = float(balanced_accuracy_score(labels, oof))
    try:
        auc = (float(roc_auc_score(labels, proba[:, 1])) if n_classes == 2
               else float(roc_auc_score(labels, proba, multi_class="ovr", average="macro")))
    except ValueError:
        auc = float("nan")
    return bal, auc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+",
                        default=["brc_t2d_cancer", "brc_t2d_diabetes", "mtbls563", "mtbls326", "barth"])
    parser.add_argument("--families", nargs="+", default=["masking", "jigsaw", "joint"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="results/analysis/linear_probe_frozen")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []

    for name in args.datasets:
        cfg = DATASETS[name]
        if cfg["loader"] == "brc_t2d":
            from brc_t2d_common import load_brc_t2d
            spectra, labels, _, label_names = load_brc_t2d(cfg["data"], cfg["metadata"], cfg["label_column"])
        else:
            spectra, labels, label_names = load_generic(
                cfg["data"], cfg["metadata"], cfg["label_column"], cfg.get("exclude_labels", ()))
        print(f"\n=== {name}  n={len(labels)}  classes={label_names} "
              f"cv={cfg['n_splits']} ===", flush=True)

        for family in args.families:
            emb = EMBEDDERS[family](CHECKPOINTS[family], spectra, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            bal, auc = cv_scores(emb, labels, cfg["n_splits"], args.seed)
            head_best, head_frozen = official_head(name, family)
            delta = (bal - head_best) if head_best is not None else float("nan")
            rows.append(dict(
                dataset=name, family=family, embed_dim=emb.shape[1],
                linear_probe_bal_acc=bal, linear_probe_roc_auc=auc,
                ssl_head_best=head_best, ssl_head_frozen=head_frozen,
                delta_probe_minus_head_best=delta, n_samples=len(labels),
                n_classes=len(label_names),
            ))
            hb = f"{head_best:.4f}" if head_best is not None else "n/a"
            print(f"  {family:8s} d={emb.shape[1]:<5d} probe_bal={bal:.4f} auc={auc:.4f} "
                  f"| reported head best={hb} | delta={delta:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "linear_probe_results.csv", index=False)
    print(f"\nWrote {out_dir / 'linear_probe_results.csv'}\n")
    show = ["dataset", "family", "embed_dim", "linear_probe_bal_acc", "ssl_head_best", "delta_probe_minus_head_best"]
    print(df[show].to_string(index=False))
    print(f"\nMean improvement of linear probe over reported head: "
          f"{df['delta_probe_minus_head_best'].mean():+.4f}")
    print(f"Wins/ties/losses: "
          f"{(df.delta_probe_minus_head_best > 0.005).sum()}/"
          f"{(df.delta_probe_minus_head_best.abs() <= 0.005).sum()}/"
          f"{(df.delta_probe_minus_head_best < -0.005).sum()}")


if __name__ == "__main__":
    main()
