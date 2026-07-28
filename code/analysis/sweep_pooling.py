#!/usr/bin/env python3
"""How much chemical-shift position does the classifier actually need?

Experiment #4 showed that replacing mean-pooling with a flattened
position-preserving embedding helps on all five targets (+0.030..+0.129). But
those two are the extreme ends of a spectrum, and they differ in TWO ways at
once: how much positional detail is retained, AND the feature dimension
(128 vs 16384 at patch_size=1024) -- which matters a lot at n=37..113 samples.

Regional pooling separates those. Split the T tokens into G contiguous groups,
mean-pool within each group, concatenate: dimension is G*d_model, and G controls
retained positional detail.

    G = 1   -> identical to mean-pool      (d = 128)
    G = T   -> identical to flatten        (d = 16384 at ps1024)

Sweeping G traces the whole curve and finds where the accuracy/dimension
trade-off is best. It is also a strictly frozen, label-independent transform, so
it composes with the existing linear probe without any leakage concern -- unlike
learned attention pooling, whose parameters would have to be fitted inside each
training fold.

Token embeddings are extracted ONCE per dataset and re-pooled for every G, so
the sweep costs barely more than a single evaluation.
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

DEFAULT_CKPT = ("models/masked_ssl/combine_unique_MetaboLights_Workbench_Water_EDTA_"
                "Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth")


def token_embeddings(ckpt_path, spectra, device, nhead=None, batch_size=16):
    """Per-token encoder output, (N, T, d_model) -- pooled later."""
    from trainer_revised import NMRMaskedAutoencoder
    from barth_all_models_loocv import infer_mae_config

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    recorded = ck.get("hyperparameters", {}).get("nhead")
    # Legacy checkpoints predate architecture recording; the committed eval
    # scripts guessed 8, so that stays the fallback for comparability.
    nh = nhead if nhead is not None else (recorded if recorded else 8)
    cfg = infer_mae_config(state, int(nh), 0.0)
    model = NMRMaskedAutoencoder(spectrum_length=spectra.shape[1], **cfg)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            _, enc = model(x, mask=None)
            out.append(enc.cpu().numpy().astype(np.float32))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0), cfg


def regional_pool(tokens: np.ndarray, groups: int) -> np.ndarray:
    """(N, T, D) -> (N, groups*D) by mean-pooling within contiguous token groups."""
    n, t, d = tokens.shape
    if groups > t:
        raise ValueError(f"groups={groups} exceeds token count {t}")
    if t % groups:
        # Trim the tail so groups are equal-sized; T is a power of two here so
        # this never actually triggers for power-of-two group counts.
        t_use = (t // groups) * groups
        tokens = tokens[:, :t_use, :]
        t = t_use
    return tokens.reshape(n, groups, t // groups, d).mean(axis=2).reshape(n, groups * d)


def cv_scores(features, labels, n_splits, seed=42):
    n_classes = len(np.unique(labels))
    if n_splits == "loo":
        split_iter = LeaveOneOut().split(features)
    else:
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(features, labels)
    oof = np.empty_like(labels)
    proba = np.zeros((len(labels), n_classes))
    for tr, te in split_iter:
        m = logreg_pipeline(seed)
        m.fit(features[tr], labels[tr])
        oof[te] = m.predict(features[te])
        p = m.predict_proba(features[te])
        for j, cls in enumerate(m.classes_):
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
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--nhead", type=int, default=None,
                        help="Override attention heads; default uses the recorded value, else 8 (legacy).")
    parser.add_argument("--groups", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/analysis/pooling_sweep")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in args.datasets:
        cfg = DATASETS[name]
        if cfg["loader"] == "brc_t2d":
            from brc_t2d_common import load_brc_t2d
            spectra, labels, _, label_names = load_brc_t2d(cfg["data"], cfg["metadata"], cfg["label_column"])
        else:
            spectra, labels, label_names = load_generic(
                cfg["data"], cfg["metadata"], cfg["label_column"], cfg.get("exclude_labels", ()))

        tokens, mcfg = token_embeddings(args.checkpoint, spectra, device, args.nhead)
        n, t, d = tokens.shape
        print(f"\n=== {name}  n={len(labels)}  tokens={t}  d_model={d}  cv={cfg['n_splits']} ===", flush=True)

        for g in args.groups:
            if g > t:
                continue
            feats = regional_pool(tokens, g)
            bal, auc = cv_scores(feats, labels, cfg["n_splits"], args.seed)
            equiv = " (= mean-pool)" if g == 1 else (" (= flatten)" if g == t else "")
            rows.append(dict(dataset=name, groups=g, embed_dim=feats.shape[1],
                             balanced_accuracy=bal, roc_auc=auc,
                             tokens=t, d_model=d, n_samples=len(labels)))
            print(f"  groups={g:<4d} d={feats.shape[1]:<6d} bal={bal:.4f} auc={auc:.4f}{equiv}", flush=True)
        del tokens

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pooling_sweep_results.csv", index=False)
    print(f"\nWrote {out_dir / 'pooling_sweep_results.csv'}\n")
    piv = df.pivot_table(index="dataset", columns="groups", values="balanced_accuracy")
    print("--- balanced accuracy by number of pooling groups ---")
    print(piv.round(3).to_string())
    print()
    print("best group count per dataset:")
    for ds, r in piv.iterrows():
        g = int(r.idxmax())
        print(f"  {ds:18s} best groups={g:<4d} bal={r.max():.4f}   "
              f"(mean-pool={r.get(1, float('nan')):.4f}, flatten={r.dropna().iloc[-1]:.4f})")
    print()
    print("mean balanced accuracy across datasets, by group count:")
    print(df.groupby("groups")["balanced_accuracy"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
