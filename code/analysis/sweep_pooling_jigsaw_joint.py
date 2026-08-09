#!/usr/bin/env python3
"""Does position-preserving pooling help jigsaw/joint the way it helped masking?

Experiment #4 (docs §5c) found flatten/regional pooling beats mean-pooling for
the masking family on 5/5 targets (+0.030..+0.129). jigsaw and joint were left
on their native pooling (mean-pool per bin size, concatenated across bin
sizes) because no alternative had been evidenced for them -- not because one
was ruled out. This sweep closes that gap using the EXISTING v3 checkpoints
(no retraining): both families already expose per-token encodings before
their internal mean-pool (jigsaw's per-bin-size transformer output; joint's
encode_bins), so the same frozen pool_tokens() transform used for masking
applies per bin size, then concatenates across bin sizes exactly as native
does at G=1.

Token embeddings are extracted ONCE per (family, dataset) and re-pooled for
every G, so the sweep costs one forward pass, not one per G. Valid G values
are powers of 2 up to 64 (the smallest bin size, 2048, gives only 64 tokens;
joint's masked-task component at bin_size=1024 gives 128 tokens, so 64 divides
it too). G=1 reproduces native; "flatten" keeps every token (~184k jigsaw
features, ~230k joint) which is likely too high-dimensional for n=37..113 but
is included for completeness.

Both families are v3-pretrained (checked: CHECKPOINTS['jigsaw'/'joint'] both
point at *_v3_... paths), matching everything else read against the v3
reference in docs/SSL_vs_classical_analysis.md.
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
from linear_probe_frozen_embeddings import CHECKPOINTS, pool_tokens, _maybe_seed  # noqa: E402

GROUPS = [1, 2, 4, 8, 16, 32, 64, "flatten"]


def jigsaw_tokens(ckpt_path, spectra, device, batch_size=4):
    """Per-bin-size per-token encodings, before any pooling. dict[bin_size] -> (N,T,D)."""
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
    per_bin = {bs: [] for bs in bin_sizes}
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            for bs in bin_sizes:
                usable = (x.shape[1] // bs) * bs
                bins = x[:, :usable].reshape(x.shape[0], usable // bs, bs)
                e = model.input_projections[str(bs)](bins)
                pos = torch.arange(e.shape[1], device=e.device)
                e = e + model.slot_embedding(pos).unsqueeze(0)
                e = model.transformer(e)
                per_bin[bs].append(e.cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {bs: np.concatenate(v, axis=0) for bs, v in per_bin.items()}, bin_sizes


def joint_tokens(ckpt_path, spectra, device, batch_size=4):
    """Per-task per-token encodings (jigsaw path per bin size + masked path)."""
    from train_joint_ssl import build_joint_model_from_loaded_checkpoint, TASK_JIGSAW, TASK_MASKED

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_joint_model_from_loaded_checkpoint(ck, device)
    model.eval()
    bin_sizes = [int(b) for b in ck.get("jigsaw_bin_sizes", model.jigsaw_bin_sizes)]
    spectrum_length, mask_bin_size = model.spectrum_length, model.mask_bin_size
    per_task = {bs: [] for bs in bin_sizes}
    per_task["masked"] = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            x = x[:, :spectrum_length]
            for bs in bin_sizes:
                trimmed = (spectrum_length // bs) * bs
                bins = x[:, :trimmed].reshape(x.shape[0], trimmed // bs, bs)
                encoded = model.encode_bins(bins, bs, TASK_JIGSAW, None)
                per_task[bs].append(encoded.cpu().numpy())
            trimmed = (spectrum_length // mask_bin_size) * mask_bin_size
            bins = x[:, :trimmed].reshape(x.shape[0], trimmed // mask_bin_size, mask_bin_size)
            no_mask = torch.zeros(bins.shape[0], bins.shape[1], dtype=torch.bool, device=x.device)
            encoded = model.encode_bins(bins, mask_bin_size, TASK_MASKED, no_mask)
            per_task["masked"].append(encoded.cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {k: np.concatenate(v, axis=0) for k, v in per_task.items()}, bin_sizes + ["masked"]


def repool(token_dict, keys, groups):
    """Apply pool_tokens per component (as a torch tensor), concat across components."""
    mode = "flatten" if groups == "flatten" else f"regional:{groups}"
    feats = []
    for k in keys:
        t = torch.from_numpy(token_dict[k])
        n_tok = t.shape[1]
        g = n_tok if groups == "flatten" else min(groups, n_tok)
        feats.append(pool_tokens(t, "flatten" if groups == "flatten" else f"regional:{g}").numpy())
    return np.concatenate(feats, axis=1)


def cv_scores(features, labels, n_splits, seed=42):
    n_classes = len(np.unique(labels))
    split_iter = (LeaveOneOut().split(features) if n_splits == "loo" else
                 StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(features, labels))
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["brc_t2d_cancer", "brc_t2d_diabetes", "mtbls563", "mtbls326", "barth"])
    ap.add_argument("--families", nargs="+", default=["jigsaw", "joint"])
    ap.add_argument("--groups", nargs="+", default=GROUPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/analysis/pooling_sweep_jigsaw_joint")
    args = ap.parse_args()
    groups = [int(g) if str(g).isdigit() else g for g in args.groups]

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
        print(f"\n=== {name}  n={len(labels)}  cv={cfg['n_splits']} ===", flush=True)

        for family in args.families:
            extractor = jigsaw_tokens if family == "jigsaw" else joint_tokens
            token_dict, keys = extractor(CHECKPOINTS[family], spectra, device)
            for g in groups:
                feats = repool(token_dict, keys, g)
                bal, auc = cv_scores(feats, labels, cfg["n_splits"], args.seed)
                rows.append(dict(dataset=name, family=family, groups=str(g), embed_dim=feats.shape[1],
                                 balanced_accuracy=bal, roc_auc=auc, n_samples=len(labels)))
                print(f"  {family:8s} groups={str(g):8s} d={feats.shape[1]:<7d} bal={bal:.4f} auc={auc:.4f}", flush=True)
            del token_dict

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pooling_sweep_jigsaw_joint.csv", index=False)
    print(f"\nWrote {out_dir / 'pooling_sweep_jigsaw_joint.csv'}\n")

    for family in args.families:
        sub = df[df.family == family]
        piv = sub.pivot_table(index="dataset", columns="groups", values="balanced_accuracy")
        cols = [str(g) for g in groups if str(g) in piv.columns]
        piv = piv[cols]
        print(f"--- {family}: balanced accuracy by pooling ---")
        print(piv.round(4).to_string())
        native = piv["1"] if "1" in piv.columns else None
        if native is not None:
            best_g = piv.drop(columns=["1"]).max(axis=1)
            print(f"  native (G=1) mean: {native.mean():.4f}   best-other-G mean: {best_g.mean():.4f}   "
                  f"delta: {best_g.mean() - native.mean():+.4f}")
        print()


if __name__ == "__main__":
    main()
