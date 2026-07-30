#!/usr/bin/env python3
"""Experiment #4: does reducing the masking backbone's patch size lift the
representation ceiling?

Background (docs/SSL_vs_classical_analysis.md §5). The masking backbone tokenizes
131,072 points into 131072/patch_size tokens, so patch_size sets an upper bound
on the spectral detail it can represent. At patch_size=1024 that is 128
positions, and the frozen embedding scored right at what LogReg achieves on 128
bins, while LogReg on 1024 bins scored far higher. Prediction: shrinking
patch_size should move the embedding up the bin-resolution curve.

This compares patch_size 1024 / 256 / 128 through a FROZEN linear probe
(StandardScaler + LogisticRegression C=1, class_weight balanced) over each
dataset's real CV protocol. The probe rather than the fine-tuned MLP head is the
right instrument for this specific question: experiment #2 established the
masking head underfits by ~0.12, which is head-fitting noise that would obscure
a representation change. The fine-tuned head is still worth running afterwards
for comparability with the committed baseline numbers.

nhead handling matters here. nn.MultiheadAttention keeps in_proj_weight at
(3*d_model, d_model) regardless of nhead, so a checkpoint loads silently under
the wrong head count while reinterpreting the trained weights. The ps=1024
checkpoint predates architecture recording and does not store nhead; the
official eval scripts guessed 8, while training actually used 4. The new
checkpoints record nhead=4. To keep this a clean resolution comparison, every
arm is evaluated at its TRUE nhead=4. The legacy nhead=8 reading of ps=1024 is
also reported, since that is what every committed number used.
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

ROOT = Path(__file__).resolve().parents[2]
for sub in ("code/evaluation", "code/training", "code/analysis"):
    sys.path.insert(0, str(ROOT / sub))

from probe_logreg_advantage import DATASETS, load_generic, logreg_pipeline  # noqa: E402

BASE = "models/masked_ssl"
ARMS = [
    # label, checkpoint, nhead override (None -> use recorded, else forced)
    ("ps1024_nhead8_legacy", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth", 8),
    ("ps1024_nhead4_true", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260725_085527_bs32_mr0.20-0.60_ps1024_best.pth", 4),
    ("ps256", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260728_054053_bs32_mr0.20-0.60_ps256_best.pth", None),
    ("ps128", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260728_053503_bs32_mr0.20-0.60_ps128_best.pth", None),
    # patch 2048 (64 tokens). Continues the trend in the opposite direction to
    # the refuted hypothesis, and carries MORE parameters (5.42M) than the
    # baseline, so if it still loses that is strong evidence patch 1024 is near
    # optimal rather than capacity-limited.
    ("ps2048", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260728_221507_bs32_mr0.20-0.60_ps2048_best.pth", None),
    # Capacity arm: patch 1024 held fixed, d_model 128->256, layers 3->6,
    # ff 256->512 (5.13M params, near-matched to ps2048's 5.42M). Comparing
    # these two isolates HOW ~5M parameters are best spent.
    ("ps1024_d256_L6", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260728_124558_bs32_mr0.20-0.60_ps1024_best.pth", None),
    # EXPERIMENT #7 -- 2x2 factorial over the pretext task, all four arms at the
    # winning geometry (ps1024, d128, L3, nhead4, 1.89M params) on IDENTICAL v4
    # data. exp7_D is the reference cell: every earlier masking baseline was
    # pretrained on v3, so without it the factorial would confound the objective
    # change with the data version.
    ("exp7_D_baseline_v4", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260729_052637_bs32_mr0.20-0.60_ps1024_best.pth", None),
    ("exp7_A_blk8", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260729_100612_bs32_mr0.20-0.60_ps1024_blk8_best.pth", None),
    ("exp7_B_pk025", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260729_054925_bs32_mr0.20-0.60_ps1024_pk0.25_best.pth", None),
    ("exp7_C_blk8_pk025", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260729_110345_bs32_mr0.20-0.60_ps1024_blk8_pk0.25_best.pth", None),
    # EXPERIMENT #7 FOLLOW-UP (docs §5f/§7b). Replicates of the v4 baseline, to
    # decide whether the -0.069 v3-vs-v4 gap is the corpus or just run-to-run
    # noise. Together with the unseeded exp7_D_baseline_v4 above these are three
    # independent draws of the SAME configuration on the SAME corpus.
    ("exp7_D_v4_seed101", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260730_050654_bs32_mr0.20-0.60_ps1024_seed101_best.pth", None),
    ("exp7_D_v4_seed202", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v4_20260730_050931_bs32_mr0.20-0.60_ps1024_seed202_best.pth", None),
    # Peak-weighted arm trained on the V3 corpus, so it can be compared against
    # the v3 reference checkpoint (ps1024_nhead4_true) with no corpus confound.
    # This launched twice with the SAME --seed 101, and the two runs did NOT come
    # out identical (max|dW| = 5.3e-2, best epoch 724 vs 776) -- cudnn.benchmark
    # autotuning plus AMP make GPU training nondeterministic regardless of RNG
    # seeding. That accident is useful: the pair isolates pure implementation
    # nondeterminism from seed choice, so keep BOTH.
    ("exp7_v3_pk025_r1", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260730_053708_bs32_mr0.20-0.60_ps1024_pk0.25_seed101_best.pth", None),
    ("exp7_v3_pk025_r2", f"{BASE}/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3_20260730_054128_bs32_mr0.20-0.60_ps1024_pk0.25_seed101_best.pth", None),
]


def load_backbone(ckpt_path, spectrum_length, device, nhead_override=None):
    from trainer_revised import NMRMaskedAutoencoder
    from barth_all_models_loocv import infer_mae_config

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    hp = ck.get("hyperparameters", {})
    recorded = hp.get("nhead")
    nhead = nhead_override if nhead_override is not None else (recorded if recorded else 8)
    cfg = infer_mae_config(state, int(nhead), 0.0)
    model = NMRMaskedAutoencoder(spectrum_length=spectrum_length, **cfg)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg, recorded


def embed(model, spectra, device, batch_size=16, strategy="mean_pool"):
    out = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            _, enc = model(x, mask=None)
            e = enc.mean(dim=1) if strategy == "mean_pool" else enc.reshape(enc.shape[0], -1)
            out.append(e.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def cv_scores(features, labels, n_splits, seed=42):
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
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
    parser.add_argument("--strategies", nargs="+", default=["mean_pool", "flatten"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/analysis/patch_size_comparison")
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
        print(f"\n=== {name}  n={len(labels)}  cv={cfg['n_splits']} ===", flush=True)

        for label, ckpt, nh_override in ARMS:
            if not Path(ckpt).exists():
                print(f"  {label:22s} SKIPPED (missing checkpoint)", flush=True)
                continue
            model, mcfg, recorded = load_backbone(ckpt, spectra.shape[1], device, nh_override)
            tokens = spectra.shape[1] // mcfg["patch_size"]
            for strategy in args.strategies:
                emb = embed(model, spectra, device, strategy=strategy)
                bal, auc = cv_scores(emb, labels, cfg["n_splits"], args.seed)
                rows.append(dict(dataset=name, arm=label, patch_size=mcfg["patch_size"],
                                 tokens=tokens, nhead=mcfg["nhead"], nhead_recorded=recorded,
                                 pooling=strategy, embed_dim=emb.shape[1],
                                 balanced_accuracy=bal, roc_auc=auc, n_samples=len(labels)))
                print(f"  {label:22s} ps={mcfg['patch_size']:<5d} tok={tokens:<5d} nhead={mcfg['nhead']} "
                      f"{strategy:9s} d={emb.shape[1]:<6d} bal={bal:.4f} auc={auc:.4f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "patch_size_results.csv", index=False)
    print(f"\nWrote {out_dir / 'patch_size_results.csv'}\n")
    for strategy in args.strategies:
        sub = df[df.pooling == strategy]
        if not len(sub):
            continue
        print(f"--- balanced accuracy, pooling={strategy} ---")
        print(sub.pivot_table(index="dataset", columns="arm", values="balanced_accuracy").round(3).to_string())
        print()


if __name__ == "__main__":
    main()
