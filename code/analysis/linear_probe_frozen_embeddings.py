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
def _maybe_seed(random_init: bool, seed: int) -> None:
    """Seed before constructing a randomly-initialized backbone so the control
    arm is reproducible (nn.init draws from the global RNG)."""
    if random_init:
        torch.manual_seed(seed)
        np.random.seed(seed)


def _reinit_all_parameters_(model, seed: int) -> None:
    """Reset every parameter of an already-built model, in place.

    Weight matrices (ndim >= 2) get Xavier-uniform; biases, norms and learned
    embeddings/tokens (ndim < 2) are handled by their module's own
    reset_parameters() where available, else zeroed. Buffers such as a
    non-learned positional-encoding table are left alone -- they are fixed
    functions of position, not learned knowledge.
    """
    import torch.nn as nn

    torch.manual_seed(seed)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)
        else:
            nn.init.normal_(p, mean=0.0, std=0.02)


def pool_tokens(enc: "torch.Tensor", pooling: str) -> "torch.Tensor":
    """Pool a (B, T, D) token sequence down to a per-spectrum feature vector.

    "mean_pool" averages over tokens, which discards WHERE in the spectrum each
    token came from -- and chemical-shift position is the discriminative
    information in NMR, so this costs real accuracy (experiment #4: +0.030..
    +0.129 recovered by keeping it).

    "flatten" keeps every position but gives T*D features (16384 at
    patch_size=1024), which is a lot against n=37..113 samples.

    "regional:G" is the middle ground: mean-pool within G contiguous token
    groups and concatenate, for G*D features. G=1 reproduces mean_pool and G=T
    reproduces flatten. All three are frozen, label-independent transforms, so
    they compose with a cross-validated probe without leakage.
    """
    if pooling == "mean_pool":
        return enc.mean(dim=1)
    if pooling == "flatten":
        return enc.reshape(enc.shape[0], -1)
    if pooling.startswith("regional:"):
        groups = int(pooling.split(":", 1)[1])
        b, t, d = enc.shape
        if groups > t:
            raise ValueError(f"regional groups={groups} exceeds token count {t}")
        t_use = (t // groups) * groups
        return enc[:, :t_use, :].reshape(b, groups, t_use // groups, d).mean(dim=2).reshape(b, groups * d)
    raise ValueError(f"unknown pooling {pooling!r}")


def embed_masking(ckpt_path, spectra, device, batch_size=8, random_init=False, seed=42,
                  pooling="mean_pool", nhead=None):
    from trainer_revised import NMRMaskedAutoencoder
    from barth_all_models_loocv import infer_mae_config

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    # nhead is not recoverable from tensor shapes (in_proj_weight is
    # (3*d_model, d_model) for any head count), so prefer the recorded value and
    # fall back to 8 -- what the committed eval scripts guessed -- for legacy
    # checkpoints that predate architecture recording.
    recorded = ck.get("hyperparameters", {}).get("nhead")
    nh = nhead if nhead is not None else (recorded if recorded else 8)
    _maybe_seed(random_init, seed)
    model = NMRMaskedAutoencoder(spectrum_length=spectra.shape[1], **infer_mae_config(state, int(nh), 0.0))
    # random_init: keep the architecture, load NOTHING -- a genuine
    # untrained-backbone control (patch embedding and positional encoding
    # included), unlike --reinit-unfrozen-xavier which only resets the layers
    # being fine-tuned and always keeps those two pretrained.
    if not random_init:
        model.load_state_dict(state, strict=True)
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            _, enc = model(x, mask=None)
            out.append(pool_tokens(enc, pooling).cpu().numpy())
    del model
    return np.vstack(out).astype(np.float32)


def embed_jigsaw(ckpt_path, spectra, device, batch_size=4, random_init=False, seed=42,
                 pooling="native", nhead=None):
    # jigsaw pools natively as concat-of-per-bin-size-mean-pools (768-d). The
    # per-bin-size token sequence `e` is available before that mean, so the same
    # frozen pool_tokens() transform that helped masking (experiment #4) applies
    # per bin size, then concatenates across bin sizes exactly as native does.
    # "native"/"mean_pool" are synonyms (G=1 per bin size); "flatten" concats
    # every token (large -- e.g. 4 bin sizes x ~few hundred tokens x d_model);
    # "regional:G" applies G groups within EACH bin size's tokens.
    if pooling not in ("native", "mean_pool", "flatten") and not pooling.startswith("regional:"):
        raise ValueError(f"unknown jigsaw pooling {pooling!r}")
    pool_mode = "mean_pool" if pooling in ("native", "mean_pool") else pooling
    from train_jigsaw_spectra import JigsawNMRModel

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyperparameters", {})
    bin_sizes = [int(b) for b in ck["bin_sizes"]]
    _maybe_seed(random_init, seed)
    model = JigsawNMRModel(
        spectrum_length=int(ck["spectrum_length"]), bin_sizes=bin_sizes,
        d_model=int(hp.get("d_model", 192)), nhead=int(hp.get("nhead", 6)),
        num_layers=int(hp.get("num_layers", 4)),
        dim_feedforward=int(hp.get("dim_feedforward", 768)),
        dropout=float(hp.get("dropout", 0.15)),
    )
    if not random_init:
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
                per_bin.append(pool_tokens(e, pool_mode))
            out.append(torch.cat(per_bin, dim=1).cpu().numpy())
    del model
    return np.vstack(out).astype(np.float32)


def embed_joint(ckpt_path, spectra, device, batch_size=4, random_init=False, seed=42,
                pooling="native", nhead=None):
    # joint's native path is model.encode_spectrum: per bin size (jigsaw task)
    # plus the masked-reconstruction task at mask_bin_size, each mean-pooled
    # and concatenated (960-d). Reimplemented here (rather than calling
    # encode_spectrum) so pool_tokens() can be applied to encode_bins()'s
    # per-token output before the pool, same pattern as embed_jigsaw.
    if pooling not in ("native", "mean_pool", "flatten") and not pooling.startswith("regional:"):
        raise ValueError(f"unknown joint pooling {pooling!r}")
    pool_mode = "mean_pool" if pooling in ("native", "mean_pool") else pooling
    from train_joint_ssl import build_joint_model_from_loaded_checkpoint, TASK_JIGSAW, TASK_MASKED

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _maybe_seed(random_init, seed)
    model = build_joint_model_from_loaded_checkpoint(ck, device)
    if random_init:
        # Rebuild the same architecture from scratch: re-initialize every
        # parameter in place, so nothing pretrained survives anywhere in the
        # backbone. Done post-hoc here because the joint builder loads weights
        # as part of construction.
        _reinit_all_parameters_(model, seed)
    model.eval()
    bin_sizes = [int(b) for b in ck.get("jigsaw_bin_sizes", model.jigsaw_bin_sizes)]
    spectrum_length = model.spectrum_length
    mask_bin_size = model.mask_bin_size
    out = []
    with torch.no_grad():
        for s in range(0, len(spectra), batch_size):
            x = torch.from_numpy(np.asarray(spectra[s:s + batch_size], dtype=np.float32)).to(device)
            x = x[:, :spectrum_length]
            per_task = []
            for bin_size in bin_sizes:
                trimmed = (spectrum_length // bin_size) * bin_size
                bins = x[:, :trimmed].reshape(x.shape[0], trimmed // bin_size, bin_size)
                encoded = model.encode_bins(bins, bin_size, TASK_JIGSAW, None)
                per_task.append(pool_tokens(encoded, pool_mode))
            trimmed = (spectrum_length // mask_bin_size) * mask_bin_size
            bins = x[:, :trimmed].reshape(x.shape[0], trimmed // mask_bin_size, mask_bin_size)
            no_mask = torch.zeros(bins.shape[0], bins.shape[1], dtype=torch.bool, device=x.device)
            encoded = model.encode_bins(bins, mask_bin_size, TASK_MASKED, no_mask)
            per_task.append(pool_tokens(encoded, pool_mode))
            out.append(torch.cat(per_task, dim=1).cpu().numpy())
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
