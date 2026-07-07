#!/usr/bin/env python3
"""
Compare few-shot classical ML performance with and without a DNN backbone.

Two evaluation tracks:
1) Foundation backbone features: frozen MAE embeddings -> train SVM / Logistic Regression / XGBoost on support set.
2) Direct binned spectra: raw spectrum -> bin to 1024 or 2048 using AUC per bin -> train same classifiers on support set.
3) Prototype head baseline from prototype_fewshot.py: class prototypes + cosine/mahalanobis inference on the same backbone embeddings.

For each episode, support samples are drawn per class and all remaining samples are query samples.
The same episode splits are reused across all tracks/models for fair comparison.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from trainer_revised import NMRMaskedAutoencoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


USE_IDE_CONFIG = True

IDE_CONFIG = {
    "dataset_preset": "MTBLS326",  # "MTBLS326", "MTBLS563", or None
    "data_path": "data/mtbls326/MTBLS326_aligned_spectra.npy",
    "metadata_csv": "data/mtbls326/MTBLS326_metadata_mapping.csv",
    "model_path": "models/SSL_models/combine_unique_Water_EDTA_Suppressed_20260614_084450_bs32_mr0.20_ps1024_best.pth",
    "label_column": "label", #Factor Value[IP3R expression]
    "index_column": "npy_row",
    "support_per_class": 6,
    "episodes": 100,
    "classifiers": ["svm", "logreg", "xgboost"],
    "run_foundation_backbone": True,
    "run_direct_binned": True,
    "run_prototype_baseline": True,
    "bin_counts": [1024, 2048],
    "bin_reductions": ["auc"],
    "svm_kernel": "linear",
    "svm_c": 1.0,
    "svm_class_weight": "balanced",
    "lr_penalty": "l2",
    "lr_c": 0.1,
    "lr_solver": "lbfgs",
    "lr_max_iter": 5000,
    "lr_class_weight": "balanced",
    "xgb_n_estimators": 300,
    "xgb_max_depth": 2,
    "xgb_learning_rate": 0.05,
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.1,
    "xgb_reg_lambda": 25.0,
    "xgb_n_jobs": 8,
    "prototype_distance": "mahalanobis",  # "cosine" or "mahalanobis"
    "prototype_covariance_mode": "support",  # "support" or "fewshot_global"
    "prototype_cov_reg": 1e-3,
    "prototype_feature_projection": "anova",  # "none" or "anova"
    "prototype_select_k_best": 1000,  # 500 or 1000
    "batch_size": 16,
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "nhead": 4,
    "dropout": 0.2,
    "normalize_input": "auto",  # "auto", "true", "false"
    "embedding_strategy": "flatten",  # "flatten" or "mean_pool"
    "seed": 42,
    "out_dir": "fewshot_sandbox",
}

DATASET_PRESETS = {
    "MTBLS326": {
        "data_path": "data/mtbls326/MTBLS326_aligned_spectra.npy",
        "metadata_csv": "data/mtbls326/MTBLS326_metadata_mapping.csv",
        "label_column": "Factor Value[IP3R expression]",
        "index_column": "npy_row",
    },
    "MTBLS563": {
        "data_path": "data/mtbls563/MTBLS563_aligned_spectra.npy",
        "metadata_csv": "data/mtbls563/MTBLS563_metadata_mapping.csv",
        "label_column": "Factor Value[Diagnosis]",
        "index_column": "npy_row",
    },
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_auto_bool(mode):
    m = str(mode).strip().lower()
    if m not in {"auto", "true", "false"}:
        raise ValueError("Expected one of: auto, true, false")
    return m


def parse_csv_list(text):
    if isinstance(text, list):
        return text
    parts = [x.strip() for x in str(text).split(",")]
    return [x for x in parts if x]


def parse_int_csv_list(text):
    return [int(x) for x in parse_csv_list(text)]


def parse_float_or_list(text):
    values = [float(x) for x in parse_csv_list(text)]
    return values if len(values) > 1 else values[0]


def parse_int_or_list(text):
    values = [int(x) for x in parse_csv_list(text)]
    return values if len(values) > 1 else values[0]


def parse_prototype_projection(mode):
    m = str(mode).strip().lower()
    if m not in {"none", "anova"}:
        raise ValueError("Expected one of: none, anova")
    return m


def ensure_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def format_param_value(value):
    return str(value).replace(".", "p")


def json_safe_param(value):
    if isinstance(value, (list, tuple)):
        return [json_safe_param(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_unit_range(x, tol=1e-6):
    return float(np.min(x)) >= -tol and float(np.max(x)) <= (1.0 + tol)


def normalize_batch_per_spectrum_minmax(batch):
    x = batch.astype(np.float32, copy=True)
    mins = x.min(axis=1, keepdims=True)
    maxs = x.max(axis=1, keepdims=True)
    denom = maxs - mins
    good = denom.squeeze(1) > 1e-8
    if np.any(good):
        x[good] = (x[good] - mins[good]) / denom[good]
    return x


def infer_model_dims(state_dict):
    patch_w = state_dict["encoder.patch_embedding.0.weight"]
    ff_w = state_dict["encoder.transformer.layers.0.linear1.weight"]
    patch_size = int(patch_w.shape[1])
    d_model = int(patch_w.shape[0])
    dim_feedforward = int(ff_w.shape[0])

    layer_ids = []
    prefix = "encoder.transformer.layers."
    for key in state_dict.keys():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            layer_id = suffix.split(".", 1)[0]
            if layer_id.isdigit():
                layer_ids.append(int(layer_id))
    num_layers = max(layer_ids) + 1 if layer_ids else 3

    return {
        "patch_size": patch_size,
        "d_model": d_model,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
    }


def resolve_dataset_settings(args):
    preset_name = getattr(args, "dataset_preset", None)
    if preset_name is None:
        return args
    if preset_name not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown dataset preset '{preset_name}'. Available presets: {list(DATASET_PRESETS.keys())}"
        )

    preset = DATASET_PRESETS[preset_name]
    if not getattr(args, "data_path", None):
        args.data_path = preset["data_path"]
    if not getattr(args, "metadata_csv", None):
        args.metadata_csv = preset["metadata_csv"]
    if not getattr(args, "label_column", None):
        args.label_column = preset["label_column"]
    if not getattr(args, "index_column", None):
        args.index_column = preset["index_column"]
    return args


def load_data(data_path, metadata_csv, label_col, index_col):
    spectra = np.load(data_path).astype(np.float32)
    meta = pd.read_csv(metadata_csv).copy()

    if label_col not in meta.columns:
        raise ValueError(f"Label column '{label_col}' not found in metadata CSV.")
    if index_col not in meta.columns:
        raise ValueError(f"Index column '{index_col}' not found in metadata CSV.")

    work = meta[[index_col, label_col]].copy()
    work[index_col] = pd.to_numeric(work[index_col], errors="coerce")
    work = work.dropna(subset=[index_col, label_col])
    work[index_col] = work[index_col].astype(int)
    work[label_col] = work[label_col].astype(str).str.strip()
    work = work[work[label_col] != ""]
    work = work[(work[index_col] >= 0) & (work[index_col] < spectra.shape[0])]
    work = work.sort_values(index_col).reset_index(drop=True)

    idx = work[index_col].to_numpy(dtype=int)
    x = spectra[idx]
    y_text = work[label_col].to_numpy()
    return x, y_text, work


def build_model(model_path, spectrum_length, device, nhead, dropout):
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint["model_state_dict"]
    dims = infer_model_dims(state)

    if "hyperparameters" in checkpoint and "nhead" in checkpoint["hyperparameters"]:
        nhead = int(checkpoint["hyperparameters"]["nhead"])

    if dims["d_model"] % int(nhead) != 0:
        raise ValueError(
            f"d_model={dims['d_model']} is not divisible by nhead={nhead}. Pass a valid --nhead value."
        )

    model = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length,
        patch_size=dims["patch_size"],
        d_model=dims["d_model"],
        nhead=int(nhead),
        num_layers=dims["num_layers"],
        dim_feedforward=dims["dim_feedforward"],
        dropout=float(dropout),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, dims


def encode_batch(model, xb, embedding_strategy):
    _, encoded = model(xb, mask=None)
    if embedding_strategy == "flatten":
        return encoded.reshape(encoded.shape[0], -1)
    if embedding_strategy == "mean_pool":
        return encoded.mean(dim=1)
    raise ValueError(f"Unsupported embedding_strategy: {embedding_strategy}")


def extract_embeddings_from_array(model, spectra, batch_size, device, normalize_input, embedding_strategy):
    embs = []
    auto_norm = None
    with torch.no_grad():
        for start in tqdm(range(0, spectra.shape[0], batch_size), desc="Extracting backbone embeddings"):
            batch = np.asarray(spectra[start:start + batch_size], dtype=np.float32)
            if normalize_input == "true":
                batch = normalize_batch_per_spectrum_minmax(batch)
            elif normalize_input == "auto":
                if auto_norm is None:
                    auto_norm = not is_unit_range(batch)
                    print(f"Normalization mode (backbone): auto -> {'enabled' if auto_norm else 'skipped'}")
                if auto_norm:
                    batch = normalize_batch_per_spectrum_minmax(batch)

            xb = torch.from_numpy(batch).to(device)
            emb = encode_batch(model, xb, embedding_strategy)
            embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(embs)


def bin_spectra(spectra, n_bins, reduction):
    if reduction != "auc":
        raise ValueError(f"Unsupported reduction '{reduction}'. Use 'auc'.")
    if n_bins <= 0:
        raise ValueError("n_bins must be >= 1.")

    n_samples, n_points = spectra.shape
    if n_bins > n_points:
        raise ValueError(f"n_bins={n_bins} is greater than spectrum length={n_points}.")

    if n_points % n_bins == 0:
        width = n_points // n_bins
        x = spectra.reshape(n_samples, n_bins, width)
        # AUC per bin via trapezoidal integration at unit spacing.
        trapz = getattr(np, "trapezoid", np.trapz)
        return trapz(x, dx=1.0, axis=2).astype(np.float32)

    out = np.empty((n_samples, n_bins), dtype=np.float32)
    edges = np.linspace(0, n_points, n_bins + 1, dtype=int)
    for i in range(n_bins):
        seg = spectra[:, edges[i]:edges[i + 1]]
        trapz = getattr(np, "trapezoid", np.trapz)
        out[:, i] = trapz(seg, dx=1.0, axis=1).astype(np.float32)
    return out


def build_episode_splits(y, class_names, support_per_class, n_episodes, seed):
    rng = np.random.default_rng(seed)
    class_to_indices = {c: np.where(y == c)[0] for c in range(len(class_names))}

    for c, idx in class_to_indices.items():
        if len(idx) <= support_per_class:
            raise ValueError(
                f"Class '{class_names[c]}' has {len(idx)} samples; needs > {support_per_class} for at least 1 query sample."
            )

    splits = []
    for _ in range(n_episodes):
        support_idx = []
        for c in range(len(class_names)):
            chosen = rng.choice(class_to_indices[c], size=support_per_class, replace=False)
            support_idx.extend(chosen.tolist())
        support_idx = np.array(sorted(support_idx), dtype=int)

        query_mask = np.ones(y.shape[0], dtype=bool)
        query_mask[support_idx] = False
        query_idx = np.where(query_mask)[0]
        splits.append((support_idx, query_idx))
    return splits


def estimate_inverse_cov(features, reg=1e-3):
    if features.shape[0] < 2:
        cov = np.eye(features.shape[1], dtype=np.float64)
    else:
        cov = np.cov(features, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    cov += float(reg) * np.eye(cov.shape[0], dtype=np.float64)
    return np.linalg.pinv(cov).astype(np.float32)


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def classify_cosine(query_emb, prototypes):
    q = l2_normalize_rows(query_emb)
    p = l2_normalize_rows(prototypes)
    sim = q @ p.T
    return np.argmax(sim, axis=1)


def classify_mahalanobis(query_emb, prototypes, inv_cov):
    dists = np.empty((query_emb.shape[0], prototypes.shape[0]), dtype=np.float64)
    for k in range(prototypes.shape[0]):
        diff = query_emb - prototypes[k]
        dists[:, k] = np.einsum("bi,ij,bj->b", diff, inv_cov, diff, optimize=True)
    return np.argmin(dists, axis=1)


def evaluate_prototype_head(features, y, class_names, splits, args):
    rows = []
    distance = str(args.prototype_distance).lower()
    covariance_mode = str(args.prototype_covariance_mode).lower()
    projection = str(args.prototype_feature_projection).lower()
    reg = float(args.prototype_cov_reg)

    if distance not in {"cosine", "mahalanobis"}:
        raise ValueError("prototype_distance must be one of: cosine, mahalanobis")
    if covariance_mode not in {"support", "fewshot_global"}:
        raise ValueError("prototype_covariance_mode must be one of: support, fewshot_global")
    if projection not in {"none", "anova"}:
        raise ValueError("prototype_feature_projection must be one of: none, anova")

    model_name = f"prototype_{distance}_{covariance_mode}_{projection}"

    for ep_idx, (support_idx, query_idx) in enumerate(tqdm(splits, desc="Episodes [prototype_head]"), start=1):
        y_support = y[support_idx]
        y_query = y[query_idx]
        emb_support = features[support_idx]
        emb_query = features[query_idx]

        selected_mask = None
        if projection == "anova":
            n_features = emb_support.shape[1]
            k = min(int(args.prototype_select_k_best), n_features)
            selector = SelectKBest(score_func=f_classif, k=k if k < n_features else "all")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                selector.fit(emb_support, y_support)
            selected_mask = selector.get_support()
            emb_support = emb_support[:, selected_mask]
            emb_query = emb_query[:, selected_mask]

        prototypes = np.stack(
            [emb_support[y_support == c].mean(axis=0) for c in range(len(class_names))],
            axis=0,
        )

        if distance == "cosine":
            pred = classify_cosine(emb_query, prototypes)
        else:
            if covariance_mode == "support":
                inv_cov = estimate_inverse_cov(emb_support, reg=reg)
            else:
                global_features = features if selected_mask is None else features[:, selected_mask]
                inv_cov = estimate_inverse_cov(global_features, reg=reg)
            pred = classify_mahalanobis(emb_query, prototypes, inv_cov)

        rows.append(
            {
                "feature_group": "prototype_head",
                "feature_name": "mae_embeddings",
                "classifier": model_name,
                "episode": ep_idx,
                "n_train": int(len(support_idx)),
                "n_test": int(len(query_idx)),
                "status": "ok",
                "error": "",
                "accuracy": float(accuracy_score(y_query, pred)),
                "macro_f1": float(f1_score(y_query, pred, average="macro", zero_division=0)),
                "macro_precision": float(precision_score(y_query, pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_query, pred, average="macro", zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def make_classifier_variants(name, n_classes, seed, args):
    key = str(name).strip().lower()

    if key == "svm":
        variants = []
        for c_value in ensure_list(args.svm_c):
            label = f"svm_C{format_param_value(c_value)}"
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel=str(args.svm_kernel),
                            C=float(c_value),
                            class_weight=None if str(args.svm_class_weight).lower() == "none" else str(args.svm_class_weight),
                        ),
                    ),
                ]
            )
            variants.append((label, model))
        return variants

    if key in {"logreg", "logistic", "logistic_regression"}:
        variants = []
        for c_value in ensure_list(args.lr_c):
            label = f"logreg_C{format_param_value(c_value)}"
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty=str(args.lr_penalty),
                            C=float(c_value),
                            solver=str(args.lr_solver),
                            max_iter=int(args.lr_max_iter),
                            class_weight=None if str(args.lr_class_weight).lower() == "none" else str(args.lr_class_weight),
                        ),
                    ),
                ]
            )
            variants.append((label, model))
        return variants

    if key in {"xgboost", "xgb"}:
        if XGBClassifier is None:
            return [("xgboost", None)]

        variants = []
        for depth in ensure_list(args.xgb_max_depth):
            label = f"xgboost_depth{int(depth)}"
            params = {
                "n_estimators": int(args.xgb_n_estimators),
                "max_depth": int(depth),
                "learning_rate": float(args.xgb_learning_rate),
                "subsample": float(args.xgb_subsample),
                "colsample_bytree": float(args.xgb_colsample_bytree),
                "reg_lambda": float(args.xgb_reg_lambda),
                "tree_method": "hist",
                "random_state": seed,
                "n_jobs": int(args.xgb_n_jobs),
                "verbosity": 0,
            }
            if n_classes > 2:
                params["objective"] = "multi:softprob"
                params["num_class"] = int(n_classes)
                params["eval_metric"] = "mlogloss"
            else:
                params["objective"] = "binary:logistic"
                params["eval_metric"] = "logloss"
            variants.append((label, XGBClassifier(**params)))
        return variants

    raise ValueError(f"Unsupported classifier '{name}'.")


def evaluate_feature_set(features, y, class_names, splits, classifiers, seed, feature_group, feature_name, args):
    rows = []

    for ep_idx, (support_idx, query_idx) in enumerate(tqdm(splits, desc=f"Episodes [{feature_name}]"), start=1):
        x_train = features[support_idx]
        y_train = y[support_idx]
        x_test = features[query_idx]
        y_test = y[query_idx]

        for clf_name in classifiers:
            clf_key = str(clf_name).strip().lower()
            variants = make_classifier_variants(clf_key, n_classes=len(class_names), seed=seed + ep_idx, args=args)

            for clf_label, model in variants:
                if model is None:
                    rows.append(
                        {
                            "feature_group": feature_group,
                            "feature_name": feature_name,
                            "classifier": clf_label,
                            "episode": ep_idx,
                            "n_train": int(x_train.shape[0]),
                            "n_test": int(x_test.shape[0]),
                            "status": "skipped",
                            "error": "xgboost is not installed",
                        }
                    )
                    continue

                try:
                    model.fit(x_train, y_train)
                    pred = model.predict(x_test)

                    rows.append(
                        {
                            "feature_group": feature_group,
                            "feature_name": feature_name,
                            "classifier": clf_label,
                            "episode": ep_idx,
                            "n_train": int(x_train.shape[0]),
                            "n_test": int(x_test.shape[0]),
                            "status": "ok",
                            "error": "",
                            "accuracy": float(accuracy_score(y_test, pred)),
                            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                            "macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
                            "macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "feature_group": feature_group,
                            "feature_name": feature_name,
                            "classifier": clf_label,
                            "episode": ep_idx,
                            "n_train": int(x_train.shape[0]),
                            "n_test": int(x_test.shape[0]),
                            "status": "error",
                            "error": str(exc),
                        }
                    )

    return pd.DataFrame(rows)


def summarize_results(df):
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    grouped = (
        ok.groupby(["feature_group", "feature_name", "classifier"], as_index=False)
        .agg(
            n_episodes=("episode", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_precision_mean=("macro_precision", "mean"),
            macro_recall_mean=("macro_recall", "mean"),
        )
        .sort_values(["feature_group", "feature_name", "classifier"])
        .reset_index(drop=True)
    )
    return grouped


def save_best_hyperparams(summary_df, out_dir):
    if summary_df.empty:
        return None, []

    best = (
        summary_df.sort_values(["feature_group", "feature_name", "accuracy_mean"], ascending=[True, True, False])
        .groupby(["feature_group", "feature_name"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    path = Path(out_dir) / "fewshot_ml_best_hyperparams.csv"
    best.to_csv(path, index=False)
    return path, best.to_dict(orient="records")


def create_plots(results_df, summary_df, out_dir):
    out_dir = Path(out_dir)
    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty or summary_df.empty:
        return []

    saved = []
    summary_plot = summary_df.copy()
    summary_plot["method"] = (
        summary_plot["feature_group"]
        + " | "
        + summary_plot["feature_name"]
        + " | "
        + summary_plot["classifier"]
    )
    summary_plot = summary_plot.sort_values("accuracy_mean", ascending=False)

    # Plot 1: mean accuracy +/- std.
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(summary_plot))
    ax.bar(
        x,
        summary_plot["accuracy_mean"].to_numpy(),
        yerr=summary_plot["accuracy_std"].fillna(0.0).to_numpy(),
        capsize=3,
    )
    ax.set_title("Few-shot Comparison: Accuracy (mean ± std)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_plot["method"], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p1 = out_dir / "comparison_accuracy_bar.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(str(p1))

    # Plot 2: mean macro-F1 +/- std.
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x,
        summary_plot["macro_f1_mean"].to_numpy(),
        yerr=summary_plot["macro_f1_std"].fillna(0.0).to_numpy(),
        capsize=3,
        color="#2a9d8f",
    )
    ax.set_title("Few-shot Comparison: Macro-F1 (mean ± std)")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_plot["method"], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p2 = out_dir / "comparison_macro_f1_bar.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(str(p2))

    # Plot 3: episode-level accuracy distribution.
    ok["method"] = ok["feature_group"] + " | " + ok["feature_name"] + " | " + ok["classifier"]
    method_order = summary_plot["method"].tolist()
    box_data = [ok.loc[ok["method"] == m, "accuracy"].to_numpy() for m in method_order]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(box_data, labels=method_order, showfliers=False)
    ax.set_title("Episode Accuracy Distribution")
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(method_order, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p3 = out_dir / "comparison_accuracy_boxplot.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved.append(str(p3))

    # Plot 4: per-episode mean accuracy by feature group.
    group_episode = (
        ok.groupby(["feature_group", "episode"], as_index=False)["accuracy"].mean()
        .sort_values(["feature_group", "episode"])
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for grp, gdf in group_episode.groupby("feature_group"):
        ax.plot(gdf["episode"], gdf["accuracy"], label=grp, linewidth=1.5)
    ax.set_title("Per-Episode Mean Accuracy by Feature Group")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    p4 = out_dir / "comparison_episode_trend_by_group.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    saved.append(str(p4))

    return saved


def build_parser():
    parser = argparse.ArgumentParser(description="Few-shot ML comparison: backbone features vs direct binned spectra.")
    parser.add_argument("--dataset-preset", choices=list(DATASET_PRESETS.keys()), default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--index-column", type=str, default=None)
    parser.add_argument("--support-per-class", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--classifiers", type=parse_csv_list, default=["svm", "logreg", "xgboost"])
    parser.add_argument("--run-foundation-backbone", type=str2bool, default=True)
    parser.add_argument("--run-direct-binned", type=str2bool, default=True)
    parser.add_argument("--run-prototype-baseline", type=str2bool, default=True)
    parser.add_argument("--bin-counts", type=parse_int_csv_list, default=[1024, 2048])
    parser.add_argument("--bin-reductions", type=parse_csv_list, default=["auc"])
    parser.add_argument("--svm-kernel", type=str, default="linear")
    parser.add_argument("--svm-c", type=parse_float_or_list, default=1.0)
    parser.add_argument("--svm-class-weight", type=str, default="balanced")
    parser.add_argument("--lr-penalty", type=str, default="l2")
    parser.add_argument("--lr-c", type=parse_float_or_list, default=0.1)
    parser.add_argument("--lr-solver", type=str, default="lbfgs")
    parser.add_argument("--lr-max-iter", type=int, default=5000)
    parser.add_argument("--lr-class-weight", type=str, default="balanced")
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=parse_int_or_list, default=1)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.1)
    parser.add_argument("--xgb-reg-lambda", type=float, default=25.0)
    parser.add_argument("--xgb-n-jobs", type=int, default=8)
    parser.add_argument("--prototype-distance", choices=["cosine", "mahalanobis"], default="mahalanobis")
    parser.add_argument("--prototype-covariance-mode", choices=["support", "fewshot_global"], default="support")
    parser.add_argument("--prototype-cov-reg", type=float, default=1e-3)
    parser.add_argument("--prototype-feature-projection", type=parse_prototype_projection, default="anova")
    parser.add_argument("--prototype-select-k-best", type=int, choices=[500, 1000], default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--normalize-input", type=parse_auto_bool, default="auto")
    parser.add_argument("--embedding-strategy", choices=["flatten", "mean_pool"], default="flatten")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/fewshot/prototype_default")
    return parser


def args_from_ide_config():
    parser = build_parser()
    args = parser.parse_args([])
    for k, v in IDE_CONFIG.items():
        setattr(args, k, v)
    return args


def run_comparison(args):
    args = resolve_dataset_settings(args)

    if not args.data_path or not args.metadata_csv or not args.label_column or not args.index_column:
        raise ValueError(
            "Missing dataset settings. Set --dataset-preset or provide --data-path, --metadata-csv, "
            "--label-column, and --index-column."
        )
    run_backbone_required = bool(args.run_foundation_backbone or args.run_prototype_baseline)
    if run_backbone_required and not args.model_path:
        raise ValueError("Backbone-based run requested but model_path is missing.")
    if not args.run_foundation_backbone and not args.run_direct_binned and not args.run_prototype_baseline:
        raise ValueError("Enable at least one of: run_foundation_backbone, run_direct_binned, run_prototype_baseline.")
    if any(str(r).lower() != "auc" for r in args.bin_reductions):
        raise ValueError("bin_reductions only supports 'auc' in this script.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spectra + metadata...")
    x, y_text, used_meta = load_data(args.data_path, args.metadata_csv, args.label_column, args.index_column)
    print(f"Loaded {x.shape[0]} samples, spectrum length {x.shape[1]}.")

    unique_labels = pd.unique(y_text)
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    int_to_label = {i: lab for lab, i in label_to_int.items()}
    y = np.array([label_to_int[v] for v in y_text], dtype=int)
    class_names = [int_to_label[i] for i in range(len(int_to_label))]
    class_counts = {name: int(np.sum(y == i)) for i, name in enumerate(class_names)}
    print(f"Classes: {class_counts}")

    splits = build_episode_splits(
        y=y,
        class_names=class_names,
        support_per_class=args.support_per_class,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    print(f"Prepared {len(splits)} shared episode splits.")

    all_results = []
    model_dims = None
    emb = None

    if run_backbone_required:
        device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        print(f"Using device for backbone embeddings: {device}")

        model, model_dims = build_model(
            args.model_path,
            spectrum_length=x.shape[1],
            device=device,
            nhead=args.nhead,
            dropout=args.dropout,
        )
        print(f"Backbone model dims: {model_dims}")

        emb = extract_embeddings_from_array(
            model=model,
            spectra=x,
            batch_size=args.batch_size,
            device=device,
            normalize_input=args.normalize_input,
            embedding_strategy=args.embedding_strategy,
        )
        np.save(out_dir / "comparison_embeddings.npy", emb)

    if args.run_foundation_backbone:
        df_backbone = evaluate_feature_set(
            features=emb,
            y=y,
            class_names=class_names,
            splits=splits,
            classifiers=args.classifiers,
            seed=args.seed,
            feature_group="foundation_backbone",
            feature_name=f"mae_{args.embedding_strategy}",
            args=args,
        )
        all_results.append(df_backbone)

    if args.run_prototype_baseline:
        df_prototype = evaluate_prototype_head(
            features=emb,
            y=y,
            class_names=class_names,
            splits=splits,
            args=args,
        )
        all_results.append(df_prototype)

    if args.run_direct_binned:
        for n_bins in args.bin_counts:
            for reduction in args.bin_reductions:
                print(f"Binning spectra: n_bins={n_bins}, reduction={reduction}")
                xb = bin_spectra(x, n_bins=n_bins, reduction=reduction)
                np.save(out_dir / f"binned_{n_bins}_{reduction}.npy", xb)

                df_binned = evaluate_feature_set(
                    features=xb,
                    y=y,
                    class_names=class_names,
                    splits=splits,
                    classifiers=args.classifiers,
                    seed=args.seed,
                    feature_group="direct_binned",
                    feature_name=f"bins{n_bins}_{reduction}",
                    args=args,
                )
                all_results.append(df_binned)

    if not all_results:
        raise RuntimeError("No results were produced.")

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_results(results_df)

    results_path = out_dir / "fewshot_ml_comparison_episode_metrics.csv"
    summary_path = out_dir / "fewshot_ml_comparison_summary.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    best_hyperparams_path, best_hyperparams = save_best_hyperparams(summary_df, out_dir)
    plot_paths = create_plots(results_df, summary_df, out_dir)

    skipped_or_error = results_df[results_df["status"] != "ok"][["feature_group", "feature_name", "classifier", "status", "error"]]
    skipped_records = skipped_or_error.to_dict(orient="records")
    if len(skipped_records) > 50:
        skipped_records = skipped_records[:50]

    summary_json = {
        "n_samples": int(x.shape[0]),
        "spectrum_length": int(x.shape[1]),
        "class_counts": class_counts,
        "class_order": class_names,
        "support_per_class": int(args.support_per_class),
        "episodes": int(args.episodes),
        "classifiers_requested": args.classifiers,
        "xgboost_available": bool(XGBClassifier is not None),
        "run_foundation_backbone": bool(args.run_foundation_backbone),
        "run_direct_binned": bool(args.run_direct_binned),
        "run_prototype_baseline": bool(args.run_prototype_baseline),
        "bin_counts": [int(v) for v in args.bin_counts],
        "bin_reductions": list(args.bin_reductions),
        "svm_params": {
            "kernel": str(args.svm_kernel),
            "C": json_safe_param(args.svm_c),
            "class_weight": str(args.svm_class_weight),
        },
        "logreg_params": {
            "penalty": str(args.lr_penalty),
            "C": json_safe_param(args.lr_c),
            "solver": str(args.lr_solver),
            "max_iter": int(args.lr_max_iter),
            "class_weight": str(args.lr_class_weight),
        },
        "xgboost_params": {
            "n_estimators": int(args.xgb_n_estimators),
            "max_depth": json_safe_param(args.xgb_max_depth),
            "learning_rate": float(args.xgb_learning_rate),
            "subsample": float(args.xgb_subsample),
            "colsample_bytree": float(args.xgb_colsample_bytree),
            "reg_lambda": float(args.xgb_reg_lambda),
            "n_jobs": int(args.xgb_n_jobs),
        },
        "prototype_params": {
            "distance": str(args.prototype_distance),
            "covariance_mode": str(args.prototype_covariance_mode),
            "cov_reg": float(args.prototype_cov_reg),
            "feature_projection": str(args.prototype_feature_projection),
            "select_k_best": int(args.prototype_select_k_best),
        },
        "embedding_strategy": args.embedding_strategy,
        "model_dims": model_dims,
        "results_csv": str(results_path),
        "summary_csv": str(summary_path),
        "best_hyperparams_csv": None if best_hyperparams_path is None else str(best_hyperparams_path),
        "best_hyperparams": best_hyperparams,
        "plots": plot_paths,
        "difference_note": {
            "fewshot_ml_foundation_backbone": "MAE embeddings used as fixed features for classical ML heads (SVM/LR/XGBoost).",
            "prototype_fewshot_style": "MAE embeddings classified by class prototypes with cosine/mahalanobis distance and optional ANOVA support-only feature selection.",
        },
        "aggregate": summary_df.to_dict(orient="records"),
        "non_ok_examples": skipped_records,
    }

    with open(out_dir / "fewshot_ml_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    used_meta.to_csv(out_dir / "fewshot_ml_used_metadata_rows.csv", index=False)

    print("\nDone.")
    print(f"Saved episode metrics: {results_path}")
    print(f"Saved aggregate summary: {summary_path}")
    if best_hyperparams_path is not None:
        print(f"Saved best hyperparameters: {best_hyperparams_path}")
    print(f"Saved JSON summary: {out_dir / 'fewshot_ml_comparison_summary.json'}")
    if plot_paths:
        print("Saved plots:")
        for p in plot_paths:
            print(f"  - {p}")
    if XGBClassifier is None and any(str(c).lower() in {"xgboost", "xgb"} for c in args.classifiers):
        print("Warning: xgboost is not installed. XGBoost rows were skipped.")
    return summary_json


def main():
    if USE_IDE_CONFIG:
        args = args_from_ide_config()
    else:
        args = build_parser().parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
