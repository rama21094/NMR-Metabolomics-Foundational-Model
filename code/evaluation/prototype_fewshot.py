#!/usr/bin/env python3
"""
One-shot / few-shot prototype classification using frozen MAE embeddings.

Usage modes:
1) IDE mode: edit IDE_CONFIG below and run this file.
2) CLI mode: set USE_IDE_CONFIG = False and pass arguments.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from trainer_revised import NMRMaskedAutoencoder


USE_IDE_CONFIG = True

IDE_CONFIG = {
    "dataset_preset": "MTBLS563",  # "MTBLS326", "MTBLS563", or None
    "data_path": "data/mtbls563/MTBLS563_aligned_spectra.npy",
    "metadata_csv": "data/mtbls563/MTBLS563_metadata_mapping.csv",
    "model_path": "models/SSL_models/aligned_nmr_spectra_128K_Plasma_WS625to680Zero_merged2_20260520_014209_bs16_mr0.35_ps1024_best.pth",
    "label_column": "Factor Value[Diagnosis]",
    "index_column": "npy_row",
    "support_per_class": 25,
    "episodes": 200,
    "distance": "mahalanobis",  # "cosine" or "mahalanobis"
    "covariance_mode": "pretrain",  # "pretrain", "fewshot_global", "support"
    "cov_reg": 1e-3,
    "batch_size": 16,
    "device": "cuda",
    "nhead": 4,
    "dropout": 0.2,
    "normalize_input": "auto",  # "auto", "true", "false"
    "embedding_strategy": "flatten",  # "flatten" or "mean_pool"
    "feature_projection": "pca",  # "anova" or "pca"
    "select_k_best": 1000,  # use 500 or 1000
    "pca_components": 512,  # <= 0 to disable when feature_projection="pca"
    "pca_fit_source": "pretrain_corpus",  # "pretrain_corpus" or "fewshot"
    "pretrain_corpus_paths": [
        "data/combined/combined_unique.npy",
    ],
    "pretrain_max_samples": 8892,
    "pretrain_batch_size": 16,
    "pretrain_stats_path": "results/fewshot/prototype_default/pretrain_precision_pca512.npz",
    "use_cached_pretrain_stats": True,
    "seed": 42,
    "out_dir": "results/fewshot/prototype_default",
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


def parse_feature_projection(mode):
    m = str(mode).strip().lower()
    if m not in {"anova", "pca"}:
        raise ValueError("Expected one of: anova, pca")
    return m


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
    _, encoded = model(xb, mask=None)  # [B, n_patches, d_model]
    if embedding_strategy == "flatten":
        return encoded.reshape(encoded.shape[0], -1)
    if embedding_strategy == "mean_pool":
        return encoded.mean(dim=1)
    raise ValueError(f"Unsupported embedding_strategy: {embedding_strategy}")


def extract_embeddings_from_array(model, spectra, batch_size, device, normalize_input, embedding_strategy):
    embs = []
    auto_norm = None
    with torch.no_grad():
        for start in tqdm(range(0, spectra.shape[0], batch_size), desc="Extracting embeddings"):
            batch = np.asarray(spectra[start:start + batch_size], dtype=np.float32)
            if normalize_input == "true":
                batch = normalize_batch_per_spectrum_minmax(batch)
            elif normalize_input == "auto":
                if auto_norm is None:
                    auto_norm = not is_unit_range(batch)
                    print(f"Normalization mode (small dataset): auto -> {'enabled' if auto_norm else 'skipped'}")
                if auto_norm:
                    batch = normalize_batch_per_spectrum_minmax(batch)

            xb = torch.from_numpy(batch).to(device)
            emb = encode_batch(model, xb, embedding_strategy)
            embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(embs)


def sample_indices_per_path(paths, max_total_samples, seed):
    rng = np.random.default_rng(seed)
    sizes = []
    for path in paths:
        arr = np.load(path, mmap_mode="r")
        sizes.append(int(arr.shape[0]))

    total = sum(sizes)
    if total == 0:
        return {p: np.array([], dtype=int) for p in paths}, sizes, 0

    if max_total_samples is None or int(max_total_samples) <= 0 or int(max_total_samples) >= total:
        selections = {p: np.arange(n, dtype=int) for p, n in zip(paths, sizes)}
        return selections, sizes, total

    want = int(max_total_samples)
    flat_choices = np.sort(rng.choice(total, size=want, replace=False))

    selections = {}
    cumsum = np.cumsum([0] + sizes)
    for i, path in enumerate(paths):
        lo, hi = cumsum[i], cumsum[i + 1]
        local = flat_choices[(flat_choices >= lo) & (flat_choices < hi)] - lo
        selections[path] = local.astype(int)
    return selections, sizes, want


def extract_embeddings_from_path_indices(
    model,
    data_path,
    indices,
    batch_size,
    device,
    normalize_input,
    embedding_strategy,
):
    arr = np.load(data_path, mmap_mode="r")
    if indices is None:
        indices = np.arange(arr.shape[0], dtype=int)
    if len(indices) == 0:
        out_dim = model.encoder.n_patches * model.encoder.d_model if embedding_strategy == "flatten" else model.encoder.d_model
        return np.empty((0, out_dim), dtype=np.float32)

    indices = np.asarray(indices, dtype=int)
    embs = []
    auto_norm = None
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc=f"Embeddings {Path(data_path).name}"):
            chunk_idx = indices[start:start + batch_size]
            batch = np.asarray(arr[chunk_idx], dtype=np.float32)

            if normalize_input == "true":
                batch = normalize_batch_per_spectrum_minmax(batch)
            elif normalize_input == "auto":
                if auto_norm is None:
                    auto_norm = not is_unit_range(batch)
                    print(
                        f"Normalization mode ({Path(data_path).name}): auto -> "
                        f"{'enabled' if auto_norm else 'skipped'}"
                    )
                if auto_norm:
                    batch = normalize_batch_per_spectrum_minmax(batch)

            xb = torch.from_numpy(batch).to(device)
            emb = encode_batch(model, xb, embedding_strategy)
            embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(embs)


def fit_incremental_pca(x, n_components, batch_size):
    n_components = int(n_components)
    if n_components <= 0 or n_components >= x.shape[1]:
        return None
    if x.shape[0] < n_components:
        raise ValueError(
            f"PCA requires at least n_components samples. Got n_samples={x.shape[0]}, n_components={n_components}."
        )

    ipca = IncrementalPCA(n_components=n_components)
    fit_batch = max(int(batch_size), n_components)
    fit_ranges = []
    start = 0
    n = x.shape[0]
    while start < n:
        end = min(start + fit_batch, n)
        if n - end > 0 and n - end < n_components:
            end = n
        fit_ranges.append((start, end))
        start = end

    for start, end in tqdm(fit_ranges, desc="PCA fit"):
        ipca.partial_fit(x[start:end])

    transformed = []
    for start in tqdm(range(0, x.shape[0], fit_batch), desc="PCA transform"):
        transformed.append(ipca.transform(x[start:start + fit_batch]).astype(np.float32))

    x_proj = np.vstack(transformed)
    pca_state = {
        "components": ipca.components_.astype(np.float32),
        "mean": ipca.mean_.astype(np.float32),
        "n_components": int(ipca.n_components_),
        "explained_variance_ratio_sum": float(np.sum(ipca.explained_variance_ratio_)),
    }
    return x_proj, pca_state


def transform_with_pca_state(x, pca_state):
    components = pca_state["components"]
    mean = pca_state["mean"]
    return ((x - mean) @ components.T).astype(np.float32)


def estimate_inverse_cov(features, reg=1e-3):
    if features.shape[0] < 2:
        cov = np.eye(features.shape[1], dtype=np.float64)
    else:
        cov = np.cov(features, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    cov += reg * np.eye(cov.shape[0], dtype=np.float64)
    return np.linalg.pinv(cov).astype(np.float32)


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def classify_cosine(query_emb, prototypes):
    q = l2_normalize_rows(query_emb)
    p = l2_normalize_rows(prototypes)
    sim = q @ p.T
    return np.argmax(sim, axis=1), sim


def classify_mahalanobis(query_emb, prototypes, inv_cov):
    dists = np.empty((query_emb.shape[0], prototypes.shape[0]), dtype=np.float64)
    for k in range(prototypes.shape[0]):
        diff = query_emb - prototypes[k]
        dists[:, k] = np.einsum("bi,ij,bj->b", diff, inv_cov, diff, optimize=True)
    return np.argmin(dists, axis=1), dists


def run_episodes(
    embeddings,
    y,
    class_names,
    support_per_class,
    n_episodes,
    distance,
    covariance_mode,
    cov_reg,
    seed,
    feature_projection,
    select_k_best,
    pretrain_inv_cov=None,
):
    rng = np.random.default_rng(seed)
    class_to_indices = {c: np.where(y == c)[0] for c in range(len(class_names))}

    for c, idx in class_to_indices.items():
        if len(idx) <= support_per_class:
            raise ValueError(
                f"Class '{class_names[c]}' has {len(idx)} samples; needs > {support_per_class} to keep at least 1 query sample."
            )

    if distance == "mahalanobis" and covariance_mode == "pretrain" and pretrain_inv_cov is None:
        raise ValueError("covariance_mode='pretrain' requires a precomputed precision matrix.")

    episode_rows = []
    cm_total = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for ep in range(n_episodes):
        support_idx = []
        for c in range(len(class_names)):
            chosen = rng.choice(class_to_indices[c], size=support_per_class, replace=False)
            support_idx.extend(chosen.tolist())
        support_idx = np.array(sorted(support_idx), dtype=int)

        query_mask = np.ones(embeddings.shape[0], dtype=bool)
        query_mask[support_idx] = False
        query_idx = np.where(query_mask)[0]

        y_support = y[support_idx]
        y_query = y[query_idx]
        emb_support = embeddings[support_idx]
        emb_query = embeddings[query_idx]

        selected_mask = None
        if feature_projection == "anova":
            n_features = emb_support.shape[1]
            k = min(int(select_k_best), n_features)
            if k <= 0:
                raise ValueError(f"select_k_best must be >= 1, got {select_k_best}")

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
            pred, _ = classify_cosine(emb_query, prototypes)
        else:
            if covariance_mode == "support":
                inv_cov = estimate_inverse_cov(emb_support, reg=cov_reg)
            elif covariance_mode == "fewshot_global":
                global_features = embeddings if selected_mask is None else embeddings[:, selected_mask]
                inv_cov = estimate_inverse_cov(global_features, reg=cov_reg)
            else:
                inv_cov = pretrain_inv_cov if selected_mask is None else pretrain_inv_cov[np.ix_(selected_mask, selected_mask)]
            pred, _ = classify_mahalanobis(emb_query, prototypes, inv_cov)

        acc = accuracy_score(y_query, pred)
        f1m = f1_score(y_query, pred, average="macro", zero_division=0)
        precm = precision_score(y_query, pred, average="macro", zero_division=0)
        recm = recall_score(y_query, pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_query, pred, labels=np.arange(len(class_names)))
        cm_total += cm

        episode_rows.append(
            {
                "episode": ep + 1,
                "n_query": int(len(query_idx)),
                "accuracy": float(acc),
                "macro_f1": float(f1m),
                "macro_precision": float(precm),
                "macro_recall": float(recm),
            }
        )

    return pd.DataFrame(episode_rows), cm_total


def save_pretrain_stats(path, inv_cov, paths, n_samples, embedding_strategy, cov_reg, pca_state=None):
    out = {
        "inv_cov": inv_cov.astype(np.float32),
        "source_paths": np.array(paths, dtype=object),
        "n_source_samples": np.array([int(n_samples)], dtype=np.int64),
        "embedding_strategy": np.array([embedding_strategy], dtype=object),
        "cov_reg": np.array([float(cov_reg)], dtype=np.float64),
    }
    if pca_state is not None:
        out["pca_components"] = pca_state["components"]
        out["pca_mean"] = pca_state["mean"]
        out["pca_n_components"] = np.array([int(pca_state["n_components"])], dtype=np.int64)
        out["pca_explained_variance_ratio_sum"] = np.array(
            [float(pca_state["explained_variance_ratio_sum"])], dtype=np.float64
        )
    np.savez(path, **out)


def load_pretrain_stats(path):
    z = np.load(path, allow_pickle=True)
    inv_cov = z["inv_cov"].astype(np.float32)
    pca_state = None
    if "pca_components" in z and "pca_mean" in z:
        pca_state = {
            "components": z["pca_components"].astype(np.float32),
            "mean": z["pca_mean"].astype(np.float32),
            "n_components": int(z["pca_n_components"][0]) if "pca_n_components" in z else int(z["pca_components"].shape[0]),
            "explained_variance_ratio_sum": float(z["pca_explained_variance_ratio_sum"][0]) if "pca_explained_variance_ratio_sum" in z else None,
        }
    meta = {
        "source_paths": z["source_paths"].tolist() if "source_paths" in z else [],
        "n_source_samples": int(z["n_source_samples"][0]) if "n_source_samples" in z else None,
        "embedding_strategy": str(z["embedding_strategy"][0]) if "embedding_strategy" in z else None,
        "cov_reg": float(z["cov_reg"][0]) if "cov_reg" in z else None,
    }
    return inv_cov, pca_state, meta


def compute_or_load_pretrain_stats(args, model, device, out_dir, pca_state=None, save_pca_state=False):
    stats_path = Path(args.pretrain_stats_path) if args.pretrain_stats_path else (out_dir / "pretrain_precision.npz")
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_cached_pretrain_stats and stats_path.exists():
        print(f"Loading cached pretrain stats: {stats_path}")
        return (*load_pretrain_stats(stats_path), stats_path)

    if not args.pretrain_corpus_paths:
        raise ValueError("pretrain_corpus_paths is empty; cannot compute pretrain precision matrix.")

    selections, sizes, n_selected = sample_indices_per_path(
        args.pretrain_corpus_paths,
        args.pretrain_max_samples,
        args.seed,
    )
    print("Pretrain corpus sizes:", {p: n for p, n in zip(args.pretrain_corpus_paths, sizes)})
    print(f"Selected {n_selected} total samples for pretrain stats.")

    emb_list = []
    for path in args.pretrain_corpus_paths:
        idx = selections[path]
        emb_path = extract_embeddings_from_path_indices(
            model=model,
            data_path=path,
            indices=idx,
            batch_size=args.pretrain_batch_size,
            device=device,
            normalize_input=args.normalize_input,
            embedding_strategy=args.embedding_strategy,
        )
        emb_list.append(emb_path)

    pretrain_emb = np.vstack(emb_list)
    print(f"Pretrain embedding matrix shape: {pretrain_emb.shape}")

    pca_state_out = pca_state
    pretrain_features = pretrain_emb
    if args.feature_projection == "pca":
        if args.pca_fit_source == "pretrain_corpus":
            pca_result = fit_incremental_pca(pretrain_emb, args.pca_components, args.pretrain_batch_size)
            if pca_result is None:
                raise ValueError(
                    f"Invalid pca_components={args.pca_components} for pretrain feature dimension {pretrain_emb.shape[1]}."
                )
            pretrain_features, pca_state_out = pca_result
            print(
                f"PCA fitted on pretrain corpus: {pretrain_emb.shape[1]} -> {pretrain_features.shape[1]} "
                f"(explained variance sum: {pca_state_out['explained_variance_ratio_sum']:.4f})"
            )
        else:
            if pca_state is None:
                raise ValueError("pca_fit_source='fewshot' requires a fitted few-shot PCA state.")
            pretrain_features = transform_with_pca_state(pretrain_emb, pca_state)

    inv_cov = estimate_inverse_cov(pretrain_features, reg=args.cov_reg)
    meta = {
        "source_paths": args.pretrain_corpus_paths,
        "n_source_samples": int(pretrain_features.shape[0]),
        "embedding_strategy": args.embedding_strategy,
        "cov_reg": args.cov_reg,
    }

    save_pretrain_stats(
        stats_path,
        inv_cov=inv_cov,
        paths=args.pretrain_corpus_paths,
        n_samples=pretrain_features.shape[0],
        embedding_strategy=args.embedding_strategy,
        cov_reg=args.cov_reg,
        pca_state=pca_state_out if save_pca_state else None,
    )
    print(f"Saved pretrain stats: {stats_path}")
    return inv_cov, pca_state_out, meta, stats_path


def build_parser():
    parser = argparse.ArgumentParser(description="Prototype-based one/few-shot classification from frozen MAE embeddings.")
    parser.add_argument("--dataset-preset", choices=list(DATASET_PRESETS.keys()), default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--index-column", type=str, default=None)
    parser.add_argument("--support-per-class", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--distance", choices=["cosine", "mahalanobis"], default="cosine")
    parser.add_argument("--covariance-mode", choices=["pretrain", "fewshot_global", "support"], default="pretrain")
    parser.add_argument("--cov-reg", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--normalize-input", type=parse_auto_bool, default="auto")
    parser.add_argument("--embedding-strategy", choices=["flatten", "mean_pool"], default="flatten")
    parser.add_argument("--feature-projection", type=parse_feature_projection, default="anova")
    parser.add_argument("--select-k-best", type=int, choices=[500, 1000], default=1000)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--pca-fit-source", choices=["pretrain_corpus", "fewshot"], default="pretrain_corpus")
    parser.add_argument("--pretrain-corpus-paths", nargs="*", default=[])
    parser.add_argument("--pretrain-max-samples", type=int, default=8000)
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--pretrain-stats-path", type=str, default=None)
    parser.add_argument("--use-cached-pretrain-stats", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/fewshot/prototype_default")
    return parser


def args_from_ide_config():
    parser = build_parser()
    args = parser.parse_args([])
    for k, v in IDE_CONFIG.items():
        setattr(args, k, v)
    return args


def main():
    if USE_IDE_CONFIG:
        args = args_from_ide_config()
    else:
        args = build_parser().parse_args()

    args = resolve_dataset_settings(args)

    if not args.data_path or not args.metadata_csv or not args.label_column or not args.index_column:
        raise ValueError(
            "Missing dataset settings. Set --dataset-preset or provide --data-path, --metadata-csv, "
            "--label-column, and --index-column."
        )
    if not args.model_path:
        raise ValueError("Missing model path. Set model_path in IDE_CONFIG or pass --model-path.")
    if args.embedding_strategy != "flatten":
        print("Warning: embedding_strategy is not 'flatten'.")

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

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    model, dims = build_model(
        args.model_path,
        spectrum_length=x.shape[1],
        device=device,
        nhead=args.nhead,
        dropout=args.dropout,
    )
    print(f"Model dims: {dims}")

    emb_raw = extract_embeddings_from_array(
        model=model,
        spectra=x,
        batch_size=args.batch_size,
        device=device,
        normalize_input=args.normalize_input,
        embedding_strategy=args.embedding_strategy,
    )
    np.save(out_dir / "embeddings_raw.npy", emb_raw)

    pretrain_inv_cov = None
    pretrain_pca_state = None
    pretrain_meta = None
    pretrain_stats_path = None

    emb = emb_raw
    pca_state_used = None
    pca_source_used = None

    if args.feature_projection == "pca":
        if args.pca_fit_source == "pretrain_corpus":
            pretrain_inv_cov, pretrain_pca_state, pretrain_meta, pretrain_stats_path = compute_or_load_pretrain_stats(
                args=args,
                model=model,
                device=device,
                out_dir=out_dir,
                pca_state=None,
                save_pca_state=True,
            )
            if pretrain_pca_state is None:
                print("Cached pretrain stats have no PCA state. Recomputing pretrain stats with PCA state.")
                use_cached_prev = args.use_cached_pretrain_stats
                args.use_cached_pretrain_stats = False
                pretrain_inv_cov, pretrain_pca_state, pretrain_meta, pretrain_stats_path = compute_or_load_pretrain_stats(
                    args=args,
                    model=model,
                    device=device,
                    out_dir=out_dir,
                    pca_state=None,
                    save_pca_state=True,
                )
                args.use_cached_pretrain_stats = use_cached_prev
            if pretrain_pca_state is None:
                raise ValueError("PCA fit source is pretrain_corpus but no PCA state is available in pretrain stats.")
            emb = transform_with_pca_state(emb_raw, pretrain_pca_state)
            pca_state_used = pretrain_pca_state
            pca_source_used = "pretrain_corpus"
            print(f"Applied pretrain PCA: {emb_raw.shape[1]} -> {emb.shape[1]}")
        else:
            pca_result = fit_incremental_pca(emb_raw, args.pca_components, args.batch_size)
            if pca_result is None:
                raise ValueError(
                    f"Invalid pca_components={args.pca_components} for feature dimension {emb_raw.shape[1]}."
                )
            emb, pca_state_used = pca_result
            pca_source_used = "fewshot"
            print(
                f"Applied few-shot PCA: {emb_raw.shape[1]} -> {emb.shape[1]} "
                f"(explained variance sum: {pca_state_used['explained_variance_ratio_sum']:.4f})"
            )

            if args.distance == "mahalanobis" and args.covariance_mode == "pretrain":
                pretrain_inv_cov, pretrain_pca_state, pretrain_meta, pretrain_stats_path = compute_or_load_pretrain_stats(
                    args=args,
                    model=model,
                    device=device,
                    out_dir=out_dir,
                    pca_state=pca_state_used,
                    save_pca_state=False,
                )
    else:
        if args.distance == "mahalanobis" and args.covariance_mode == "pretrain":
            pretrain_inv_cov, pretrain_pca_state, pretrain_meta, pretrain_stats_path = compute_or_load_pretrain_stats(
                args=args,
                model=model,
                device=device,
                out_dir=out_dir,
                pca_state=None,
                save_pca_state=False,
            )

    np.save(out_dir / "embeddings_projected.npy", emb)

    if args.distance == "mahalanobis" and args.covariance_mode == "pretrain" and pretrain_inv_cov is None:
        raise ValueError("Pretrain covariance mode requested but pretrain precision matrix not available.")
    if args.distance == "mahalanobis" and args.covariance_mode == "pretrain" and pretrain_inv_cov.shape[0] != emb.shape[1]:
        raise ValueError(
            "Dimension mismatch between pretrain precision matrix and embedding dimension. "
            "Recompute pretrain stats with current embeddings."
        )

    episode_df, cm_total = run_episodes(
        embeddings=emb,
        y=y,
        class_names=class_names,
        support_per_class=args.support_per_class,
        n_episodes=args.episodes,
        distance=args.distance,
        covariance_mode=args.covariance_mode,
        cov_reg=args.cov_reg,
        seed=args.seed,
        feature_projection=args.feature_projection,
        select_k_best=args.select_k_best,
        pretrain_inv_cov=pretrain_inv_cov,
    )

    summary = {
        "n_samples": int(x.shape[0]),
        "spectrum_length": int(x.shape[1]),
        "label_column": args.label_column,
        "class_counts": class_counts,
        "support_per_class": int(args.support_per_class),
        "episodes": int(args.episodes),
        "distance": args.distance,
        "covariance_mode": args.covariance_mode,
        "cov_reg": float(args.cov_reg),
        "embedding_strategy": args.embedding_strategy,
        "feature_projection": args.feature_projection,
        "embedding_dim_raw": int(emb_raw.shape[1]),
        "embedding_dim_final": int(emb.shape[1]),
        "select_k_best": int(args.select_k_best) if args.feature_projection == "anova" else None,
        "pca_components_requested": int(args.pca_components) if args.feature_projection == "pca" else None,
        "pca_fit_source": args.pca_fit_source if args.feature_projection == "pca" else None,
        "pca_source_used": pca_source_used,
        "pca_explained_variance_ratio_sum": None if pca_state_used is None else float(pca_state_used["explained_variance_ratio_sum"]),
        "pretrain_stats_path": None if pretrain_stats_path is None else str(pretrain_stats_path),
        "pretrain_meta": pretrain_meta,
        "accuracy_mean": float(episode_df["accuracy"].mean()),
        "accuracy_std": float(episode_df["accuracy"].std(ddof=1)),
        "macro_f1_mean": float(episode_df["macro_f1"].mean()),
        "macro_f1_std": float(episode_df["macro_f1"].std(ddof=1)),
        "macro_precision_mean": float(episode_df["macro_precision"].mean()),
        "macro_recall_mean": float(episode_df["macro_recall"].mean()),
        "confusion_matrix_query_total": cm_total.tolist(),
        "class_order": class_names,
        "model_dims": dims,
    }

    used_meta.to_csv(out_dir / "used_metadata_rows.csv", index=False)
    episode_df.to_csv(out_dir / "episode_metrics.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
