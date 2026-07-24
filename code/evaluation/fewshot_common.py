"""Shared few-shot episode utilities.

The key correctness requirement for a fair few-shot comparison is that every
model (3 classical baselines + 3 SSL families, each swept over fine-tune
modes) sees *the exact same* support/query draws at a given support size.
`build_shared_splits` generates those draws once, keyed only by
(seed, support_per_class) -- never by model -- so every family below just
consumes the same dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "code" / "evaluation", ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from brc_t2d_common import aggregate_metrics, classical_models, probability_matrix  # noqa: E402
from joint_ssl_eval_common import (  # noqa: E402
    build_joint_classifier,
    fine_tune_count,
    maybe_normalize_eval_spectra,
    set_seed,
    train_one_fold as train_joint_one_fold,
)
from barth_all_models_loocv import (  # noqa: E402
    build_jigsaw_classifier,
    build_masked_classifier,
    checkpoint_state,
    load_jigsaw_checkpoint_file,
    normalize_batch,
    resolve_normalize_mode,
    train_classifier_one_fold,
)


# ---------------------------------------------------------------------------
# Episode construction
# ---------------------------------------------------------------------------

def class_indices(y: np.ndarray, n_classes: int) -> dict:
    return {c: np.where(y == c)[0] for c in range(n_classes)}


def max_support_per_class(y: np.ndarray, n_classes: int, min_query_per_class: int = 2) -> int:
    """Largest support_per_class that still leaves >= min_query_per_class query
    samples in every class (the smallest class is the binding constraint)."""
    counts = class_indices(y, n_classes)
    min_class_count = min(len(idx) for idx in counts.values())
    max_support = min_class_count - int(min_query_per_class)
    if max_support < 1:
        raise ValueError(
            f"Smallest class has {min_class_count} samples; cannot leave "
            f">= {min_query_per_class} query samples per class at any positive support size."
        )
    return max_support


def support_size_grid(support_min: int, support_max: int, support_step: int = 1) -> list:
    if support_min < 1:
        raise ValueError("support_min must be >= 1")
    if support_max < support_min:
        raise ValueError(f"support_max ({support_max}) must be >= support_min ({support_min})")
    sizes = list(range(support_min, support_max + 1, support_step))
    if sizes[-1] != support_max:
        sizes.append(support_max)
    return sizes


def build_shared_splits(y: np.ndarray, n_classes: int, support_sizes: list, repeats: int, seed: int) -> dict:
    """One class-balanced (support_idx, query_idx) list per support size.

    Each support size draws from its own RNG stream seeded by
    `seed + 1000 * support_size` -- independent of any model -- so the same
    dict can be handed to the classical and every SSL family unchanged.
    """
    counts = class_indices(y, n_classes)
    splits_by_size = {}
    for size in support_sizes:
        for c, idx in counts.items():
            if len(idx) <= size:
                raise ValueError(
                    f"Class {c} has {len(idx)} samples; needs > {size} to keep >=1 query "
                    f"sample at support_per_class={size}."
                )
        rng = np.random.default_rng(seed + 1000 * size)
        episodes = []
        for _ in range(repeats):
            support_idx = []
            for c in range(n_classes):
                chosen = rng.choice(counts[c], size=size, replace=False)
                support_idx.extend(chosen.tolist())
            support_idx = np.asarray(sorted(support_idx), dtype=int)
            query_mask = np.ones(len(y), dtype=bool)
            query_mask[support_idx] = False
            query_idx = np.where(query_mask)[0]
            episodes.append((support_idx, query_idx))
        splits_by_size[size] = episodes
    return splits_by_size


def _base_row(family, model, fine_tune_mode, size, rep, n_train, n_query):
    return {
        "family": family,
        "model": model,
        "fine_tune_mode": fine_tune_mode,
        "support_per_class": size,
        "repeat": rep,
        "n_train": n_train,
        "n_query": n_query,
        "status": "ok",
        "error": "",
    }


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def evaluate_classical_fewshot(features, y, n_classes, label_names, splits_by_size, seed, xgb_jobs):
    from sklearn.base import clone

    rows = []
    models = classical_models(seed, xgb_jobs, n_classes)
    total = sum(len(v) for v in splits_by_size.values()) * len(models)
    done = 0
    for size, episodes in splits_by_size.items():
        for rep, (support_idx, query_idx) in enumerate(episodes, 1):
            for name, estimator in models.items():
                row = _base_row("classical", name, "-", size, rep, len(support_idx), len(query_idx))
                try:
                    model = clone(estimator)
                    model.fit(features[support_idx], y[support_idx])
                    pred = model.predict(features[query_idx])
                    prob = probability_matrix(model, features[query_idx], n_classes)
                    row.update(aggregate_metrics(y[query_idx], pred, prob, label_names))
                except Exception as exc:  # noqa: BLE001 - keep the sweep alive on a degenerate episode
                    row["status"] = "error"
                    row["error"] = str(exc)
                rows.append(row)
                done += 1
                print(
                    f"\r[classical/{name}] support={size} repeat={rep}/{len(episodes)} ({done}/{total})",
                    end="",
                    flush=True,
                )
    print()
    return rows


# ---------------------------------------------------------------------------
# Masked-SSL
# ---------------------------------------------------------------------------

def evaluate_masking_fewshot(spectra, y, n_classes, label_names, checkpoint_path, fine_tune_modes,
                              splits_by_size, args, device, model_label="mask_metabolights"):
    state = checkpoint_state(checkpoint_path)
    normalize_input = resolve_normalize_mode(args.normalize_input, spectra)
    normalized_spectra = normalize_batch(spectra) if normalize_input else spectra.astype(np.float32, copy=True)

    rows = []
    total = sum(len(v) for v in splits_by_size.values()) * len(fine_tune_modes)
    done = 0
    for mode in fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        for size, episodes in splits_by_size.items():
            for rep, (support_idx, query_idx) in enumerate(episodes, 1):
                row = _base_row("masking", model_label, mode, size, rep, len(support_idx), len(query_idx))
                episode_seed = args.seed + size * 1000 + rep
                try:
                    set_seed(episode_seed)
                    model, _config = build_masked_classifier(
                        state, normalized_spectra.shape[1], n_classes, args, device, unfreeze_count
                    )
                    train_classifier_one_fold(
                        model,
                        normalized_spectra[support_idx],
                        y[support_idx],
                        device,
                        args.epochs,
                        args.batch_size,
                        args.head_lr,
                        args.backbone_lr,
                        args.weight_decay,
                        episode_seed,
                        lambda m: m.backbone.encoder.transformer.layers,
                    )
                    model.eval()
                    with torch.no_grad():
                        prob = model(torch.from_numpy(normalized_spectra[query_idx]).to(device)).cpu().numpy()
                    pred = np.argmax(prob, axis=1)
                    row.update(aggregate_metrics(y[query_idx], pred, prob, label_names))
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = str(exc)
                rows.append(row)
                done += 1
                print(
                    f"\r[masking/{model_label}/{mode}] support={size} repeat={rep}/{len(episodes)} ({done}/{total})",
                    end="",
                    flush=True,
                )
    print()
    return rows


# ---------------------------------------------------------------------------
# Jigsaw-SSL
# ---------------------------------------------------------------------------

def evaluate_jigsaw_fewshot(spectra, y, n_classes, label_names, checkpoint_path, fine_tune_modes,
                             splits_by_size, args, device, model_label="jigsaw_multibin"):
    checkpoint = load_jigsaw_checkpoint_file(checkpoint_path)

    rows = []
    total = sum(len(v) for v in splits_by_size.values()) * len(fine_tune_modes)
    done = 0
    for mode in fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        for size, episodes in splits_by_size.items():
            for rep, (support_idx, query_idx) in enumerate(episodes, 1):
                row = _base_row("jigsaw", model_label, mode, size, rep, len(support_idx), len(query_idx))
                episode_seed = args.seed + size * 1000 + rep
                try:
                    set_seed(episode_seed)
                    model, _config = build_jigsaw_classifier(
                        checkpoint, spectra, n_classes, args, device, unfreeze_count
                    )
                    train_classifier_one_fold(
                        model,
                        spectra[support_idx],
                        y[support_idx],
                        device,
                        args.epochs,
                        args.batch_size,
                        args.head_lr,
                        args.backbone_lr,
                        args.weight_decay,
                        episode_seed,
                        lambda m: m.backbone.transformer.layers,
                    )
                    model.eval()
                    with torch.no_grad():
                        prob = model(torch.from_numpy(spectra[query_idx]).to(device)).cpu().numpy()
                    pred = np.argmax(prob, axis=1)
                    row.update(aggregate_metrics(y[query_idx], pred, prob, label_names))
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = str(exc)
                rows.append(row)
                done += 1
                print(
                    f"\r[jigsaw/{model_label}/{mode}] support={size} repeat={rep}/{len(episodes)} ({done}/{total})",
                    end="",
                    flush=True,
                )
    print()
    return rows


# ---------------------------------------------------------------------------
# Joint (masked + jigsaw) SSL
# ---------------------------------------------------------------------------

def evaluate_joint_fewshot(spectra, y, n_classes, label_names, checkpoint_path, fine_tune_modes,
                            splits_by_size, args, device, model_label="joint_ssl_metabolights"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # normalize_input is a fixed property of the checkpoint/mode, not of any
    # particular episode -- resolve it once via a throwaway frozen build.
    _throwaway, config = build_joint_classifier(
        checkpoint=checkpoint,
        spectra=spectra,
        n_classes=n_classes,
        head_dropout=args.head_dropout,
        normalize_input_mode=args.joint_normalize_input,
        unfreeze_layers=0,
        device=device,
        include_masked_task=args.joint_include_masked_task,
    )
    del _throwaway
    normalized_spectra = maybe_normalize_eval_spectra(spectra, config["normalize_input"])

    rows = []
    total = sum(len(v) for v in splits_by_size.values()) * len(fine_tune_modes)
    done = 0
    for mode in fine_tune_modes:
        unfreeze_count = fine_tune_count(mode)
        for size, episodes in splits_by_size.items():
            for rep, (support_idx, query_idx) in enumerate(episodes, 1):
                row = _base_row("joint_ssl", model_label, mode, size, rep, len(support_idx), len(query_idx))
                episode_seed = args.seed + size * 1000 + rep
                try:
                    set_seed(episode_seed)
                    model, _config = build_joint_classifier(
                        checkpoint=checkpoint,
                        spectra=spectra,
                        n_classes=n_classes,
                        head_dropout=args.head_dropout,
                        normalize_input_mode=args.joint_normalize_input,
                        unfreeze_layers=unfreeze_count,
                        device=device,
                        include_masked_task=args.joint_include_masked_task,
                        reinit_unfrozen=getattr(args, "reinit_unfrozen_xavier", False),
                    )
                    train_joint_one_fold(
                        model,
                        normalized_spectra[support_idx],
                        y[support_idx],
                        device,
                        args.epochs,
                        args.batch_size,
                        args.head_lr,
                        args.backbone_lr,
                        args.weight_decay,
                        episode_seed,
                    )
                    model.eval()
                    with torch.no_grad():
                        prob = model(torch.from_numpy(normalized_spectra[query_idx]).to(device)).cpu().numpy()
                    pred = np.argmax(prob, axis=1)
                    row.update(aggregate_metrics(y[query_idx], pred, prob, label_names))
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = str(exc)
                rows.append(row)
                done += 1
                print(
                    f"\r[joint_ssl/{model_label}/{mode}] support={size} repeat={rep}/{len(episodes)} ({done}/{total})",
                    end="",
                    flush=True,
                )
    print()
    return rows
