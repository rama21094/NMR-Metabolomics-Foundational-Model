"""Shared utilities for BrC/T2D LOOCV evaluation scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_MAPPINGS = {
    "cancer_status": ["No Cancer", "Cancer"],
    "diabetes_status": ["No Diabetes", "Diabetes"],
    "combined_status": [
        "No Cancer No Diabetes",
        "No Cancer Diabetes",
        "Breast Cancer No Diabetes",
        "Breast Cancer Diabetes",
    ],
}


def load_brc_t2d(data_path: str | Path, metadata_path: str | Path, label_column: str):
    """Load spectra and selected labels using metadata npy_row indices."""
    data_path = Path(data_path)
    metadata_path = Path(metadata_path)
    if label_column not in LABEL_MAPPINGS:
        raise ValueError(f"Unsupported label column {label_column!r}; choose from {sorted(LABEL_MAPPINGS)}")

    spectra = np.load(data_path).astype(np.float32)
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in metadata: {metadata_path}")
    for required in ("npy_row", label_column):
        if required not in rows[0]:
            raise ValueError(f"{metadata_path} does not contain required column {required!r}")

    label_names = LABEL_MAPPINGS[label_column]
    label_to_index = {name: i for i, name in enumerate(label_names)}
    used = []
    for row in rows:
        label = str(row.get(label_column, "")).strip()
        if not label:
            continue
        if label not in label_to_index:
            raise ValueError(
                f"Unexpected label {label!r} in {label_column}; expected {label_names}"
            )
        try:
            npy_row = int(row["npy_row"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid npy_row value: {row['npy_row']!r}") from exc
        if not 0 <= npy_row < len(spectra):
            raise IndexError(f"npy_row {npy_row} is outside spectra array with {len(spectra)} rows")
        used.append((npy_row, label_to_index[label], row))

    if not used:
        raise ValueError(f"No usable rows found for label column {label_column!r}")
    used.sort(key=lambda item: item[0])
    indices = np.asarray([item[0] for item in used], dtype=np.int64)
    labels = np.asarray([item[1] for item in used], dtype=np.int64)
    metadata = [item[2] for item in used]
    return spectra[indices], labels, metadata, label_names


def binned_abs_area(spectra: np.ndarray, n_bins: int) -> np.ndarray:
    """Reduce spectra to absolute integrated area in equal-width bins."""
    edges = np.linspace(0, spectra.shape[1], n_bins + 1, dtype=int)
    features = np.empty((len(spectra), n_bins), dtype=np.float32)
    integrate = getattr(np, "trapezoid", np.trapz)
    for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        segment = np.abs(spectra[:, start:stop])
        features[:, i] = integrate(segment, axis=1) if stop - start > 1 else segment[:, 0]
    return features


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, label_names: list[str]):
    """Return binary or multiclass metrics from OOF predictions and probabilities."""
    n_classes = len(label_names)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    if n_classes == 2:
        metrics.update(
            {
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, y_prob[:, 1])),
                "pr_auc": float(average_precision_score(y_true, y_prob[:, 1])),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
    else:
        try:
            macro_roc_auc = float(roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr"))
        except ValueError:
            macro_roc_auc = float("nan")
        metrics.update(
            {
                "macro_precision": float(
                    precision_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "macro_recall": float(
                    recall_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "macro_roc_auc_ovr": macro_roc_auc,
                "confusion_matrix": json.dumps(cm.tolist()),
            }
        )
    return metrics


def probability_matrix(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Return class probabilities for sklearn-like estimators."""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
    else:
        scores = model.decision_function(x)
        if n_classes == 2 and scores.ndim == 1:
            prob_pos = 1.0 / (1.0 + np.exp(-scores))
            prob = np.column_stack([1.0 - prob_pos, prob_pos])
        else:
            scores = np.asarray(scores, dtype=np.float64)
            scores = scores - scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            prob = exp / exp.sum(axis=1, keepdims=True)
    if prob.shape[1] != n_classes:
        aligned = np.zeros((prob.shape[0], n_classes), dtype=np.float64)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps["model"], "classes_", None)
        if classes is None:
            raise ValueError("Could not align probability columns to class labels")
        for col, cls in enumerate(classes):
            aligned[:, int(cls)] = prob[:, col]
        prob = aligned
    return np.asarray(prob, dtype=np.float64)


def classical_models(seed: int, xgb_jobs: int, n_classes: int):
    logistic = LogisticRegression(
        max_iter=5000,
        C=1.0,
        random_state=seed,
        class_weight="balanced",
    )
    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", logistic),
        ]),
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required: install the 'xgboost' package") from exc

    xgb_kwargs = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": xgb_jobs,
    }
    if n_classes == 2:
        xgb_kwargs.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        xgb_kwargs.update({
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": n_classes,
        })
    models["xgboost"] = XGBClassifier(**xgb_kwargs)
    return models


def run_classical_loocv(features, labels, label_names, seed, xgb_jobs):
    results = {}
    splitter = LeaveOneOut()
    n_classes = len(label_names)
    for name, estimator in classical_models(seed, xgb_jobs, n_classes).items():
        predictions = np.empty(len(labels), dtype=np.int64)
        probabilities = np.empty((len(labels), n_classes), dtype=np.float64)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features), 1):
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            predictions[test_idx] = model.predict(features[test_idx])
            probabilities[test_idx] = probability_matrix(model, features[test_idx], n_classes)
            print(f"\rclassical/{name}: LOOCV fold {fold}/{len(labels)}", end="", flush=True)
        print()
        results[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": aggregate_metrics(labels, predictions, probabilities, label_names),
        }
    return results


def default_output_dir(base_dir: str, label_column: str) -> str:
    return str(Path(base_dir) / label_column)


def save_results(output_dir, metadata, labels, label_names, label_column, families, run_config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    prediction_rows = []

    for i, row in enumerate(metadata):
        prediction_rows.append(
            {
                "npy_row": row["npy_row"],
                "sample_id": row.get("ID", ""),
                "sample_name": row.get("Sample name/ID", row.get("Sample Name", "")),
                "label_column": label_column,
                "label": label_names[int(labels[i])],
                "target": int(labels[i]),
            }
        )

    for family, models in families.items():
        for model_name, result in models.items():
            summary_rows.append({"family": family, "model": model_name, **result["metrics"]})
            np.save(output_dir / f"{family}_{model_name}_oof_pred.npy", result["predictions"])
            np.save(output_dir / f"{family}_{model_name}_oof_prob.npy", result["probabilities"])
            for i, pred in enumerate(result["predictions"]):
                prefix = f"{family}_{model_name}"
                prediction_rows[i][f"{prefix}_prediction"] = label_names[int(pred)]
                for class_idx, class_name in enumerate(label_names):
                    safe_name = class_name.lower().replace(" ", "_").replace("/", "_")
                    prediction_rows[i][f"{prefix}_prob_{safe_name}"] = float(
                        result["probabilities"][i, class_idx]
                    )

    if not summary_rows:
        raise ValueError("No models were evaluated; nothing to save.")

    fieldnames = sorted({key for row in summary_rows for key in row})
    preferred = ["family", "model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    prediction_fields = sorted({key for row in prediction_rows for key in row})
    preferred_prediction = ["npy_row", "sample_id", "sample_name", "label_column", "label", "target"]
    prediction_fields = [key for key in preferred_prediction if key in prediction_fields] + [
        key for key in prediction_fields if key not in preferred_prediction
    ]
    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
