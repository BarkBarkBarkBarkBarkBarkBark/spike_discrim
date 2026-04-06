"""
evaluation.py — Classification and compute-cost metrics.

All functions accept numpy arrays and return plain Python scalars or dicts
so results can be serialised directly to JSON without any pandas dependency.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


# --------------------------------------------------------------------------- #
# Classification metrics                                                      #
# --------------------------------------------------------------------------- #

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC for binary classification."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray,
               average: str = "binary") -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_confusion_matrix(y_true: np.ndarray,
                              y_pred: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "precision": float(tp / (tp + fp + 1e-12)),
            "recall":    float(tp / (tp + fn + 1e-12)),
            "fpr":       float(fp / (fp + tn + 1e-12)),
        }
    return {"matrix": cm.tolist()}


def compute_silhouette(X: np.ndarray, y: np.ndarray) -> float:
    """Silhouette score of feature space clustering by class/unit labels."""
    from sklearn.metrics import silhouette_score
    if len(np.unique(y)) < 2:
        return 0.0
    try:
        return float(silhouette_score(X, y, sample_size=min(1000, len(y))))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Fisher discriminability score (single feature)                             #
# --------------------------------------------------------------------------- #

def fisher_score(feature_col: np.ndarray, labels: np.ndarray) -> float:
    """Fisher discriminant ratio: (μ₁ - μ₀)² / (σ₁² + σ₀²).

    Higher = better separation between spike and noise classes.
    Works for binary labels {0, 1}.
    """
    mask1 = labels == 1
    mask0 = labels == 0
    if mask1.sum() < 2 or mask0.sum() < 2:
        return 0.0
    mu1, mu0 = feature_col[mask1].mean(), feature_col[mask0].mean()
    v1,  v0  = feature_col[mask1].var(),  feature_col[mask0].var()
    denom = v1 + v0 + 1e-12
    return float((mu1 - mu0) ** 2 / denom)


def mutual_information_score(feature_col: np.ndarray,
                              labels: np.ndarray,
                              n_bins: int = 20) -> float:
    """Discretise feature into n_bins and compute mutual information with label."""
    from sklearn.metrics import mutual_info_score
    bins   = np.linspace(feature_col.min(), feature_col.max() + 1e-10, n_bins + 1)
    disc   = np.digitize(feature_col, bins) - 1
    return float(mutual_info_score(labels, disc))


# --------------------------------------------------------------------------- #
# Pairwise AUC (multi-unit discrimination)                                   #
# --------------------------------------------------------------------------- #

def compute_pairwise_aucs(
    X:      np.ndarray,
    y:      np.ndarray,
    model_fn,
) -> dict[str, float]:
    """Compute one-vs-one AUC for every pair of unique labels.

    model_fn : callable() → unfitted sklearn-compatible model
    Returns dict { "u1_vs_u2": auc, ... }
    """
    from sklearn.metrics import roc_auc_score
    units = np.unique(y)
    results = {}
    for i, u1 in enumerate(units):
        for u2 in units[i+1:]:
            mask = (y == u1) | (y == u2)
            Xp   = X[mask]
            yp   = (y[mask] == u2).astype(int)
            model = model_fn()
            model.fit(Xp, yp)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(Xp)[:, 1]
            else:
                scores = model.decision_function(Xp)
            try:
                auc = float(roc_auc_score(yp, scores))
            except Exception:
                auc = float("nan")
            results[f"{u1}_vs_{u2}"] = auc
    return results


def _resolve_single_feature_model_names(model_names: Optional[list]) -> list[str]:
    """Normalise benchmark-config aliases to actual model factory names."""
    if model_names is None:
        return ["lda", "logistic_regression", "knn_k5"]

    alias_map = {
        "knn_accuracy": "knn_k5",
        "logistic_accuracy": "logistic_regression",
        "lda_accuracy": "lda",
    }
    metric_only = {
        "fisher_score",
        "mutual_information",
        "mutual_info",
        "single_feature_auc",
        "auc",
    }

    resolved: list[str] = []
    for name in model_names:
        key = str(name).strip().lower()
        if not key or key in metric_only:
            continue
        mapped = alias_map.get(key, key)
        if mapped not in resolved:
            resolved.append(mapped)

    return resolved or ["lda", "logistic_regression", "knn_k5"]


# --------------------------------------------------------------------------- #
# Full single-feature evaluation report                                      #
# --------------------------------------------------------------------------- #

def evaluate_single_feature(
    feature_col:   np.ndarray,   # shape [N]
    labels:        np.ndarray,   # shape [N]  binary {0, 1}
    feature_name:  str,
    model_names:   Optional[list] = None,
) -> dict:
    """Compute all single-feature metrics for a given feature column.

    Returns a dict suitable for direct JSON serialisation.
    """
    from sklearn.model_selection import cross_val_score
    from spike_discrim.models.discriminants import make_model

    X = feature_col.reshape(-1, 1)

    result: dict = {"feature": feature_name}
    result["fisher_score"]  = fisher_score(feature_col, labels)
    result["mutual_info"]   = mutual_information_score(feature_col, labels)

    model_names = _resolve_single_feature_model_names(model_names)

    for name in model_names:
        try:
            m = make_model(name)
            scores = cross_val_score(m, X, labels, cv=5,
                                     scoring="balanced_accuracy")
            result[f"{name}_balanced_acc_mean"] = float(scores.mean())
            result[f"{name}_balanced_acc_std"]  = float(scores.std())
        except Exception as e:
            result[f"{name}_error"] = str(e)

    # AUC via logistic regression score
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        lr = LogisticRegression(max_iter=1000)
        proba = cross_val_predict(lr, X, labels, cv=5, method="predict_proba")
        result["auc"] = float(roc_auc_score(labels, proba[:, 1]))
    except Exception as e:
        result["auc_error"] = str(e)

    return result
