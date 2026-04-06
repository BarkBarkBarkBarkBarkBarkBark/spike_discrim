"""
discriminants.py — Interpretable discriminant models for spike/noise classification.

Design principle (docs/spike_feature_validation.yaml §discrimination_models):
  Use simple, interpretable classifiers first.  The goal is to validate features,
  not hide poor features behind overly complex models.

All models share a minimal scikit-learn-compatible interface:
    model.fit(X, y)
    model.predict(X)       → int array
    model.predict_proba(X) → float array (class 1 probability)
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdDiscriminant(BaseEstimator, ClassifierMixin):
    """Single-threshold rule on a scalar feature.

    Fires spike if feature_value > threshold (for positive-polarity features)
    or feature_value < threshold (for negative-polarity features).
    """
    def __init__(self, polarity: int = 1):
        self.polarity = polarity   # +1 or -1
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdDiscriminant":
        from sklearn.metrics import roc_curve
        scores = X[:, 0] * self.polarity
        fpr, tpr, thresholds = roc_curve(y, scores)
        # Best threshold = maximise TPR - FPR (Youden's J)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        self.threshold_ = float(thresholds[best_idx]) * self.polarity
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] * self.polarity >= self.threshold_ * self.polarity).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = X[:, 0] * self.polarity
        # Soft score: sigmoid scaled around threshold
        z = scores - self.threshold_ * self.polarity
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])


class WeightBankDiscriminant(BaseEstimator, ClassifierMixin):
    """Wraps WeightBank.classify as a scikit-learn-compatible estimator."""
    def __init__(self, n_bins: int = 10, sigma_scale: float = 1.0,
                 threshold: float = 0.5):
        self.n_bins       = n_bins
        self.sigma_scale  = sigma_scale
        self.threshold    = threshold
        self.wb_: Any     = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[list] = None) -> "WeightBankDiscriminant":
        from spike_discrim.input_layer.weights import WeightBank
        # Fit on positive class (real spikes) only — calibration set
        spike_mask = y == 1
        X_spikes = X[spike_mask]
        self.wb_ = WeightBank(n_bins=self.n_bins, sigma_scale=self.sigma_scale,
                              threshold=self.threshold)
        self.wb_.fit(X_spikes, feature_names=feature_names)
        self.wb_.warmup()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.wb_.classify(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.wb_.score_batch(X)
        return np.column_stack([1.0 - scores, scores])


def make_model(name: str, **kwargs) -> Any:
    """Factory function — returns an unfitted model by name string."""
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

    models = {
        "threshold_weight_bank": lambda: WeightBankDiscriminant(
            n_bins=kwargs.get("n_bins", 10),
            sigma_scale=kwargs.get("sigma_scale", 1.0),
            threshold=kwargs.get("threshold", 0.5),
        ),
        "nearest_centroid": lambda: NearestCentroid(),
        "lda":              lambda: LinearDiscriminantAnalysis(),
        "qda":              lambda: QuadraticDiscriminantAnalysis(),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, C=kwargs.get("C", 1.0), random_state=42
        ),
        "linear_svm": lambda: LinearSVC(
            max_iter=2000, C=kwargs.get("C", 1.0), random_state=42
        ),
        "knn_k5":     lambda: KNeighborsClassifier(n_neighbors=5),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name!r}.  Choose from: {list(models)}")
    return models[name]()
