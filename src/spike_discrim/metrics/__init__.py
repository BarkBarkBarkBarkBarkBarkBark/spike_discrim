"""Metrics subpackage."""
from spike_discrim.metrics.evaluation import (
    compute_auc,
    compute_balanced_accuracy,
    compute_f1,
    compute_confusion_matrix,
    compute_silhouette,
    fisher_score,
    mutual_information_score,
    compute_pairwise_aucs,
    evaluate_single_feature,
    knn_purity,
    knn_purity_sweep,
)

__all__ = [
    "compute_auc", "compute_balanced_accuracy", "compute_f1",
    "compute_confusion_matrix", "compute_silhouette",
    "fisher_score", "mutual_information_score",
    "compute_pairwise_aucs", "evaluate_single_feature",
    "knn_purity", "knn_purity_sweep",
]
