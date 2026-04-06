"""
single_feature.py — Benchmark each feature individually.

Evaluates every scalar feature in the feature matrix using:
  - Fisher score
  - Mutual information
  - AUC (logistic regression)
  - Balanced accuracy (LDA, logistic, kNN)

Writes results to a Parquet file ranked by Fisher score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def run_single_feature_benchmark(
    feature_matrix:  np.ndarray,    # float32[N, n_features]
    labels:          np.ndarray,    # int32[N]  binary {0, 1}
    feature_names:   list[str],
    results_dir:     str | Path = "data/results",
    model_names:     Optional[list] = None,
    verbose:         bool = True,
) -> "pandas.DataFrame":
    """Evaluate each feature column individually and return ranked DataFrame.

    Parameters
    ----------
    feature_matrix : float32[N, n_features]
    labels         : int32[N] — 1=spike, 0=noise
    feature_names  : column labels for feature_matrix
    results_dir    : output directory for Parquet
    model_names    : list of classifier names (see models.discriminants.make_model)
    verbose        : print progress

    Returns
    -------
    pandas DataFrame sorted by fisher_score descending.
    Also saved to results_dir/single_feature_ranks.parquet.
    """
    import pandas as pd
    from spike_discrim.metrics.evaluation import evaluate_single_feature
    from spike_discrim.io.storage import save_features_parquet

    if model_names is None:
        model_names = ["lda", "logistic_regression", "knn_k5"]

    rows = []
    n_features = feature_matrix.shape[1]
    for i, fname in enumerate(feature_names):
        if verbose:
            print(f"  [{i+1}/{n_features}] {fname}")
        row = evaluate_single_feature(
            feature_col  = feature_matrix[:, i],
            labels       = labels,
            feature_name = fname,
            model_names  = model_names,
        )
        row["feature_index"] = i
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("fisher_score", ascending=False)
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    results_dir = Path(results_dir)
    save_features_parquet(
        results_dir / "single_feature_ranks.parquet",
        df,
        metadata={"n_snippets": len(labels), "n_features": n_features},
    )

    if verbose:
        print("\nSingle-feature rankings (top 10):")
        cols = ["rank", "feature", "fisher_score", "auc"]
        available = [c for c in cols if c in df.columns]
        print(df[available].head(10).to_string(index=False))

    return df
