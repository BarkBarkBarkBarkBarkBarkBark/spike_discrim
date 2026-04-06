"""
feature_sets.py — Benchmark compound feature sets with multiple classifiers.

Evaluates each feature set defined in configs/benchmarks.yaml using
cross-validated classification and writes a ranked results Parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import fnmatch


def _build_feature_alias_map(feature_names: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name in feature_names:
        aliases.setdefault(name, name)
        if name.startswith("ev_"):
            aliases.setdefault(name[3:], name)
        else:
            aliases.setdefault(f"ev_{name}", name)
    return aliases


def _expand_feature_specs(feature_specs: list[str], feature_names: list[str]) -> list[str]:
    """Expand exact names and wildcard patterns against feature columns."""
    aliases = _build_feature_alias_map(feature_names)
    expanded: list[str] = []
    seen: set[str] = set()

    for spec in feature_specs:
        spec = str(spec).strip()
        if not spec:
            continue

        matches: list[str] = []
        is_pattern = any(ch in spec for ch in "*?[]")
        if is_pattern:
            matches.extend([
                name for name in feature_names
                if fnmatch.fnmatch(name, spec)
            ])
            if spec.startswith("ev_"):
                stripped = spec[3:]
                matches.extend([
                    name for name in feature_names
                    if fnmatch.fnmatch(name, stripped)
                ])
            else:
                matches.extend([
                    name for name in feature_names
                    if name.startswith("ev_") and fnmatch.fnmatch(name[3:], spec)
                ])
        else:
            mapped = aliases.get(spec)
            if mapped is not None:
                matches.append(mapped)

        for match in matches:
            if match not in seen:
                expanded.append(match)
                seen.add(match)
    return expanded


def run_feature_set_benchmark(
    feature_matrix:  np.ndarray,    # float32[N, n_features]
    labels:          np.ndarray,    # int32[N]
    feature_names:   list[str],
    feature_sets:    list[dict],    # from configs/benchmarks.yaml §feature_sets
    model_names:     list[str],
    results_dir:     str | Path = "data/results",
    n_cv_folds:      int  = 5,
    random_seed:     int  = 42,
    verbose:         bool = True,
) -> "pandas.DataFrame":
    """Evaluate each feature set × classifier combination.

    Parameters
    ----------
    feature_sets : list of dicts, each with keys:
        name             : str label
        scalar_features  : list of feature column names to include
    model_names  : list of classifier name strings (see make_model)

    Returns
    -------
    DataFrame with columns [set_name, model, balanced_acc_mean, auc_mean, ...]
    Saved to results_dir/feature_set_ranks.parquet.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, balanced_accuracy_score, roc_auc_score
    from spike_discrim.models.discriminants import make_model
    from spike_discrim.metrics.evaluation import compute_silhouette
    from spike_discrim.io.storage import save_features_parquet

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    rows = []

    for fset in feature_sets:
        set_name    = fset["name"]
        feat_cols   = fset.get("scalar_features", [])
        valid_cols  = _expand_feature_specs(feat_cols, feature_names)
        if not valid_cols:
            if verbose:
                print(f"  WARNING: no valid features for set {set_name!r}; skipping")
            continue

        col_idxs = [name_to_idx[c] for c in valid_cols]
        X_set    = feature_matrix[:, col_idxs]

        sil = compute_silhouette(X_set, labels)

        for model_name in model_names:
            if verbose:
                print(f"  {set_name}  ×  {model_name}")
            try:
                model = make_model(model_name)
                skf   = StratifiedKFold(n_splits=n_cv_folds, shuffle=True,
                                        random_state=random_seed)
                scoring = {"balanced_acc": make_scorer(balanced_accuracy_score)}
                cv_res  = cross_validate(model, X_set, labels, cv=skf,
                                         scoring=scoring, return_train_score=False)
                bal_acc_mean = float(cv_res["test_balanced_acc"].mean())
                bal_acc_std  = float(cv_res["test_balanced_acc"].std())

                rows.append({
                    "set_name":          set_name,
                    "n_features_in_set": len(valid_cols),
                    "features_used":     ",".join(valid_cols),
                    "model":             model_name,
                    "balanced_acc_mean": bal_acc_mean,
                    "balanced_acc_std":  bal_acc_std,
                    "silhouette":        sil,
                })
            except Exception as e:
                rows.append({
                    "set_name":  set_name,
                    "model":     model_name,
                    "error":     str(e),
                })

    df = pd.DataFrame(rows).sort_values(
        "balanced_acc_mean", ascending=False
    ).reset_index(drop=True)
    df["rank"] = df.index + 1

    results_dir = Path(results_dir)
    save_features_parquet(
        results_dir / "feature_set_ranks.parquet",
        df,
        metadata={"n_snippets": len(labels)},
    )

    if verbose:
        print("\nFeature-set rankings (top 10):")
        cols = ["rank", "set_name", "model", "balanced_acc_mean", "silhouette"]
        available = [c for c in cols if c in df.columns]
        print(df[available].head(10).to_string(index=False))

    return df
