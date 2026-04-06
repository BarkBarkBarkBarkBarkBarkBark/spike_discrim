"""Post-hoc clustering of afferent activations for known spikes.

This module takes the fitted input-layer representation, stores the afferent
activations for each accepted/known spike, and benchmarks how well different
feature families separate units under unsupervised clustering.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _feature_family_map(feature_names: list[str]) -> dict[str, list[int]]:
    scalar = [i for i, name in enumerate(feature_names)
              if not name.startswith("ev_") and not name.startswith("mad_wta_bin_")]
    temporal = [i for i, name in enumerate(feature_names)
                if name.startswith("mad_wta_bin_")]
    event = [i for i, name in enumerate(feature_names) if name.startswith("ev_")]
    full = list(range(len(feature_names)))

    families = {
        "scalar": scalar,
        "temporal_mad": temporal,
        "event": event,
        "full": full,
    }
    return {name: idxs for name, idxs in families.items() if idxs}


def _normalise_config(cfg: dict[str, Any] | None, n_units: int) -> dict[str, Any]:
    raw = dict(cfg or {})
    return {
        "enabled": bool(raw.get("enabled", True)),
        "n_clusters": int(raw.get("n_clusters") or n_units),
        "random_seed": int(raw.get("random_seed", 42)),
        "n_init": int(raw.get("n_init", 10)),
        "max_iter": int(raw.get("max_iter", 300)),
        "save_outputs": bool(raw.get("save_outputs", True)),
    }


def _cluster_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[int, int]]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=true_labels)
    # confusion_matrix with labels only controls y_true axis; rebuild manually for predicted labels
    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    true_to_idx = {label: i for i, label in enumerate(true_labels)}
    pred_to_idx = {label: i for i, label in enumerate(pred_labels)}
    for truth, pred in zip(y_true, y_pred):
        cm[true_to_idx[int(truth)], pred_to_idx[int(pred)]] += 1

    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    matched = int(cm[row_ind, col_ind].sum())
    mapping = {int(pred_labels[c]): int(true_labels[r]) for r, c in zip(row_ind, col_ind)}
    return matched / max(1, len(y_true)), mapping


def _cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(y_pred):
        members = y_true[y_pred == cluster]
        if len(members) == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        total += int(counts.max())
    return total / max(1, len(y_true))


def run_afferent_clustering_benchmark(
    feature_matrix: np.ndarray,
    class_labels: np.ndarray,
    unit_ids: np.ndarray,
    feature_names: list[str],
    results_dir: str | Path = "data/results",
    n_bins: int = 10,
    sigma_scale: float = 1.0,
    threshold: float = 0.5,
    clustering_cfg: dict[str, Any] | None = None,
    verbose: bool = True,
) -> "pandas.DataFrame":
    """Cluster afferent activations for known spikes and score unit separation."""
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    from spike_discrim.input_layer.weights import WeightBank
    from spike_discrim.io.storage import save_features_parquet, save_results_json
    from spike_discrim.metrics.evaluation import compute_silhouette

    spike_mask = (class_labels == 1) & (unit_ids > 0)
    X_spikes = np.ascontiguousarray(feature_matrix[spike_mask], dtype=np.float32)
    y_units = np.asarray(unit_ids[spike_mask], dtype=np.int32)

    if X_spikes.shape[0] == 0 or len(np.unique(y_units)) < 2:
        df = pd.DataFrame([])
        save_features_parquet(Path(results_dir) / "afferent_cluster_ranks.parquet", df, metadata={"status": "insufficient_units"})
        save_results_json(Path(results_dir) / "afferent_cluster_summary.json", {
            "status": "insufficient_units",
            "n_spike_events": int(X_spikes.shape[0]),
            "n_unique_units": int(len(np.unique(y_units))),
        })
        return df

    n_units = int(len(np.unique(y_units)))
    cfg = _normalise_config(clustering_cfg, n_units=n_units)
    if not cfg["enabled"]:
        df = pd.DataFrame([])
        save_features_parquet(Path(results_dir) / "afferent_cluster_ranks.parquet", df, metadata={"status": "disabled"})
        return df

    wb = WeightBank(n_bins=n_bins, sigma_scale=sigma_scale, threshold=threshold)
    wb.fit(X_spikes, feature_names=feature_names)
    wb.warmup()
    activations = wb.project_batch(X_spikes)

    families = _feature_family_map(feature_names)
    rows: list[dict[str, Any]] = []
    output_dir = Path(results_dir) / "afferent_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for family_name, feat_idxs in families.items():
        family_feature_names = [feature_names[i] for i in feat_idxs]
        family_activations = activations[:, feat_idxs, :].reshape(X_spikes.shape[0], -1)

        if cfg["save_outputs"]:
            np.savez_compressed(
                output_dir / f"{family_name}.npz",
                afferent_outputs=family_activations.astype(np.float32),
                unit_ids=y_units.astype(np.int32),
                feature_indices=np.array(feat_idxs, dtype=np.int32),
                feature_names=np.array(family_feature_names, dtype=object),
            )
            with open(output_dir / f"{family_name}_metadata.json", "w") as fh:
                json.dump({
                    "family": family_name,
                    "n_events": int(X_spikes.shape[0]),
                    "n_features": int(len(feat_idxs)),
                    "n_bins": int(n_bins),
                    "flat_dim": int(family_activations.shape[1]),
                    "feature_names": family_feature_names,
                }, fh, indent=2)

        if verbose:
            print(f"  Afferent clustering: {family_name} ({family_activations.shape[1]} dims)")

        km = KMeans(
            n_clusters=cfg["n_clusters"],
            random_state=cfg["random_seed"],
            n_init=cfg["n_init"],
            max_iter=cfg["max_iter"],
        )
        pred = km.fit_predict(family_activations)

        match_acc, mapping = _cluster_match_accuracy(y_units, pred)
        purity = _cluster_purity(y_units, pred)
        ari = float(adjusted_rand_score(y_units, pred))
        nmi = float(normalized_mutual_info_score(y_units, pred))
        sil = compute_silhouette(family_activations, pred)

        rows.append({
            "family": family_name,
            "n_input_features": int(len(feat_idxs)),
            "n_bins": int(n_bins),
            "flat_dim": int(family_activations.shape[1]),
            "n_events": int(X_spikes.shape[0]),
            "n_units": int(n_units),
            "n_clusters": int(cfg["n_clusters"]),
            "ari": ari,
            "nmi": nmi,
            "matched_accuracy": float(match_acc),
            "purity": float(purity),
            "silhouette": float(sil),
            "inertia": float(km.inertia_),
            "feature_names": ",".join(family_feature_names),
            "cluster_to_unit_map": json.dumps(mapping),
        })

        assign_df = pd.DataFrame({
            "unit_id": y_units.astype(np.int32),
            "cluster_id": pred.astype(np.int32),
        })
        assign_df.to_parquet(output_dir / f"{family_name}_assignments.parquet", index=False)

    df = pd.DataFrame(rows).sort_values(
        ["matched_accuracy", "ari", "nmi"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if len(df):
        df["rank"] = np.arange(1, len(df) + 1)

    save_features_parquet(
        Path(results_dir) / "afferent_cluster_ranks.parquet",
        df,
        metadata={
            "n_spike_events": int(X_spikes.shape[0]),
            "n_units": int(n_units),
            "n_bins": int(n_bins),
        },
    )

    summary = {
        "n_spike_events": int(X_spikes.shape[0]),
        "n_units": int(n_units),
        "n_bins": int(n_bins),
        "families_evaluated": list(df["family"]) if len(df) else [],
        "best_family": str(df.iloc[0]["family"]) if len(df) else None,
        "best_matched_accuracy": float(df.iloc[0]["matched_accuracy"]) if len(df) else None,
        "best_ari": float(df.iloc[0]["ari"]) if len(df) else None,
    }
    save_results_json(Path(results_dir) / "afferent_cluster_summary.json", summary)
    return df