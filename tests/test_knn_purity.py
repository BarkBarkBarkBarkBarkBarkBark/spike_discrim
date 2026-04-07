"""Tests for KNN purity metric and its integration into afferent clustering."""
import numpy as np
import pytest

from spike_discrim.metrics.evaluation import knn_purity, knn_purity_sweep


# --------------------------------------------------------------------------- #
# Unit tests for knn_purity                                                   #
# --------------------------------------------------------------------------- #

class TestKnnPurity:
    """Core metric correctness."""

    def test_perfect_separation_returns_one(self):
        """Two well-separated clusters → every neighbour shares the label."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(loc=-5.0, scale=0.1, size=(50, 3)),
            rng.normal(loc=5.0, scale=0.1, size=(50, 3)),
        ])
        y = np.array([0] * 50 + [1] * 50)
        assert knn_purity(X, y, k=5) == pytest.approx(1.0)

    def test_random_labels_near_chance(self):
        """Random labels → purity ≈ 1 / n_classes."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(500, 4))
        y = rng.integers(0, 5, size=500)  # 5 classes
        p = knn_purity(X, y, k=10)
        # chance = 0.2; allow generous margin for randomness
        assert 0.10 < p < 0.40

    def test_k_equals_one_matches_nearest_neighbour(self):
        """k=1 → purity is just the 1-NN accuracy."""
        X = np.array([[0.0], [0.1], [10.0], [10.1]])
        y = np.array([0, 0, 1, 1])
        assert knn_purity(X, y, k=1) == pytest.approx(1.0)

    def test_too_few_samples_returns_nan(self):
        """When n_samples < k+1, return nan gracefully."""
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 1])
        assert np.isnan(knn_purity(X, y, k=5))

    def test_output_bounded_zero_one(self):
        """Result is always in [0, 1]."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 2))
        y = rng.integers(0, 3, size=100)
        p = knn_purity(X, y, k=5)
        assert 0.0 <= p <= 1.0

    def test_three_units_geometry(self):
        """Three tight clusters at known positions — purity should be ~1."""
        rng = np.random.default_rng(99)
        centres = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float64)
        X = np.vstack([c + rng.normal(scale=0.2, size=(40, 2)) for c in centres])
        y = np.repeat([0, 1, 2], 40)
        assert knn_purity(X, y, k=10) > 0.95


# --------------------------------------------------------------------------- #
# Sweep tests                                                                 #
# --------------------------------------------------------------------------- #

class TestKnnPuritySweep:
    """Sweep over multiple K values."""

    def test_returns_all_requested_keys(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = np.array([0] * 30 + [1] * 30)
        result = knn_purity_sweep(X, y, k_values=(1, 5, 10, 20))
        assert set(result.keys()) == {
            "knn_purity_k01", "knn_purity_k05",
            "knn_purity_k10", "knn_purity_k20",
        }
        for v in result.values():
            assert isinstance(v, float)

    def test_monotonicity_for_perfect_clusters(self):
        """For well-separated data purity stays 1.0 across all K."""
        rng = np.random.default_rng(11)
        X = np.vstack([
            rng.normal(loc=-10, scale=0.1, size=(50, 2)),
            rng.normal(loc=10, scale=0.1, size=(50, 2)),
        ])
        y = np.array([0] * 50 + [1] * 50)
        result = knn_purity_sweep(X, y, k_values=(1, 5, 10))
        assert all(v == pytest.approx(1.0) for v in result.values())


# --------------------------------------------------------------------------- #
# Integration: afferent clustering benchmark includes KNN purity columns     #
# --------------------------------------------------------------------------- #

def test_afferent_clustering_includes_knn_purity(tmp_path):
    """End-to-end: benchmark dataframe contains knn_purity_k* columns."""
    from spike_discrim.benchmarking.afferent_clustering import (
        run_afferent_clustering_benchmark,
    )

    rng = np.random.default_rng(1)
    unit1 = rng.normal(loc=-1.0, scale=0.15, size=(30, 4)).astype(np.float32)
    unit2 = rng.normal(loc=1.0, scale=0.15, size=(30, 4)).astype(np.float32)
    X = np.vstack([unit1, unit2]).astype(np.float32)
    class_labels = np.ones(len(X), dtype=np.int32)
    unit_ids = np.array([1] * 30 + [2] * 30, dtype=np.int32)
    feature_names = [
        "peak_amplitude", "trough_amplitude",
        "mad_wta_bin_00", "ev_baseline_rms",
    ]

    df = run_afferent_clustering_benchmark(
        feature_matrix=X,
        class_labels=class_labels,
        unit_ids=unit_ids,
        feature_names=feature_names,
        results_dir=tmp_path,
        n_bins=5,
        clustering_cfg={"enabled": True, "save_outputs": True, "random_seed": 0},
        verbose=False,
    )

    knn_cols = [c for c in df.columns if c.startswith("knn_purity_k")]
    assert len(knn_cols) == 4  # k=1, 5, 10, 20
    for col in knn_cols:
        vals = df[col].tolist()
        assert all(0.0 <= v <= 1.0 for v in vals), f"out-of-range in {col}"

    # Well-separated clusters → high purity expected
    best = df.iloc[0]
    assert best["knn_purity_k05"] > 0.8

    # Summary JSON should contain best KNN purity values
    import json
    summary = json.loads((tmp_path / "afferent_cluster_summary.json").read_text())
    assert "best_knn_purity_k05" in summary
