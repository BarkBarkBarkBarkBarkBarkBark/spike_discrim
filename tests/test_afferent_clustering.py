import numpy as np

from spike_discrim.benchmarking.afferent_clustering import run_afferent_clustering_benchmark
from spike_discrim.input_layer.weights import WeightBank


def test_weightbank_project_batch_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (20, 4)).astype(np.float32)
    wb = WeightBank(n_bins=6, sigma_scale=1.0, threshold=0.5)
    wb.fit(X, feature_names=[f"f{i}" for i in range(4)])
    wb.warmup()

    projected = wb.project_batch(X[:5])
    assert projected.shape == (5, 4, 6)
    assert np.all(projected >= 0.0)
    assert np.all(projected <= 1.0)


def test_afferent_clustering_benchmark_outputs(tmp_path):
    rng = np.random.default_rng(1)
    unit1 = rng.normal(loc=-1.0, scale=0.15, size=(30, 4)).astype(np.float32)
    unit2 = rng.normal(loc=1.0, scale=0.15, size=(30, 4)).astype(np.float32)
    X = np.vstack([unit1, unit2]).astype(np.float32)
    class_labels = np.ones(len(X), dtype=np.int32)
    unit_ids = np.array([1] * len(unit1) + [2] * len(unit2), dtype=np.int32)
    feature_names = ["peak_amplitude", "trough_amplitude", "mad_wta_bin_00", "ev_baseline_rms"]

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

    assert not df.empty
    assert {"scalar", "temporal_mad", "event", "full"}.issuperset(set(df["family"].tolist()))
    assert (tmp_path / "afferent_cluster_ranks.parquet").exists()
    assert (tmp_path / "afferent_cluster_summary.json").exists()
    assert (tmp_path / "afferent_outputs" / "full.npz").exists()