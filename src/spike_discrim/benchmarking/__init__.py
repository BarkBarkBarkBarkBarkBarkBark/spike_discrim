"""Benchmarking subpackage."""
from spike_discrim.benchmarking.afferent_clustering import run_afferent_clustering_benchmark
from spike_discrim.benchmarking.single_feature import run_single_feature_benchmark
from spike_discrim.benchmarking.feature_sets import run_feature_set_benchmark

__all__ = [
	"run_afferent_clustering_benchmark",
	"run_single_feature_benchmark",
	"run_feature_set_benchmark",
]
