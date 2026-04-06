"""
run_benchmark.py — Full spike feature benchmarking pipeline.

Usage
-----
    python scripts/run_benchmark.py [options]

    --data-dir       data/synthetic/lfpy     (or procedural)
    --config         configs/default.yaml
    --benchmarks     configs/benchmarks.yaml
    --results-dir    data/results
    --tier           2    (1, 2, or 3 — controls which features to extract)
    --no-profile         skip op-count profiling
    --seed           42

Pipeline steps
--------------
1. Load waveforms from data-dir (waveforms.npz + labels.parquet)
2. Extract scalar features (Tier 1/2/3 per --tier)
3. Profile all Tier 1/2 kernels (op counts + timing)
4. Fit WeightBank on spike-class waveforms; report discriminant scores
5. Run single-feature benchmarks (Fisher, AUC, balanced accuracy)
6. Run feature-set benchmarks (cross-validated classifiers)
7. Save all outputs to timestamped results-dir subdirectory

All outputs are human-inspectable NPZ / Parquet / JSON in data/results/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(
        description="spike_discrim: run feature benchmarking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",     default="data/synthetic",
                   help="Directory containing waveforms.npz and labels.parquet")
    p.add_argument("--waveforms-file", default=None,
                   help="Explicit waveforms NPZ path (overrides data-dir/waveforms.npz). "
                        "Use 'data/real_units/waveforms_real.npz' for real OSort data.")
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--benchmarks",   default="configs/benchmarks.yaml")
    p.add_argument("--results-dir",  default="data/results")
    p.add_argument("--tier",         type=int, default=2,
                   choices=[1, 2, 3])
    p.add_argument("--no-profile",   action="store_true",
                   help="Skip kernel profiling (faster)")
    tm_group = p.add_mutually_exclusive_group()
    tm_group.add_argument(
        "--temporal-mad",
        action="store_true",
        help="Enable Temporal MAD/WTA overlapping-window features",
    )
    tm_group.add_argument(
        "--no-temporal-mad",
        action="store_true",
        help="Disable Temporal MAD/WTA overlapping-window features",
    )
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--quiet",        action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    verbose = not args.quiet

    # ── Load configs ─────────────────────────────────────────────────────── #
    from spike_discrim.config.loader import load_config, load_benchmark_config
    cfg   = load_config(args.config)
    bench = load_benchmark_config(args.benchmarks)
    tm_cfg = dict(cfg.get("temporal_mad", {}) or {})
    if args.temporal_mad:
        tm_cfg["enabled"] = True
    if args.no_temporal_mad:
        tm_cfg["enabled"] = False
    cfg["temporal_mad"] = tm_cfg

    # ── Create timestamped run directory ─────────────────────────────────── #
    from spike_discrim.io.storage import make_run_dir, save_results_json
    run_dir = make_run_dir(args.results_dir)
    if verbose:
        print("=" * 60)
        print("spike_discrim — Feature Benchmarking Pipeline")
        print("=" * 60)
        print(f"  Run dir    : {run_dir}")
        print(f"  Data dir   : {args.data_dir}")
        print(f"  Feature tier: {args.tier}")

    # Save config snapshot for reproducibility
    with open(run_dir / "config_snapshot.yaml", "w") as fh:
        import yaml
        yaml.safe_dump({"default": cfg, "benchmarks": bench}, fh)

    # ── Load waveforms ────────────────────────────────────────────────────── #
    data_dir = Path(args.data_dir)
    npz_path = (Path(args.waveforms_file) if args.waveforms_file
                else data_dir / "waveforms.npz")
    if not npz_path.exists():
        # Try procedural fallback: auto-generate if no data found
        print(f"  No waveforms.npz found in {data_dir}. "
              f"Generating procedural dataset...")
        from spike_discrim.synthetic.procedural_generator import generate_dataset
        ds = generate_dataset(output_dir=str(data_dir), verbose=verbose)
        waveforms    = ds["waveforms"]
        class_labels = ds["labels"].astype(np.int32)
        unit_ids     = ds["unit_ids"].astype(np.int32)
    else:
        from spike_discrim.io.storage import load_waveforms
        loaded       = load_waveforms(npz_path)
        waveforms    = loaded["waveforms"].astype(np.float32)
        class_labels = loaded.get("class_labels", np.ones(len(waveforms), dtype=np.int32))
        unit_ids     = loaded.get("unit_ids",     np.zeros(len(waveforms), dtype=np.int32))

    N, T = waveforms.shape
    if verbose:
        print(f"  Waveforms  : {N} snippets × {T} samples")
        print(f"  Spikes     : {int((class_labels == 1).sum())}  "
              f"Noise: {int((class_labels == 0).sum())}")

    waveform_summary = {
        "n_snippets": int(N),
        "n_samples": int(T),
        "n_spikes": int((class_labels == 1).sum()),
        "n_noise": int((class_labels == 0).sum()),
        "spike_fraction": float(np.mean(class_labels == 1)),
        "waveforms_file": str(npz_path),
    }
    save_results_json(run_dir / "waveform_summary.json", waveform_summary)

    # ── Step 1: Profile Tier 1/2 kernels ─────────────────────────────────── #
    if not args.no_profile:
        if verbose:
            print("\n[1/5] Profiling Tier 1/2 feature kernels...")
        from spike_discrim.profiling.op_counter import profile_all_features
        profile_results = profile_all_features(
            waveforms     = waveforms,
            window        = cfg.get("window_size_samples", 16),
            profiling_dir = run_dir / "profiling",
        )
        # Save summary
        summary = {k: {
            "wall_time_ms":   r.wall_time_ms,
            "throughput_ksnippets_per_sec": r.throughput_snippets_per_sec / 1000,
            "total_arith_ops_per_sample":   r.ops_add_per_sample + r.ops_mul_per_sample,
        } for k, r in profile_results.items()}
        save_results_json(run_dir / "profiling_summary.json", summary)

    # ── Step 2: Extract scalar feature matrix ────────────────────────────── #
    if verbose:
        print("\n[2/5] Extracting scalar features...")
    from spike_discrim.features.extraction import build_feature_matrix

    all_features, all_feat_names, feature_metadata = build_feature_matrix(
        waveforms,
        cfg=cfg,
        tier=args.tier,
    )

    if verbose:
        print(f"  Feature matrix: {all_features.shape}")
        if feature_metadata.get("temporal_mad"):
            print(
                "  Temporal MAD: enabled "
                f"({len(feature_metadata['temporal_mad'].get('window_starts', []))} bins)"
            )

    # Save feature matrix
    import pandas as pd
    feat_df = pd.DataFrame(all_features, columns=all_feat_names)
    feat_df["class_label"] = class_labels
    feat_df["unit_id"]     = unit_ids
    feat_df.to_parquet(run_dir / "feature_matrix.parquet", index=False)
    save_results_json(run_dir / "feature_metadata.json", feature_metadata)
    if feature_metadata.get("temporal_mad"):
        save_results_json(
            run_dir / "temporal_mad_metadata.json",
            feature_metadata["temporal_mad"],
        )

    # ── Step 3: Fit and evaluate WeightBank ─────────────────────────────── #
    if verbose:
        print("\n[3/5] Fitting WeightBank input layer...")
    from spike_discrim.input_layer.weights import WeightBank
    spike_mask = class_labels == 1
    wb = WeightBank(
        n_bins       = cfg.get("n_bins",       10),
        sigma_scale  = cfg.get("sigma_scale",  1.0),
        threshold    = cfg.get("discriminant_threshold", 0.5),
    )
    wb.fit(all_features[spike_mask], feature_names=all_feat_names)
    wb.warmup()
    wb.save(run_dir / "weight_bank.json")

    scores   = wb.score_batch(all_features)
    is_spike = wb.classify(all_features)

    from spike_discrim.metrics.evaluation import (
        compute_auc, compute_confusion_matrix, compute_balanced_accuracy
    )
    wb_metrics = {
        "auc":               compute_auc(class_labels, scores),
        "balanced_accuracy": compute_balanced_accuracy(class_labels, is_spike.astype(int)),
        "confusion_matrix":  compute_confusion_matrix(class_labels, is_spike.astype(int)),
        "threshold":         wb.threshold,
    }
    save_results_json(run_dir / "weight_bank_metrics.json", wb_metrics)
    if verbose:
        print(f"  WeightBank AUC: {wb_metrics['auc']:.3f}  "
              f"Balanced acc: {wb_metrics['balanced_accuracy']:.3f}")
        print(wb.describe())

    # ── Step 4: Single-feature benchmark ─────────────────────────────────── #
    if verbose:
        print("\n[4/5] Single-feature benchmark...")
    from spike_discrim.benchmarking.single_feature import run_single_feature_benchmark
    sf_df = run_single_feature_benchmark(
        feature_matrix = all_features,
        labels         = class_labels,
        feature_names  = all_feat_names,
        results_dir    = run_dir,
        model_names    = bench.get("single_feature_methods",
                                   ["lda", "logistic_regression", "knn_k5"]),
        verbose        = verbose,
    )

    # ── Step 5: Feature-set benchmark ────────────────────────────────────── #
    if verbose:
        print("\n[5/5] Feature-set benchmark...")
    from spike_discrim.benchmarking.feature_sets import run_feature_set_benchmark
    fs_df = run_feature_set_benchmark(
        feature_matrix = all_features,
        labels         = class_labels,
        feature_names  = all_feat_names,
        feature_sets   = bench.get("feature_sets", []),
        model_names    = bench.get("models", ["lda", "logistic_regression"]),
        results_dir    = run_dir,
        n_cv_folds     = bench.get("evaluation", {}).get("n_cv_folds", 5),
        random_seed    = args.seed,
        verbose        = verbose,
    )

    # ── Step 6: Afferent clustering benchmark ───────────────────────────── #
    if verbose:
        print("\n[6/6] Afferent clustering benchmark...")
    from spike_discrim.benchmarking.afferent_clustering import run_afferent_clustering_benchmark
    aff_df = run_afferent_clustering_benchmark(
        feature_matrix = all_features,
        class_labels   = class_labels,
        unit_ids       = unit_ids,
        feature_names  = all_feat_names,
        results_dir    = run_dir,
        n_bins         = cfg.get("n_bins", 10),
        sigma_scale    = cfg.get("sigma_scale", 1.0),
        threshold      = cfg.get("discriminant_threshold", 0.5),
        clustering_cfg = cfg.get("afferent_clustering", {}),
        verbose        = verbose,
    )

    # ── Final summary ─────────────────────────────────────────────────────── #
    summary = {
        "run_dir":          str(run_dir),
        "n_snippets":       N,
        "n_samples":        T,
        "n_features":       len(all_feat_names),
        "n_temporal_features": len(feature_metadata.get("temporal_mad", {}).get("window_starts", [])),
        "temporal_mad_enabled": bool(feature_metadata.get("temporal_mad")),
        "n_spikes":         int((class_labels == 1).sum()),
        "n_noise":          int((class_labels == 0).sum()),
        "weight_bank_auc":  wb_metrics["auc"],
        "top_feature":      str(sf_df.iloc[0]["feature"]) if len(sf_df) else "N/A",
        "top_feature_set":  str(fs_df.iloc[0]["set_name"]) if len(fs_df) else "N/A",
        "top_afferent_family": str(aff_df.iloc[0]["family"]) if len(aff_df) else "N/A",
        "finished_at":      datetime.now(timezone.utc).isoformat(),
    }
    save_results_json(run_dir / "run_summary.json", summary)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.  Results in: {run_dir}")
    print(f"  Top feature   : {summary['top_feature']}")
    print(f"  Top set       : {summary['top_feature_set']}")
    print(f"  Top afferent  : {summary['top_afferent_family']}")
    print(f"  WeightBank AUC: {summary['weight_bank_auc']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
