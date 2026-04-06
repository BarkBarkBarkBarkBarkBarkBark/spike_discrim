# spike_discrim — Project Orientation
**For the technically fluent first-time reader**  
*You should finish this document knowing exactly where to look, what to run, and why each piece exists.*

---

## What This Project Is

**spike_discrim** is a benchmarking framework that answers one scientific question:

> *Which waveform features best discriminate individual neurons from noise, under the hard constraint that the computation must fit inside a ~1 ms real-time window?*

The output is not a spike sorter. It is a **ranked evidence table** — feature × discriminability × compute cost — that feeds directly into the design of the input layer of a real-time spiking neural network (SNN). The intended downstream application is closed-loop neurofeedback and brain-computer interfaces, where offline sorting is too slow and offline-trained classifiers are too opaque to certify.

---

## The Scientific Problem in One Picture

A microwire electrode records voltage as a mix of:

```
                raw recording at electrode
                ──────────────────────────
                  neuron A action potential    ←  what we want
                + neuron B action potential    ←  multi-unit contamination
                + noise (oscillations, drift)  ←  what we need to reject
                + movement artifacts           ←  hard negatives
```

Each threshold-crossing event produces a **waveform snippet** of ~64 samples (~2.1 ms at 30 kHz). The discriminator must classify it — spike or noise — before the next one arrives, without ever seeing the full recording context.

This project measures which mathematical summaries of that snippet carry the most information for making that decision, and at what computational cost.

---

## Three-Tier Data Strategy

The framework is deliberately designed around two independent data sources. Their differences are what make the science rigorous:

| Tier | Source | Ground truth | Purpose |
|------|--------|-------------|---------|
| **Synthetic** | Procedural double-exponential generator | Exact (generated) | Feature space exploration, controlled variation, repeatable baselines |
| **Real curated** | OSort .mat files + human expert labels (`ephys_data/sort/`) | Expert-verified | Generalisation test — does the synthetic-trained model survive real physiology? |

The **critical scientific test** is the gap between performance on synthetic data and performance on real data. A feature that looks powerful on synthetic waveforms but fails on real recordings reveals a distribution mismatch — exactly the kind of result that prevents premature hardware commitment.

---

## Complete Implementation Walk-Through

### Step 0 — Install

```bash
git clone <repo>
cd spike_discrim
pip install -e .              # core: numpy, numba, scipy, fastapi, pandas, pyarrow
pip install -e ".[lfpy]"     # optional: LFPy + NEURON for biophysical waveforms
```

The package installs in editable mode. All code lives in `src/spike_discrim/`.

---

### Step 1 — Acquire or Generate Waveforms → `data/synthetic/waveforms.npz`

Two paths:

**A. Procedural (fast, no dependencies):**  
The `procedural_generator` synthesises physiologically shaped waveforms using a double-exponential trough model:

$$\text{trough}(t) = -A \cdot \left[\exp\!\left(-\frac{t - t_0}{\tau_\text{fall}}\right) - \exp\!\left(-\frac{t - t_0}{\tau_\text{rise}}\right)\right], \quad t \geq t_0$$

with $\tau_\text{rise} \approx 3$ samples (fast Na⁺ depolarisation) and $\tau_\text{fall} \approx 12$ samples (K⁺/Na⁺ repolarisation). Noise events are generated as Gaussian white noise, coloured noise, clipped artifacts, and multi-peak transients — all lacking the clean biphasic morphology of real spikes.

```bash
# Auto-generates if no waveforms.npz exists, or trigger explicitly:
python scripts/run_benchmark.py --data-dir data/synthetic
```

**B. LFPy/NEURON biophysical forward model** (slower, requires NEURON install):
```bash
python scripts/run_lfpy_generation.py --output-dir data/synthetic \
    --n-distances 8 --n-angles 16
```
This places virtual neurons at varying distances and orientations from the electrode and computes the extracellular field with the line-source approximation (Holt & Koch 1999).

**Output data object:**
```
data/synthetic/waveforms.npz
├── waveforms     (N, 64)  float32  µV   — one row per threshold-crossing event
├── class_labels  (N,)     int32         — 0 = noise, 1 = spike
└── unit_ids      (N,)     int32         — 0 = noise, 1..K = neuron identity

data/synthetic/labels.parquet           — same labels as structured DataFrame
data/synthetic/generation_config.json  — parameters used, for reproducibility
```

**B. Real OSort data** (ground truth from actual recordings):

Run `osort_file_extraction.ipynb` to extract from `.mat` files and produce:
```
data/real_units/waveforms_real.npz
├── waveforms     (96006, 64) float32  µV  — trough-aligned, 64 samples
├── class_labels  (96006,)    int32        — 0 = noise, 1 = SU (10 units, 5 channels)
└── unit_ids      (96006,)    int32        — 1..10 = SU identity, 0 = noise

data/real_units/waveforms_real_meta.csv
└── source_file, cluster_id, label, unit_id, timestamp_s  — full provenance

ephys_data/sort/curation_log.npz
└── channels, quality_code, n_sua, has_neurons, threshold_used, cluster_ids_raw
```

The noise class in `waveforms_real.npz` includes two sub-types:
- **`useNegativeExcluded` events** — OSort-accepted clusters that a human expert subsequently rejected after visual inspection. These are the hardest negatives: they have spike-like amplitude and morphology but were judged non-isolated.
- **Cluster `99999999` events** — OSort's hard noise sentinel: oscillations, saturations, cross-talk.

---

### Step 2 — Feature Extraction → `data/results/<run_id>/feature_matrix.parquet`

```bash
python scripts/run_benchmark.py --data-dir data/synthetic --tier 2
python scripts/run_benchmark.py --waveforms-file data/real_units/waveforms_real.npz --tier 3
```

Feature extraction is organised into three tiers with progressively increasing compute cost:

| Tier | Features | Ops/sample | Causal? | Physical quantity |
|------|----------|-----------|---------|-------------------|
| **1** | `amplitude` $x[t]$ | 0 | ✓ | Signal value |
| **1** | `first_derivative` $x[t] - x[t{-}1]$ | 1 | ✓ | Instantaneous slope |
| **1** | `second_derivative` $x[t{+}1] - 2x[t] + x[t{-}1]$ | 3 | ✓ | Curvature / inflection |
| **2** | `absolute_window_sum` $\sum_{w}\|x[t]\|$ | 1 | ✓ | Local signal energy |
| **2** | `short_window_energy` $\sum_{w} x[t]^2$ | 2 | ✓ | Squared local energy |
| **2** | `teager_energy` $x[t]^2 - x[t{-}1]{\cdot}x[t{+}1]$ | 3 | ✓ | Instantaneous frequency proxy |
| **2** | `mad_wta_bin_*` | window reduction | ✓ | Overlapping-window robust amplitude / afferent code |
| **3** | `trough_amplitude`, `peak_amplitude` | full snippet | — | Waveform extrema in µV |
| **3** | `trough_to_peak_time` | full snippet | — | Spike duration (samples) |
| **3** | `half_width`, `full_width` | full snippet | — | Temporal width at 50%/10% depth |
| **3** | `biphasic_ratio` $\|peak\| / (\|trough\| + \varepsilon)$ | full snippet | — | Waveform asymmetry |
| **3** | `baseline_rms`, `signed_area`, `zero_crossings` | full snippet | — | Shape summary statistics |

All kernels are Numba-JIT compiled (`@numba.njit(cache=True, fastmath=True)`) and operate on pre-allocated float32 buffers — zero Python-level heap allocation occurs in the hot path.

**Output data object:**
```
data/results/<run_id>/feature_matrix.parquet
└── columns: [peak_amplitude, max_slope, ..., class_label, unit_id]
    rows: one per waveform snippet  (same order as waveforms.npz)
    dtypes: float32 for features, int32 for labels
```

The run directory is timestamped (`20260405_153115`) and also contains:
```
config_snapshot.yaml    — exact hyperparameters used (pinned for reproducibility)
waveform_summary.json   — N spikes, N noise, snippet shape, class balance
temporal_mad_metadata.json — time-bin layout + normalisation settings (when enabled)
```

---

### Step 3 — Op-Count Profiling → `data/results/<run_id>/profiling_summary.json`

Before benchmarking discrimination, the framework measures the **compute cost** of each feature kernel — independently of performance. This is critical for the real-time constraint: a feature that costs 10× more than another but discriminates only 5% better is a poor choice for hardware.

The profiler reports:
- **Wall-clock time** per kernel per batch (µs)
- **Throughput** in ksnippets/second
- **Arithmetic operations per sample** (statically counted, not measured — immune to CPU load variation)

```
profiling_summary.json
└── {
      "first_derivative": {"wall_time_ms": 0.12, "throughput_ksnippets_per_sec": 8200,
                           "total_arith_ops_per_sample": 1},
      "teager_energy":    {"wall_time_ms": 0.41, "throughput_ksnippets_per_sec": 2400,
                           "total_arith_ops_per_sample": 3},
      ...
    }
```

This table is the **compute axis** of the final compute-vs-discrimination scatter plot. It does not change with the dataset — rerun only when the kernel code changes.

---

### Step 4 — WeightBank Fitting → `data/results/<run_id>/weight_bank.json`

The WeightBank is the **candidate input layer** — the component that would be hard-coded into the real-time SNN if these features are selected.

**Architecture:** Population code. Each scalar feature is encoded by $N_\text{bins}$ Gaussian tuning neurons. Neuron $b$ of feature $f$ fires with activation:

$$a_{f,b} = \exp\!\left(-\frac{(x_f - c_{f,b})^2}{2\sigma_{f,b}^2}\right)$$

Bin centers $c_{f,b}$ are set from **quantiles of a calibration spike set** — not uniformly spaced. This guarantees equal neuron utilisation across the real distribution: no neuron is wasted on rare amplitude values.

The discriminant score aggregates all activations:

$$\text{score} = \frac{\sum_f w_f \cdot \max_b\, a_{f,b}}{\sum_f w_f} \in [0, 1]$$

A real spike — whose feature values lie within the learned distribution — produces high activation across most bins → score near 1. A noise event with wrong shape or amplitude produces sparse, inconsistent activation → score near 0.

The score threshold (default 0.5) is tunable from ROC analysis on a labelled validation set.

**Output data object:**
```
weight_bank.json  (human-readable JSON, ~5 KB)
└── {
      "n_bins": 10,
      "sigma_scale": 1.0,
      "threshold": 0.5,
      "feature_names": ["peak_amplitude", "max_slope", ...],
      "centers": [[c_0, c_1, ..., c_9], ...],   # (n_features, n_bins)
      "widths":  [[w_0, w_1, ..., w_9], ...],   # (n_features, n_bins)
      "feature_weights": [1.0, 1.0, ...]         # (n_features,)
    }

weight_bank_metrics.json
└── {
      "auc": 0.97,
      "balanced_accuracy": 0.94,
      "confusion_matrix": {"TP": ..., "FP": ..., "TN": ..., "FN": ...},
      "threshold": 0.5
    }
```

The JSON is intentionally human-inspectable: you can read the bin centers for `trough_amplitude` and confirm they span the real spike amplitude distribution.

---

### Step 5 — Single-Feature Benchmark → `data/results/<run_id>/single_feature_ranks.parquet`

Every feature is evaluated **individually** using four metrics:

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Fisher score** $(μ_1 - μ_0)^2 / (σ_1^2 + σ_0^2)$ | Ratio of between-class to within-class variance | Interpretable statistical separation |
| **Mutual information** | Reduction in label entropy given feature | Non-linear, distribution-free separability |
| **AUC** (logistic regression) | Area under ROC curve for this feature alone | Calibrated probability of correct rank order |
| **Balanced accuracy** (LDA, logistic, kNN) | Mean of spike recall and noise recall | Robust to class imbalance |

**Output data object:**
```
single_feature_ranks.parquet
└── columns: [rank, feature, fisher_score, mutual_info, auc,
              balanced_acc_lda, balanced_acc_logistic, balanced_acc_knn_k5,
              feature_index]
    rows: one per feature, sorted by fisher_score descending
```

This table is the **performance axis** of the compute-vs-discrimination plot. The ideal feature occupies the upper-left: high Fisher score (discriminative) with low ops/sample (cheap).

---

### Step 6 — Feature-Set Benchmark → `data/results/<run_id>/feature_set_ranks.parquet`

Individual feature rankings do not capture **interactions**: two correlated features may add little incremental value, while two uncorrelated features may be jointly far stronger. The feature-set benchmark evaluates compound feature combinations using 5-fold cross-validated classification.

Feature sets are defined in `configs/benchmarks.yaml`:

| Set | Features | Tier | Description |
|-----|----------|------|-------------|
| `set_A_ultra_fast` | amplitude, d1, d2, max_slope, curvature | 1 | FPGA-portable minimal set |
| `set_B_fast_plus_shape` | set_A + abs_window_sum | 2 | Best overall tradeoff |
| `set_C_refined_discrimination` | trough/peak amplitude, timing, width, area | 3 | Post-detection unit identity |
| `set_D_combined` | union of all above | 2+3 | Upper bound reference |

Each set is tested against every classifier in `benchmarks.yaml §models`: LDA, logistic regression, kNN(k=5), and a threshold gate.

**Output data object:**
```
feature_set_ranks.parquet
└── columns: [rank, feature_set, classifier, balanced_accuracy, auc,
              n_features, tier, mean_cv_score, std_cv_score]
    rows: one per (feature_set × classifier) combination
```

---

### Step 7 — Real-Data Generalisation Test

This is the step that makes the science reproducible and honest.

```bash
# Option A: train on real data from scratch
POST /api/pipeline/run   {"use_real_data": true, "tier": 2}

# Option B: test a synthetic-trained WeightBank on real waveforms
POST /api/ephys/evaluate {"run_id": "20260405_153115", "tier": 2}
```

The evaluate endpoint returns:

```json
{
  "overall": {
    "auc": 0.44,
    "balanced_accuracy": 0.50,
    "precision": 0.24,
    "recall": 0.007
  },
  "per_su_recall": {"1": 0.001, "4": 0.215, ...},
  "per_source_file": [
    {"source_file": "A_ss11_Min_sorted_new.mat", "accuracy": 0.93},
    {"source_file": "A_ss13_Max_sorted_new.mat", "accuracy": 0.16}
  ]
}
```

An AUC of 0.44 on real data after synthetic training is **an important result, not a failure**. It means the procedural synthetic model does not capture the full variability of real recordings on these channels. The per-channel breakdown tells you *which* channels the model transfers to — informing which aspects of the waveform model to improve next.

---

## The API Layer

The FastAPI server (port 8099) exposes every step above as queryable endpoints. This enables **agentic evaluation** — an agent or external script can run the full pipeline, query results, and iterate without touching Python directly.

```bash
# Start the server
.venv/bin/python -m uvicorn api.main:app --port 8099 --reload
# or: spike-api (if installed as console script)

# Full interactive docs
open http://localhost:8099/api/docs
```

### Endpoint Map

```
── Data ────────────────────────────────────────────────────────────────────
GET  /api/runs                        list all benchmark runs
GET  /api/runs/{run_id}/summary       dataset statistics for this run
GET  /api/runs/{run_id}/features/single  per-feature rankings (Parquet → JSON)
GET  /api/runs/{run_id}/features/sets    per-feature-set rankings
GET  /api/runs/{run_id}/weightbank    WeightBank AUC and metrics
GET  /api/runs/{run_id}/waveforms     first N waveforms for visualisation
GET  /api/runs/{run_id}/export/csv    ZIP of all results as CSV

── Pipeline control ────────────────────────────────────────────────────────
POST /api/pipeline/run                launch benchmark pipeline (background job)
     body: {tier, data_dir, no_profile, use_real_data}
GET  /api/pipeline/status/{job_id}    tail log + status (queued/running/done/failed)
GET  /api/pipeline/jobs               list all launched jobs

── Real data evaluation ─────────────────────────────────────────────────────
GET  /api/ephys/dataset               inventory of waveforms_real.npz
GET  /api/ephys/tests                 5 biological plausibility tests (pass/fail)
POST /api/ephys/evaluate              score real waveforms with a run's WeightBank
GET  /api/ephys/waveforms             sample waveforms filtered by class/unit

── Validation (proof of correctness) ────────────────────────────────────────
GET  /api/validate/checksums/{run_id}     SHA-256 of all result files
GET  /api/validate/metrics/{run_id}       recompute AUC from raw data, compare stored
GET  /api/validate/feature_stats/{run_id} per-feature mean/std/min/max by class
GET  /api/validate/waveform_checksums     checksum of the waveform dataset
GET  /api/validate/roundtrip/{run_id}     CSV round-trip fidelity test

── Conceptual guide (self-documenting API) ──────────────────────────────────
GET  /api/guide/features              glossary of all 18 features
GET  /api/guide/pipeline              step-by-step pipeline description
GET  /api/guide/metrics               definition of every evaluation metric
GET  /api/guide/panels                explanation of each dashboard panel
```

---

## The Validation Layer

Every numeric result is independently verifiable without re-running the pipeline:

- **`/api/validate/metrics/{run_id}`** — recomputes AUC and balanced accuracy from `feature_matrix.parquet` + `weight_bank.json` and compares against `weight_bank_metrics.json`. If they match within float32 tolerance (1×10⁻⁴), the pipeline, storage, and API are consistent.
- **`/api/validate/checksums/{run_id}`** — SHA-256 of every file in the run directory. Stable across machines if the same seed and data are used.
- **`/api/validate/roundtrip/{run_id}`** — re-parses the CSV export and confirms values match the Parquet source to float32 precision.
- **`/api/ephys/tests`** — 5 biological plausibility assertions on the real dataset that any competent electrophysiologist would recognise as necessary conditions (trough depth, class balance, hard-negative challenge level).

---

## File System Layout at a Glance

```
spike_discrim/
│
├── src/spike_discrim/          core Python package (install with pip -e .)
│   ├── features/
│   │   ├── core_features.py    Tier 1+2: Numba JIT kernels, OP_COUNTS table
│   │   └── event_features.py   Tier 3: full-snippet metrics (12 features)
│   ├── input_layer/
│   │   └── weights.py          WeightBank: quantile-init population code
│   ├── synthetic/
│   │   └── procedural_generator.py  double-exponential spike model + noise
│   ├── benchmarking/
│   │   ├── single_feature.py   Fisher/MI/AUC per-feature ranking
│   │   └── feature_sets.py     5-fold CV for feature set × classifier combos
│   ├── metrics/
│   │   └── evaluation.py       AUC, balanced acc, confusion matrix, Fisher, MI
│   ├── models/
│   │   └── discriminants.py    LDA, logistic, kNN, threshold — sklearn wrappers
│   ├── profiling/
│   │   └── op_counter.py       wall-clock timing + static op-count reporter
│   ├── adapters/
│   │   └── osort_loader.py     MATLAB v5 .mat → canonical Python dict
│   ├── io/
│   │   └── storage.py          NPZ + Parquet read/write helpers
│   └── config/
│       └── loader.py           YAML config loader + validation
│
├── api/                        FastAPI server
│   ├── main.py                 app factory, router registration, static mount
│   └── routers/
│       ├── runs.py             read-only results endpoints
│       ├── pipeline.py         run trigger + job status
│       ├── validate.py         proof-of-correctness endpoints
│       ├── ephys_eval.py       real OSort data evaluation + bio tests
│       └── guide.py            self-documenting feature/metric glossary
│
├── scripts/
│   ├── run_benchmark.py        CLI entry point — runs steps 1–6
│   └── run_lfpy_generation.py  biophysical waveform generation
│
├── configs/
│   ├── default.yaml            all runtime hyperparameters (documented)
│   └── benchmarks.yaml         feature set definitions + classifier list
│
├── data/
│   ├── synthetic/
│   │   ├── waveforms.npz           generated / synthetic dataset
│   │   ├── waveforms_real.npz      real OSort curated dataset (96K snippets)
│   │   └── waveforms_real_meta.csv provenance: source file, cluster, timestamp
│   └── results/<run_id>/
│       ├── config_snapshot.yaml
│       ├── feature_matrix.parquet
│       ├── single_feature_ranks.parquet
│       ├── feature_set_ranks.parquet
│       ├── weight_bank.json
│       └── weight_bank_metrics.json
│
├── ephys_data/
│   └── sort/
│       ├── 5.1/  (32 .mat)     raw OSort output, threshold ×5.1σ
│       ├── 5.2/  (32 .mat)     raw OSort output, threshold ×5.2σ
│       ├── final/ (5 .mat)     human-curated ground truth
│       ├── A00_5-16_2_cells_final copy.xlsx  curation scorecard
│       └── curation_log.npz    machine-readable version of xlsx
│
├── docs/
│   ├── ORIENTATION.md          ← you are here
│   ├── MANIFEST.md             publication-grade design rationale
│   ├── DATA_HIERARCHY.md       file-by-file field reference for all data types
│   ├── features.yaml           feature spec: cost, value, equations
│   └── spike_feature_validation.yaml  full project specification
│
└── tests/                      pytest coverage for features, WeightBank,
    ├── test_features.py         OSort loader, input layer serialisation
    ├── test_input_layer.py
    └── test_osort_adapter.py
```

---

## How to Begin Effectively in 15 Minutes

```bash
# 1. Install and verify
pip install -e .
pytest                          # should show 38 passed

# 2. Run the full pipeline on auto-generated synthetic data
python scripts/run_benchmark.py --tier 2 --no-profile
# → creates data/results/<timestamp>/

# 3. Start the API and explore interactively
.venv/bin/python -m uvicorn api.main:app --port 8099 --reload &
open http://localhost:8099/api/docs    # Swagger UI for every endpoint

# 4. Inspect results programmatically
import pandas as pd
ranks = pd.read_parquet('data/results/<run_id>/single_feature_ranks.parquet')
print(ranks[['rank','feature','fisher_score','auc']].head(10))

# 5. Run biological validation tests on real data
curl http://localhost:8099/api/ephys/tests | python -m json.tool

# 6. Evaluate a trained model against real OSort recordings
curl -X POST http://localhost:8099/api/ephys/evaluate \
     -H 'Content-Type: application/json' \
     -d '{"run_id": "<run_id>", "tier": 2}' | python -m json.tool
```

---

## How Results Feed Research

| What you find | What it means for your work |
|--------------|---------------------------|
| Feature X has Fisher score > 10 on synthetic, < 2 on real | The synthetic model misses a key source of variability in your recording. Improve the waveform generator or collect more curated real data. |
| WeightBank AUC = 0.97 on synthetic, 0.44 on real | The generalisation gap is real and large. Do not deploy this feature set in closed-loop without fine-tuning on real data. |
| `set_B_fast_plus_shape` matches `set_D_combined` balanced accuracy | The extra Tier 3 features add no value — hardware budget is saved. |
| All 5 biological tests pass | The curated real dataset is internally consistent and can be trusted as a validation set. |
| Per-SU recall varies from 0.001 to 0.215 | Unit discrimination is unit-specific. Report per-unit metrics, not just overall AUC. |
| Validation endpoint `metrics` mismatch > 1e-4 | Numerical instability in the pipeline — investigate float32 accumulation. |

---

## Key Design Decisions (with Rationale)

**Why Numba over C/Cython?**  
Pure Python source, no build step, LLVM-optimised, supports `@numba.prange` parallelism. The ~1 s JIT warmup is acceptable for a benchmarking tool; deploy the JIT cache for real-time use.

**Why quantile bins, not k-means?**  
Quantiles are $O(N \log N)$, deterministic, and guarantee equal neuron load across the real data distribution. k-means is better for multimodal distributions but requires iteration and initialization tuning — unjustifiable complexity for a first input layer.

**Why synthetic data first?**  
Controlled variation. You can independently vary amplitude, rise time, and noise level and measure each feature's sensitivity. Real data conflates all these. Synthetic-first allows hypothesis generation; real-data validation tests the hypotheses.

**Why 64 samples / 2.1 ms snippets?**  
Sufficient to capture the full biphasic waveform (trough at ~20 samples, repolarisation peak at ~32–34 samples, AHP decay through ~60 samples) at 30 kHz with 20 samples pre-trough onset buffer. Longer windows add noise without adding discriminative signal.

**Why Parquet for results?**  
Column-oriented, compressed, schema-typed. A 96K-row feature matrix with 18 float32 columns occupies ~6 MB on disk vs ~30 MB CSV, loads in milliseconds, and preserves dtypes across languages.

---

*See [docs/MANIFEST.md](MANIFEST.md) for full mathematical derivations and hyperparameter justification suitable for a Methods section.*  
*See [docs/DATA_HIERARCHY.md](DATA_HIERARCHY.md) for field-by-field documentation of every file type.*
