

# spike_discrim

`spike_discrim` is a repository for evaluating waveform representations for real-time single-unit decoding.

The working question is fairly simple:

> Given a 64-sample extracellular waveform snippet, which input representation is cheap enough for closed-loop use, good enough at rejecting noise, and structured enough to preserve unit-specific information?

This is not a full offline sorter. It is closer to an input-layer workbench.

At present it can:

- benchmark spike-vs-noise features on synthetic and curated real snippets
- build and score a population-coded input layer (`WeightBank`)
- rank individual features and feature sets
- store afferent activations for known spikes
- cluster those afferent patterns post hoc to see which feature family appears to separate units best

See [docs/AFFERENT_CLUSTERING_OVERVIEW.md](docs/AFFERENT_CLUSTERING_OVERVIEW.md) for the afferent clustering path.

---

## What problem this repo is solving

The intended use case is fairly practical. Typical questions are:

- Can I reject obvious noise in real time?
- Can I preserve enough waveform structure to distinguish units later?
- Which features are worth carrying into a downstream spiking layer?
- How much computation am I paying for each feature?

This repository is organized around those questions.

It compares candidate waveform features using:

- **procedural / synthetic waveforms** for controlled experiments
- **curated real OSort snippets** for reality checks

The output is a set of fairly inspectable artifacts:

- ranked feature tables
- ranked feature-set tables
- input-layer parameters (`WeightBank`)
- profiling summaries
- afferent activation files for clustering and inspection

---

## Pipeline sketch

```text
Waveform snippet (64 samples @ 30 kHz)
                    │
                    ├─► Tier 1 / 2 / 3 feature extraction
                    │      - scalar features
                    │      - event features
                    │      - temporal MAD / WTA afferent bins
                    │
                    ├─► WeightBank input layer
                    │      - quantile-defined centers per feature
                    │      - Gaussian tuning neurons per feature
                    │      - fast spike-likeness score
                    │
                    ├─► Stage 1 output
                    │      - spike vs noise gate
                    │
                    └─► Stage 2 analysis
                                    - store afferent activations for known spikes
                                    - cluster them post hoc
                                    - measure which input pattern best separates units
```

In plain terms:

- **Stage 1** asks whether the event looks like a spike at all
- **Stage 2** asks whether the resulting afferent code appears to carry unit-specific structure

---

## Current implementation

### 1. Spike-vs-noise benchmarking

The repo extracts features from waveform snippets and evaluates them with:

- Fisher score
- mutual information
- AUC
- balanced accuracy
- simple interpretable classifiers
- a custom input-layer population code (`WeightBank`)

### 2. Fast input-layer population code

The input layer lives in [src/spike_discrim/input_layer/weights.py](src/spike_discrim/input_layer/weights.py).

`WeightBank` does two things:

- scores how spike-like a feature vector is
- projects that feature vector into the underlying afferent activation space

That second capability is useful because it exposes the input code itself rather than only the final scalar score.

### 3. Post-hoc afferent clustering

The repo now stores afferent outputs for known spikes and clusters them by feature family.

This is the current bridge between reasonable noise rejection and a more serious question about unit separation.

The implementation is in [src/spike_discrim/benchmarking/afferent_clustering.py](src/spike_discrim/benchmarking/afferent_clustering.py).

---

## Current status

The repository is already usable as a front-end design and benchmarking tool.

It is **not yet** a full online unsupervised WTA spiking network with:

- membrane accumulation across time
- winner-take-all output neurons
- online Hebbian / STDP learning

It does, however, now answer a more interesting intermediate question:

> Which afferent code preserves the most unit-specific structure?

That is usually worth knowing before investing in a more elaborate downstream layer.

---

## Quick start

### Install

```bash
pip install -e .
```

Optional if you want the LFPy / NEURON synthetic generator:

```bash
pip install -e ".[lfpy]"
```

### Run tests

```bash
pytest
```

### Run the benchmark pipeline

```bash
python scripts/run_benchmark.py --tier 3 --no-profile --quiet
```

This will:

- load or generate waveform data
- extract the configured feature stack
- fit the input layer
- benchmark single features
- benchmark feature sets
- cluster afferent outputs for known spikes
- write a timestamped run into `data/results/`

### Generate synthetic LFPy data

```bash
python scripts/run_lfpy_generation.py \
     --output-dir data/synthetic \
     --n-distances 8 \
     --n-angles 16
```

---

## Where to start

There are two straightforward entry points.

### Option A — just run the pipeline

If a new benchmark run is all you need:

```bash
python scripts/run_benchmark.py --tier 3 --no-profile --quiet
```

Then inspect the latest folder under `data/results/`.

Most of the time, these are the useful files:

- `run_summary.json`
- `single_feature_ranks.parquet`
- `feature_set_ranks.parquet`
- `weight_bank.json`
- `weight_bank_metrics.json`
- `afferent_cluster_summary.json`
- `afferent_outputs/`

### Option B — use the notebooks

If you want to inspect things visually:

- [notebooks/main.ipynb](notebooks/main.ipynb) — the main end-to-end analysis notebook
- [notebooks/repo_process_overview.ipynb](notebooks/repo_process_overview.ipynb) — high-level repo walkthrough
- [notebooks/osort_file_extraction.ipynb](notebooks/osort_file_extraction.ipynb) — real OSort extraction workflow

Notebook usage is roughly:

- start with `repo_process_overview.ipynb` if you want orientation
- use `main.ipynb` if you want actual benchmarking, figures, and scoreboards
- use `osort_file_extraction.ipynb` only when touching the real-data extraction path

---

## Run outputs

Each benchmark run creates a timestamped folder under `data/results/`, for example:

```text
data/results/20260405_200722/
├── config_snapshot.yaml
├── waveform_summary.json
├── feature_metadata.json
├── temporal_mad_metadata.json
├── feature_matrix.parquet
├── single_feature_ranks.parquet
├── feature_set_ranks.parquet
├── weight_bank.json
├── weight_bank_metrics.json
├── afferent_cluster_summary.json
├── afferent_cluster_ranks.parquet
├── afferent_outputs/
│   ├── scalar.npz
│   ├── temporal_mad.npz
│   ├── event.npz
│   ├── full.npz
│   └── *_assignments.parquet
└── run_summary.json
```

### Files that tend to matter

#### `single_feature_ranks.parquet`
Which individual features are strongest?

#### `feature_set_ranks.parquet`
Which combined feature stacks work best?

#### `weight_bank_metrics.json`
How strong is the stage-1 spike gate?

#### `afferent_cluster_summary.json`
Which afferent family best preserves unit structure?

#### `afferent_outputs/*.npz`
Stored input-layer activation codes for known spikes.

---

## Example from the current repo state

In a recent run:

- top single feature was `mad_wta_bin_06`
- top feature set was `set_E_combined_plus_temporal_mad`
- top afferent family was `temporal_mad`

At least in that run, the overlapping-window temporal MAD / WTA representation was the strongest part of the input code for both:

- spike-vs-noise separation
- post-hoc unit-structure separation

That is generally the kind of answer this repo is meant to produce.

---

## Repository layout

```text
spike_discrim/
├── src/spike_discrim/
│   ├── features/        # feature extraction kernels and shared extraction path
│   ├── input_layer/     # WeightBank population-coded input layer
│   ├── benchmarking/    # feature ranking, feature-set ranking, afferent clustering
│   ├── profiling/       # operation counting and timing
│   ├── synthetic/       # procedural and LFPy waveform generation
│   ├── adapters/        # OSort / MATLAB loading
│   ├── models/          # simple classifiers and future downstream models
│   ├── metrics/         # evaluation utilities
│   ├── io/              # NPZ / Parquet / JSON storage helpers
│   └── config/          # config loading
├── configs/
│   ├── default.yaml
│   └── benchmarks.yaml
├── data/
│   ├── synthetic/
│   ├── results/
│   └── profiling/
├── notebooks/
├── docs/
└── tests/
```

### If you only open five files

Open these:

1. [scripts/run_benchmark.py](scripts/run_benchmark.py)
2. [src/spike_discrim/features/extraction.py](src/spike_discrim/features/extraction.py)
3. [src/spike_discrim/input_layer/weights.py](src/spike_discrim/input_layer/weights.py)
4. [src/spike_discrim/benchmarking/afferent_clustering.py](src/spike_discrim/benchmarking/afferent_clustering.py)
5. [notebooks/main.ipynb](notebooks/main.ipynb)

---

## Practical notes

### 1. Start with the full 64-sample snippet

For benchmarking, keep the full snippet.
It is the cleaner way to judge whether a feature family is intrinsically useful.

### 2. Treat the current input layer as a front end, not the final network

Right now the repo is strongest as a tool for:

- designing the input representation
- measuring computational cost
- identifying whether unit structure is present in the afferent code

### 3. Use the afferent clustering outputs before building a new learning rule

If afferent clustering is weak, a more complex unsupervised downstream layer is less compelling. If it is strong, the next layer begins to look justified.

### 4. Use the notebooks when you want a visual check

The notebooks are the fastest route to:

- waveform inspection
- feature scoreboards
- temporal sweep figures
- best-model confusion matrices

### 5. Profile only when you need cost numbers

Use `--no-profile` during rapid iteration.
Turn profiling back on when you want true compute/cost comparisons.

---

## Likely next step

The obvious next step is a real second layer:

- online accumulation of afferent input
- winner-take-all output neurons
- unsupervised learning on spike-passing events

The repository now has enough substrate to make that next step testable.

---

## Summary

If the aim is to decide **which waveform code is worth carrying into a real-time single-unit decoder**, this repo is already reasonably useful.

It provides:

- fast benchmarking
- realistic outputs
- interpretable input-layer behavior
- a tentative bridge from simple noise rejection to actual unit discrimination

If the goal is closed-loop experimentation, that is usually enough to make it worth keeping around.

## Architecture

```
Input waveform (64 samples @ 30 kHz)
         │
    ┌────▼─────────────────────────────────┐
    │  Tier 1 features (Numba JIT)         │  ← amplitude, d1, d2
    │  Tier 2 features (Numba JIT)         │  ← abs-window-sum, energy
    └────┬─────────────────────────────────┘
         │  scalar summaries per snippet
    ┌────▼─────────────────────────────────┐
    │  WeightBank (input layer)            │  ← quantile-bin population code
    │  Gaussian tuning neurons             │  ← real spikes → high activation
    └────┬─────────────────────────────────┘
         │  discriminant score ∈ [0, 1]
    ┌────▼─────────────────────────────────┐
    │  pass / fail gate (threshold)        │
    └──────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install (core deps)
pip install -e .

# 2. Install LFPy + NEURON for synthetic generation
pip install -e ".[lfpy]"

# 3. Generate synthetic LFPy dataset
python scripts/run_lfpy_generation.py --output-dir data/synthetic --n-distances 8 --n-angles 16

# 4. Run full benchmarks
python scripts/run_benchmark.py --data-dir data/synthetic --results-dir data/results

# 5. Run tests
pytest
```

All outputs are written to `data/` as human-inspectable NPZ (waveforms) and
Parquet (feature tables + metrics), plus JSON profiling logs.

---

## Project Layout

```
spike_discrim/
├── src/spike_discrim/
│   ├── features/        ← Numba-jitted Tier 1/2/3 kernels (zero-copy)
│   ├── input_layer/     ← WeightBank: quantile-initialized population code
│   ├── profiling/       ← Op counter + wall-clock timer
│   ├── synthetic/       ← LFPy (biophysical) + procedural generators
│   ├── adapters/        ← osort MATLAB (.mat) → canonical Python schema
│   ├── benchmarking/    ← Single-feature and feature-set benchmark runners
│   ├── models/          ← Interpretable classifiers (threshold, LDA, LR, SVM)
│   ├── metrics/         ← AUC, F1, silhouette, Fisher score
│   ├── io/              ← NPZ + Parquet read/write
│   └── config/          ← YAML config loader + schema validation
├── configs/
│   ├── default.yaml     ← All runtime hyperparameters (documented)
│   └── benchmarks.yaml  ← Feature set definitions and classifier lists
├── data/
│   ├── synthetic/       ← Generated synthetic waveforms (NPZ + Parquet labels)
│   ├── real_units/      ← Converted real OSort snippets + provenance CSV
│   ├── results/         ← Benchmark outputs (timestamped Parquet + JSON)
│   └── profiling/       ← Op count + timing results (JSON)
├── docs/
│   ├── MANIFEST.md      ← Publication-grade design rationale
│   ├── features.yaml    ← Feature spec with cost/value scores
│   └── spike_feature_validation.yaml  ← Full project specification
├── tests/               ← Pytest unit tests
└── scripts/             ← Entry-point scripts
```

---

## Data Transparency

Every run writes a timestamped subdirectory under `data/results/`:

```
data/results/20260405_143022/
├── config_snapshot.yaml        ← exact config used
├── waveform_summary.json       ← dataset statistics
├── temporal_mad_metadata.json  ← overlapping-window MAD/WTA layout (when enabled)
├── single_feature_ranks.parquet
├── feature_set_ranks.parquet
├── weight_bank.json
├── weight_bank_metrics.json
├── run_summary.json
└── profiling/
    ├── first_derivative_*.json
    └── ...
```

Waveform arrays are stored as `float32` NPZ with separate label Parquet files
so any downstream tool can inspect results without this package.

---

## Computational Efficiency

All feature kernels are `@numba.njit(parallel=True, cache=True, fastmath=True)`
and operate on contiguous `float32` memory views with pre-allocated output
buffers. No array copies occur in the hot path.

Op counts (adds, multiplies, memory accesses) per sample are documented in
[`docs/features.yaml`](docs/features.yaml) and verified by the profiling module.

---

## Publications

If this benchmarking framework identifies optimal features for a deployed
real-time SNN, a methods paper will cite this repository. All hyperparameter
decisions and rationale are documented in [`docs/MANIFEST.md`](docs/MANIFEST.md)
for direct inclusion in Methods sections.
