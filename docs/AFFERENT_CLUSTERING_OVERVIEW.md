# Afferent Clustering Overview

This note explains the new post-hoc afferent clustering path that was added to the repository.

---

## Why this was added

The repo already had a good **stage-1 gate**:

- extract waveform features
- map them through the input layer
- separate likely real spikes from noise

But that was still mostly a **binary decision**.

The new addition starts to address the second problem:

> Once an event looks like a spike, does the pattern of afferent activity contain enough structure to separate different units?

The new benchmark does **not** yet implement a full online spiking winner-take-all learning layer.
Instead, it provides the correct intermediate representation and a systematic way to test whether that representation is useful for unit separation.

---

## Core idea

For every known spike:

1. Extract features from the waveform.
2. Pass those features through the existing `WeightBank` input layer.
3. Store the resulting afferent activation pattern.
4. Cluster those activation patterns post hoc.
5. Compare the cluster assignments against known `unit_id` labels.

This gives a direct test of:

- which afferent pattern family is most structured,
- which feature family best separates units,
- whether the input layer is already close to supporting unit discrimination.

---

## What `WeightBank` now does

File: [src/spike_discrim/input_layer/weights.py](src/spike_discrim/input_layer/weights.py)

Previously, `WeightBank` mostly returned:

- a scalar spike-likeness score per snippet
- a thresholded spike/noise decision

Now it can also return the **full afferent activation tensor**.

### New methods

- `project_snippet()`
- `project_batch()`

### Shapes

Let:

- $N$ = number of snippets
- $F$ = number of extracted features
- $B$ = number of bins / tuning neurons per feature

Then:

- feature matrix: $X \in \mathbb{R}^{N \times F}$
- afferent tensor: $A \in \mathbb{R}^{N \times F \times B}$

Each scalar feature is turned into a small population-code activation pattern across `B` tuned afferents.

So instead of only keeping the final score, the repo now preserves the pattern that produced that score.

---

## New benchmark module

File: [src/spike_discrim/benchmarking/afferent_clustering.py](src/spike_discrim/benchmarking/afferent_clustering.py)

Main function:

- `run_afferent_clustering_benchmark()`

### What it does

It filters to known spikes:

- `class_label == 1`
- `unit_id > 0`

Then it:

1. fits a `WeightBank` on spike features
2. projects spikes into afferent space
3. groups features into families
4. flattens those afferent responses for clustering
5. runs unsupervised clustering (`KMeans`)
6. scores the result against known units

---

## Feature families currently compared

The benchmark tests these families separately:

### `scalar`
Base scalar features such as:

- `peak_amplitude`
- `trough_amplitude`
- `max_slope`
- `min_slope`
- `max_abs_curvature`
- `abs_window_sum_peak`

### `temporal_mad`
Temporal afferent bins:

- `mad_wta_bin_00`
- `mad_wta_bin_01`
- ...

These come from the overlapping-window MAD/WTA code and are currently the strongest family in the recent run.

### `event`
Event-level waveform descriptors such as:

- `ev_trough_to_peak_time_samples`
- `ev_half_width_samples`
- `ev_baseline_rms`
- `ev_zero_crossing_count`
- etc.

### `full`
All extracted features together.

---

## What gets saved

The main pipeline now writes post-hoc afferent outputs into each run directory.

For example, in a run like:

- [data/results/20260405_200722](data/results/20260405_200722)

there is now a folder:

- [data/results/20260405_200722/afferent_outputs](data/results/20260405_200722/afferent_outputs)

That folder contains files like:

- [data/results/20260405_200722/afferent_outputs/scalar.npz](data/results/20260405_200722/afferent_outputs/scalar.npz)
- [data/results/20260405_200722/afferent_outputs/temporal_mad.npz](data/results/20260405_200722/afferent_outputs/temporal_mad.npz)
- [data/results/20260405_200722/afferent_outputs/event.npz](data/results/20260405_200722/afferent_outputs/event.npz)
- [data/results/20260405_200722/afferent_outputs/full.npz](data/results/20260405_200722/afferent_outputs/full.npz)

and assignment/metadata files such as:

- `*_assignments.parquet`
- `*_metadata.json`

The run directory also includes:

- [data/results/20260405_200722/afferent_cluster_summary.json](data/results/20260405_200722/afferent_cluster_summary.json)
- `afferent_cluster_ranks.parquet`

---

## What is inside each afferent output file

Each family `.npz` stores:

- `afferent_outputs`
- `unit_ids`
- `feature_indices`
- `feature_names`

### Example interpretation

If the family is `temporal_mad` and there are:

- 8 temporal features
- 10 bins per feature

then each spike becomes a flattened vector of length:

$$
8 \times 10 = 80
$$

If the full stack has 26 features and 10 bins each, then each spike becomes:

$$
26 \times 10 = 260
$$

So clustering happens on the **afferent response code**, not just on the original feature values.

---

## How clustering quality is scored

The new benchmark computes:

### `matched_accuracy`
Clusters are matched to known unit labels post hoc using an optimal assignment.
This is the easiest “how separable are they really?” metric to interpret.

### `ARI`
Adjusted Rand Index.
Measures how well cluster grouping matches the true unit grouping.

### `NMI`
Normalized Mutual Information.
Measures shared information between clusters and true labels.

### `purity`
For each cluster, how dominant is its main true unit?

### `silhouette`
Measures how well-separated the learned clusters are in the afferent space itself.

### `inertia`
Within-cluster compactness from `KMeans`.

---

## What the latest run showed

From:

- [data/results/20260405_200722/afferent_cluster_summary.json](data/results/20260405_200722/afferent_cluster_summary.json)

The summary was:

- `n_spike_events = 900`
- `n_units = 3`
- `n_bins = 10`
- `best_family = temporal_mad`
- `best_matched_accuracy = 0.9256`
- `best_ari = 0.7950`

### Interpretation

This is encouraging.
It means the temporal MAD afferent code is already carrying substantial unit-specific structure.

In other words:

- the current input representation is not only useful for spike-vs-noise rejection
- it is also beginning to support spike-vs-spike separation

---

## What this does **not** mean yet

This is still a **post-hoc unsupervised benchmark**.

It is **not yet** a learned second spiking layer with:

- membrane accumulation across time
- winner-take-all threshold crossings
- lateral inhibition
- Hebbian or STDP updates

So this new step should be viewed as:

> a representation-quality test for the input afferent layer

not yet the final unit-learning network.

---

## Why this step matters

Before building a real unsupervised WTA layer, it is useful to ask:

> Does the current afferent code even contain enough structure to separate units?

If the answer is no, then adding a complicated competitive layer is unlikely to help.
If the answer is yes, then a true unsupervised learning layer becomes worth implementing.

This new benchmark gives that answer directly.

---

## Current pipeline structure

### Stage 1 — feature extraction
File: [src/spike_discrim/features/extraction.py](src/spike_discrim/features/extraction.py)

Waveforms are converted into a feature matrix.

### Stage 2 — input-layer projection
File: [src/spike_discrim/input_layer/weights.py](src/spike_discrim/input_layer/weights.py)

Features are projected into afferent tuning-bank activations.

### Stage 3 — post-hoc unsupervised clustering
File: [src/spike_discrim/benchmarking/afferent_clustering.py](src/spike_discrim/benchmarking/afferent_clustering.py)

Those afferent codes are clustered and compared to known unit identities.

---

## Tests added

A new test file was added:

- [tests/test_afferent_clustering.py](tests/test_afferent_clustering.py)

It checks:

- `WeightBank.project_batch()` shape and range
- clustering benchmark output generation
- creation of saved clustering artifacts

---

## What to read next

If you want to inspect the implementation in code, read these in order:

1. [src/spike_discrim/input_layer/weights.py](src/spike_discrim/input_layer/weights.py)
2. [src/spike_discrim/benchmarking/afferent_clustering.py](src/spike_discrim/benchmarking/afferent_clustering.py)
3. [scripts/run_benchmark.py](scripts/run_benchmark.py)
4. [configs/default.yaml](configs/default.yaml)
5. [tests/test_afferent_clustering.py](tests/test_afferent_clustering.py)

---

## Best next step

The natural next step is to replace post-hoc clustering with a true learned second layer, for example:

- competitive neurons
- leaky accumulation over input afferents
- winner-take-all competition
- winner-only learning rule

That would let the repo test:

- whether an online unsupervised layer can learn unit identities from the afferent code
- whether the best post-hoc family (`temporal_mad` so far) is also the best learnable family

---

## One-sentence summary

The repo now saves the full input-afferent responses for known spikes, clusters those responses by feature family, and uses true `unit_id` labels only afterward to measure which afferent code best separates putative units.
