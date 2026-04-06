# Real-Units Benchmark Results — 2026-04-05

## Executive summary

The current repository now evaluates the curated real waveform bundle from [data/real_units/waveforms_real.npz](data/real_units/waveforms_real.npz), not from the synthetic folder.

Within the scope of the repository, the strongest supported conclusion is:

- **Yes** — the completed notebook benchmark used all rows in the curated real snippet bundle.
- **No** — this does **not** justify claiming that all spikes from the original raw `.ncs` continuous recordings were tested.

The current best-performing feature family on the curated real dataset is the **baseline Tier 1–3 scalar stack plus overlapping-window Temporal MAD/WTA bins**, with the strongest result at **16 time bins**.

For offline benchmarking, the best model in the notebook sweep was `knn_k5`.
For a deployable real-time sorter, `knn_k5` is **not** the recommended implementation target. The best practical direction is:

1. keep the **Temporal MAD/WTA afferent**,
2. flatten it into a fixed feature vector,
3. pair it with a **simple linear gate or a redesigned WeightBank-like afferent layer**, not with k-NN lookup at runtime.

---

## 1. What dataset was actually tested?

### Tested bundle

- [data/real_units/waveforms_real.npz](data/real_units/waveforms_real.npz)
- [data/real_units/waveforms_real_meta.csv](data/real_units/waveforms_real_meta.csv)

### Dataset size

- Total snippets: **96,006**
- Spike-labelled snippets: **38,145**
- Noise-labelled snippets: **57,861**
- Curated single units: **10**
- Curated source files: **5**

### Source-file contribution

| Source file | Events |
|---|---:|
| `A_ss11_Min_sorted_new.mat` | 15,416 |
| `A_ss12_Max_sorted_new.mat` | 18,815 |
| `A_ss12_Min_sorted_new.mat` | 18,035 |
| `A_ss13_Max_sorted_new.mat` | 32,870 |
| `A_ss8_Max_sorted_new.mat` | 10,870 |

### Unit contribution

| Unit ID | Spike count |
|---|---:|
| 1 | 982 |
| 2 | 2,557 |
| 3 | 3,493 |
| 4 | 707 |
| 5 | 5,914 |
| 6 | 3,539 |
| 7 | 2,175 |
| 8 | 15,057 |
| 9 | 742 |
| 10 | 2,979 |

---

## 2. Was testing done on “all spikes in the NCS dataset”?

### Precise answer

**Not in the strongest literal sense.**

The repository currently tests **all snippets in the curated real bundle**, but that bundle is already a derived dataset:

- it comes from curated OSort `.mat` outputs,
- it is already reduced to trough-aligned **64-sample snippets**,
- it only includes the five human-curated `final/` sort files,
- it is not a reader over the original continuous `.ncs` voltage stream.

### Therefore

The valid statement is:

> The benchmark has been run on all **96,006 labelled snippets** in the repository’s curated real-unit dataset.

The invalid over-claim would be:

> all spikes from the raw NCS recording were tested.

That stronger claim is not supported by the present code or data flow.

---

## 3. What testing has actually been completed?

### Automated tests

Full test suite status:

- `42 passed`
- includes:
  - feature tests,
  - input-layer tests,
  - OSort adapter tests.

This means the codebase is currently in a **clean automated-test state**.

### Real-data benchmark status

The analysis notebook completed a full benchmark sweep on the curated real dataset, including:

- single-feature ranking,
- configured feature sets,
- sampled scalar combinations,
- Temporal MAD-only sweeps,
- baseline + Temporal MAD sweeps,
- holdout confusion matrix for the top candidate.

### What is still not fully tested

The investigation is **strong enough for feature selection**, but it is **not yet complete for deployment validation**. Missing items include:

1. raw continuous `.ncs` ingestion and event detection in one loop,
2. real-time latency benchmarking on a streaming sorter path,
3. external-session validation beyond this curated bundle,
4. calibration-time channel-noise MAD from raw baseline voltage,
5. a redesigned afferent/WeightBank path tuned for the temporal feature family.

---

## 4. Main quantitative results

### Best notebook result

Top candidate from the real-data notebook sweep:

- **Feature set:** `baseline_plus_temporal_mad_16bins`
- **Category:** baseline plus temporal afferent
- **Model:** `knn_k5`
- **Number of features:** 34
- **5-fold CV balanced accuracy:** **0.901714**
- **5-fold CV AUC:** **0.951466**
- **Holdout balanced accuracy:** **0.903750**
- **Holdout AUC:** **0.951131**

### Holdout confusion matrix

| | Pred noise | Pred spike |
|---|---:|---:|
| **True noise** | 13,664 | 802 |
| **True spike** | 1,307 | 8,229 |

Derived summary:

- Precision: **0.9112**
- Recall: **0.8629**
- False-positive rate: **0.0554**

---

## 5. Feature-set tradeoff table

The table below compares representative candidates on the full curated real bundle.

| Feature set | Model | Balanced acc. | AUC | Interpretation |
|---|---|---:|---:|---|
| `set_A_ultra_fast` | `knn_k5` | 0.869140 | 0.925299 | Strong low-cost baseline using only 5 causal scalar features |
| `set_D_combined` | `knn_k5` | 0.894425 | 0.946733 | Tier 1–3 scalar stack is already competitive |
| `temporal_mad_only_16bins` | `knn_k5` | 0.815336 | 0.885564 | Temporal afferent alone carries real information |
| `baseline_plus_temporal_mad_16bins` | `knn_k5` | 0.901714 | 0.951466 | Best observed discrimination; temporal bins add value on top of scalar stack |
| `set_D_combined` | `logistic_regression` | 0.813887 | 0.877995 | More deployment-friendly than k-NN, but weaker |
| `baseline_plus_temporal_mad_16bins` | `logistic_regression` | 0.822032 | 0.891609 | Temporal bins also help a linear-style model |
| `baseline_plus_temporal_mad_16bins` | `lda` | 0.809541 | 0.882282 | Similar story for another simple model |
| `baseline_plus_temporal_mad_16bins` | `weight_bank` | 0.500536 | 0.673009 | Current WeightBank does not yet exploit this feature family well |

---

## 6. What do the results mean scientifically?

### 6.1 Single-feature takeaway

The strongest single metrics are still classic waveform quantities such as:

- `peak_amplitude`,
- `ev_peak_amplitude`,
- `ev_absolute_area`,
- `max_slope`,
- `abs_window_sum_peak`.

So the repository confirms that basic waveform morphology remains highly informative.

### 6.2 Temporal afferent takeaway

The Temporal MAD/WTA features help most when they are **added to** the scalar stack rather than used alone.

Observed pattern from the notebook sweep:

- 3 bins improves over baseline,
- 5 bins improves further,
- 8 bins improves further,
- 12 bins is slightly better again,
- 16 bins is best among the tested values.

That is consistent with the scientific intuition that multiple timesteps preserve discriminative waveform state that scalar extrema alone discard.

### 6.3 Noise-robust interpretation

The current implementation uses local median-centering and optional MAD normalization, which is a reasonable repository-level stand-in for channel-noise calibration.

But the gold-standard interpretation remains:

- compute channel noise MAD during a calibration phase on baseline voltage,
- use that to normalize or place afferent thresholds,
- then apply the temporal bins to snippets in deployment.

---

## 7. Computational tradeoff for a real sorter

### Best raw accuracy: not the best deployment model

`knn_k5` wins the offline benchmark, but it is a poor final choice for a real-time sorter because it requires:

- storing many reference examples,
- repeated distance calculations at inference time,
- runtime cost that grows with dataset size.

That is fine for scientific ranking, but not ideal for embedded or hard real-time use.

### Better implementation target

For a real sorter, the better design target is:

1. **feature front end:**
	- Tier 1–2 scalar features,
	- Temporal MAD/WTA bins with modest overlap,
	- likely in the range of **8–16 bins**,

2. **decision layer:**
	- a small linear gate,
	- or a redesigned afferent bank / WeightBank that is aware of temporal-bin structure.

### Current WeightBank status

The present WeightBank underperforms badly on the real curated benchmark.

This should not be interpreted as proof that the afferent idea is wrong.
It more likely means the current WeightBank formulation is not yet matched to:

- heterogeneous real-waveform distributions,
- the mixed scalar + temporal feature stack,
- or the way bin centers should be calibrated from real noise statistics.

### Practical recommendation

If the goal is a sorter that can plausibly ship, the current evidence supports:

- **keep** the temporal feature family,
- **reduce** the search space to something like 8, 12, or 16 bins,
- **prioritize** a simple deployable discriminant after feature extraction,
- treat `knn_k5` as a **scientific upper-bound reference**, not the production endpoint.

---

## 8. Refactor completed in this pass

Real curated data now lives under:

- [data/real_units/waveforms_real.npz](data/real_units/waveforms_real.npz)
- [data/real_units/waveforms_real_meta.csv](data/real_units/waveforms_real_meta.csv)

Code and docs were updated so that:

- the benchmark script help text points to the new location,
- API real-data evaluation routes point to the new location,
- pipeline launch helpers point to the new location,
- notebooks and docs were updated to describe the real dataset under `data/real_units`.

---

## 9. Final assessment

### Has the scientific investigation been carried far enough?

**Yes for feature prioritization.**

The repository now has enough evidence to make a serious engineering decision:

- temporal bins are worth keeping,
- temporal MAD/WTA features improve discrimination on real curated data,
- 16 bins was best among the tested settings,
- pure scalar sets remain strong low-cost baselines,
- the current WeightBank is not yet competitive on the real dataset.

### Is the project scientifically finished?

**No.**

Before claiming final deployment readiness, the next missing work is:

1. evaluate on raw continuous recordings or a closer streaming proxy,
2. integrate calibration-time channel-noise MAD from baseline voltage,
3. redesign the runtime discriminant for the temporal feature family,
4. validate on more sessions and channels beyond this curated bundle.

---

## 10. Recommended next step

The highest-value next step is:

> implement a deployment-oriented discriminant for the `set_D_combined + Temporal MAD` family, then compare 8, 12, and 16 bins under a streaming latency harness.

That would turn the current result from a strong offline finding into a directly actionable sorter-design choice.
