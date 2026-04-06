# Data Hierarchy & Validation Reference
**Session: DA057 — 5-16_2 | OSort v300 | Cheetah DAQ | 30 kHz**

---

## 1. Folder Structure

```
spike_discrim/
├── ephys_data/
│   ├── sort/
│   │   ├── 5.1/                    # Tier 1a — raw OSort output, threshold ×5.1σ (32 channels)
│   │   │   ├── A_ss1_sorted_new.mat
│   │   │   └── ... (A_ss2 … A_ss32)
│   │   ├── 5.2/                    # Tier 1b — raw OSort output, threshold ×5.2σ (32 channels)
│   │   │   ├── A_ss1_sorted_new.mat
│   │   │   └── ... (A_ss2 … A_ss32)
│   │   ├── final/                  # Tier 2 — human-curated ground truth (5 channels)
│   │   │   ├── A_ss8_Max_sorted_new.mat
│   │   │   ├── A_ss11_Min_sorted_new.mat
│   │   │   ├── A_ss12_Max_sorted_new.mat
│   │   │   ├── A_ss12_Min_sorted_new.mat
│   │   │   └── A_ss13_Max_sorted_new.mat
│   │   ├── A00_5-16_2_cells_final copy.xlsx  # Tier 0 — human curation scorecard
│   │   └── curation_log.npz                  # Tier 0 — machine-readable version of xlsx
│   └── 001641/                     # Tier 3 — DANDI NWB files (spike times only, no raw voltage)
│       ├── sub-31/ … sub-44/       # 6 subjects (IDs: 31, 32, 34, 35, 42, 44)
│       └── ...                     # 70 NWB files total, ~214 unit-sessions, ~17.6M spikes
└── data/
    ├── synthetic/
    │   ├── waveforms.npz           # Synthetic benchmark dataset
    │   └── labels.parquet          # Synthetic labels / provenance
    └── real_units/
        ├── waveforms_real.npz      # Pipeline-compatible labelled dataset (96,006 snippets)
        └── waveforms_real_meta.csv # Human-readable sidecar with source file / cluster provenance
```

---

## 2. Tier 0 — Human Curation Scorecard

### File: `ephys_data/sort/A00_5-16_2_cells_final copy.xlsx`

**Sheet:** `'1'` | **Header row:** index 9 (row 10 in Excel numbering)

| Column | Variable Name | Type | Meaning |
|--------|---------------|------|---------|
| `File` | `channel` | int 1–32 | Electrode channel → `A_ss{channel}_sorted_new.mat` |
| `HS` | — | str | Headstage hardware label (e.g. `'B'`); **not** the quality code |
| `Ch` | — | float | Continuous channel index (same as `File`) |
| `Nr SUA` | `n_sua` | int | Number of single units accepted on this channel |
| `Th used` | `threshold_used` | str | OSort threshold multiplier used: `'5.1'`, `'5.2'`, or `'5.1;5.2'` |
| `Clusters to Use` | `cluster_ids_raw` | str | OSort cluster IDs accepted (comma/semicolon-delimited; `+ID` = merged cluster) |
| `Notes` | `quality_code` | int | **Signal quality: `0`=no signal, `2`=ephys/no neurons, `4`=has neurons** |
| `Max` / `Min` | — | str | Which deflection polarity was sorted (Max=positive, Min=negative) |

**Channels with confirmed neurons (`quality_code == 4`):**

| Channel | Nr SUA | Threshold | Cluster IDs | Polarity |
|---------|--------|-----------|-------------|----------|
| ss8 | 2 | 5.1 | 1112, 1137 | Max (+) |
| ss11 | 1 | 5.2 | 2498 | Min (−) |
| ss12 | 2 | 5.1 & 5.2 | 3303 (5.1), 2962 (5.2) | Max, Min |
| ss13 | 5 | 5.1 | 2557, 2907, 2911, 2961, 2963 | Max (+) |

**Machine-readable copy:** `curation_log.npz`
```python
log = np.load('ephys_data/sort/curation_log.npz', allow_pickle=True)
# Arrays: channels, quality_code, n_sua, has_neurons, threshold_used, cluster_ids_raw
active_channels = log['channels'][log['has_neurons']]   # [8, 11, 12, 13]
```

---

## 3. Tier 1 — Raw OSort Output (5.1/ and 5.2/)

### Files: `A_ss{N}_sorted_new.mat`  (N = 1–32)

MATLAB v5 format, OSort version 300, Cheetah DAQ at 30 kHz.

**Key arrays and their meaning:**

| Key | Shape | dtype | Units | Description |
|-----|-------|-------|-------|-------------|
| `newSpikesNegative` | (N, 256) | float64 | ADU | Raw threshold-crossing waveforms (256 samples ≈ 8.5 ms) |
| `allSpikesCorrFree` | (N, 256) | float64 | ADU | Artifact-corrected waveforms (preferred for analysis) |
| `newTimestampsNegative` | (1, N) | float64 | µs | UNIX microsecond timestamps |
| `assignedNegative` | (1, N) | int32 | — | OSort cluster ID for each event |
| `useNegative` | (K, 1) | uint16 | — | OSort-accepted SU cluster IDs |
| `useNegativeExcluded` | (0, 0) | — | — | Empty in Tier 1 (human rejection only in final/) |
| `noiseTraces` | (84, 1) | float64 | ADU | 84-sample noise power profile (not full-length templates) |
| `scalingFactor` | (1, 1) | float64 | V/ADU | Hardware calibration: 9.155273×10⁻⁸ V/ADU |
| `stdEstimateOrig` | (1, 1) | float64 | ADU | Pre-filter noise σ (used to set detection threshold) |
| `paramsUsed` | (1, 2) | float64 | — | `[0, threshold_multiplier]` → 5.1 or 5.2 |
| `*Positive` arrays | (0, 0) | — | — | All empty; positive deflections not sorted |

**Unit conversion:**
```
waveforms_µV = waveforms_ADU × scalingFactor × 1e6
```
Example: `scalingFactor = 9.155273e-8` → **0.0916 µV per ADU count**

**Timestamp decoding:**
```python
t_us  = mat['newTimestampsNegative'].ravel()   # UNIX microseconds (e.g. 1747419923083212)
t_rel = (t_us - t_us[0]) / 1e6                # seconds from first event (0 … ~1950 s)
```

**Cluster ID taxonomy:**

| Cluster ID | Meaning | Include? |
|-----------|---------|---------|
| `0` or `1` | OSort singletons (1 event, edge case) | ✗ discard |
| `1000–9999` | OSort template clusters | depends on `useNegative` |
| Listed in `useNegative` | OSort auto-accepted single units | ✓ spike |
| Not in `useNegative`, not `99999999` | Rejected by algorithm | context-dependent |
| `99999999` | OSort hard noise sentinel (oscillations, saturations, cross-talk) | ✗ noise |

**Threshold comparison (ss1 example):**

| Threshold | Total events | SU events | Noise (99M) |
|-----------|-------------|-----------|-------------|
| 5.1× | 3,025 | 1,075 | 1,949 |
| 5.2× | 2,868 | fewer | fewer |

Lower threshold → more detections → more noise + more spike candidates.

**Loading with the notebook helper:**
```python
d = load_osort_mat('ephys_data/sort/5.1/A_ss13_sorted_new.mat')
# d['waveforms_uv']   → (N, 256) float64 µV
# d['timestamps_rel'] → (N,) float64 seconds
# d['cluster_ids']    → (N,) int32
# d['su_ids']         → list of accepted cluster IDs
```

---

## 4. Tier 2 — Human-Curated Files (final/)

### Files: `A_ss{N}_{Max|Min}_sorted_new.mat`

Identical structure to Tier 1, with one additional key:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `useNegativeExcluded` | (M, 1) | uint16 | Cluster IDs **visually rejected** by the experimenter after OSort acceptance |

These rejected clusters look like real spikes but were excluded by human review — they are the most physiologically realistic noise class and are **ideal hard negatives** for discriminator training.

**Final/ file inventory:**

| File | SU IDs | n_SU | n_excluded | n_noise(99M) | Noise σ (µV) |
|------|--------|------|-----------|--------------|--------------|
| A_ss8_Max | 1112, 1137 | 3,721 | 6,695 | 454 | 28.35 |
| A_ss11_Min | 2498 | 982 | 12,953 | 1,481 | 4.88 |
| A_ss12_Max | 3303 | 2,557 | 14,158 | 2,100 | 5.15 |
| A_ss12_Min | 2962 | 3,493 | 12,595 | 1,947 | 5.15 |
| A_ss13_Max | 2557, 2907, 2911, 2961, 2963 | 27,392 | 3,540 | 1,938 | 11.87 |
| **Total** | 10 SUs | **38,145** | **49,941** | **7,920** | — |

**Three-class label scheme:**

| Source | Class Label | `unit_id` |
|--------|-------------|-----------|
| `useNegative` events | `1` (spike) | `1..K` (unique per SU) |
| `useNegativeExcluded` events | `0` (noise) | `0` |
| cluster `99999999` events | `0` (noise) | `0` |
| clusters `0` or `1` | discarded | — |

---

## 5. Pipeline-Compatible Dataset

### File: `data/real_units/waveforms_real.npz`

Generated by `osort_file_extraction.ipynb` — 96,006 labelled waveform snippets, sub-sampled from 256 → 64 samples centred on the trough (compatible with the spike_discrim pipeline's default `n_samples=64`).

**Arrays:**

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `waveforms` | (96006, 64) | float32 | µV amplitudes, trough-aligned 64-sample window |
| `class_labels` | (96006,) | int32 | `0`=noise, `1`=spike |
| `unit_ids` | (96006,) | int32 | `0`=noise, `1..10`=SU identity |

**Class balance:** 39.7% spikes / 60.3% noise (reflecting real recording conditions)

**Loading:**
```python
data = np.load('data/real_units/waveforms_real.npz')
X = data['waveforms']       # (96006, 64) float32
y = data['class_labels']    # (96006,)    int32
```

**Provenance sidecar:** `data/real_units/waveforms_real_meta.csv`  
Columns: `source_file`, `cluster_id`, `label`, `unit_id`, `timestamp_s` — maps every row in `waveforms_real.npz` back to the exact .mat file and cluster it came from.

---

## 6. Tier 3 — NWB Files (DANDI dataset 001641)

### Location: `ephys_data/001641/sub-{31,32,34,35,42,44}/`

Neurospiking Benchmark dataset (Taniguchi / Ito-Doya, Keio University):
- **70 NWB files**, 6 subjects (rat NAc/VP decision-making task)
- **214 unit-sessions**, **17.6M total spikes**
- Brain areas: NAc (nucleus accumbens) + VP (ventral pallidum)
- **No raw voltage** — spike times only (these files were submitted at the spike-train level)

**NWB hierarchy:**

```
NWBFile
├── session_description, identifier, session_start_time
├── subject/                            # Subject metadata
│   └── subject_id, species, sex, age
├── electrodes/                         # Electrode table (group, location, etc.)
│   └── DynamicTable (n_electrodes rows)
│
├── units/                              # Primary data: sorted units
│   └── DynamicTable
│       ├── spike_times          → VectorData (variable-length, accessed via spike_times_index)
│       ├── spike_times_index    → VectorIndex
│       ├── electrodes           → references into electrodes table
│       ├── unit_name            → str
│       └── [quality, waveform_mean, etc.] → optional fields
│
└── intervals/                          # Trial structure (task epochs)
    └── trials/
        ├── start_time, stop_time
        └── [condition columns] → e.g. reward, choice, outcome
```

**Accessing spike trains:**
```python
from pynwb import NWBHDF5IO

with NWBHDF5IO('ephys_data/001641/sub-31/...nwb', 'r') as io:
    nwb  = io.read()
    units = nwb.units.to_dataframe()           # one row per unit
    spk   = nwb.units['spike_times'][0]        # spike times for unit 0 (seconds)
    n_units = len(nwb.units)
```

**SpikeInterface workaround** (these files have no raw voltage → SI fails by default):
```python
import spikeinterface.extractors as se

rec = se.NwbRecordingExtractor(
    file_path = path,
    electrical_series_name = None,   # suppress ElectricalSeries detection
    sampling_frequency = 30_000.0,
    t_start = 0.0,
)
sort = se.NwbSortingExtractor(file_path=path)
```

---

## 7. How These Files Support Validation Tests

### 7.1 Waveform Shape Tests (Tier 1 / 2 → `waveforms_real.npz`)

| Test | File(s) | What to validate |
|------|---------|-----------------|
| **SU waveform morphology** | final/ + waveforms_real.npz | Mean waveform has single trough, rises back toward baseline; FWHM 0.3–1.5 ms |
| **Cluster separability** | final/ | PCA of SU vs excluded clusters shows clear separation for accepted SUs |
| **Noise amplitude distribution** | 5.1/ or 5.2/ (noise sentinel events) | Should span a broad amplitude range; not unimodal like real spikes |
| **Trough depth by channel** | final/ | ss8 troughs ≈ −62–83 µV; ss11 ≈ −33 µV; ss12 ≈ −20–24 µV; ss13 ≈ −22–47 µV |

```python
# Example: validate that accepted SU waveforms have a clear trough
d = load_osort_mat('ephys_data/sort/final/A_ss13_Max_sorted_new.mat')
for cid in d['su_ids']:
    mask  = d['cluster_ids'] == cid
    mean_w = d['waveforms_uv'][mask].mean(axis=0)
    trough_sample = mean_w.argmin()
    trough_uv     = mean_w.min()
    assert trough_uv < -10, f"SU {cid} trough too shallow: {trough_uv:.1f} µV"
    assert 50 < trough_sample < 220, f"SU {cid} trough misaligned: sample {trough_sample}"
```

### 7.2 Timestamp Continuity Tests (Tier 1 / 2)

```python
d = load_osort_mat('ephys_data/sort/5.1/A_ss1_sorted_new.mat')
isi = np.diff(np.sort(d['timestamps_rel']))
assert isi.min() > 0,        "Duplicate timestamps"
assert isi.max() < 60.0,     "Gap > 60 s — missing data block?"
assert d['timestamps_rel'][-1] > 1800, "Session < 30 min — truncated file?"
```

### 7.3 Hard-Negative Quality Tests (final/ `useNegativeExcluded`)

The excluded clusters in final/ are **genuine OSort outputs that a human rejected** — they closely resemble real spikes. Use them to verify the discriminator does not overfit to trivially-bad noise.

```python
d = load_osort_mat('ephys_data/sort/final/A_ss11_Min_sorted_new.mat')
su_mask   = np.isin(d['cluster_ids'], d['su_ids'])
excl_mask = np.isin(d['cluster_ids'], d['excluded_ids'])

su_mean   = d['waveforms_uv'][su_mask].mean(axis=0).min()
excl_mean = d['waveforms_uv'][excl_mask].mean(axis=0).min()
# SU and excluded troughs may be similar in depth — their separation is in shape, not amplitude
print(f"SU trough: {su_mean:.1f} µV,  excluded trough: {excl_mean:.1f} µV")
```

### 7.4 Class Balance Tests (waveforms_real.npz)

```python
data    = np.load('data/real_units/waveforms_real.npz')
balance = data['class_labels'].mean()
assert 0.25 < balance < 0.75, f"Dataset severely imbalanced: {balance:.1%} spikes"
n_units = len(np.unique(data['unit_ids'][data['unit_ids'] > 0]))
assert n_units == 10, f"Expected 10 SUs, got {n_units}"
```

### 7.5 NWB ↔ Pipeline Integration Tests (Tier 3)

```python
# Verify spike times lie within expected session bounds
with NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    for i in range(len(nwb.units)):
        spk = nwb.units['spike_times'][i]
        assert spk.min() >= 0, "Negative spike time"
        assert len(spk) >= 10, f"Unit {i} has fewer than 10 spikes"
        isi = np.diff(np.sort(spk))
        refrac_violations = (isi < 0.001).mean()
        assert refrac_violations < 0.01, f"Unit {i}: {refrac_violations:.1%} refractory violations"
```

---

## 8. Quick Reference — Data Loading Cheatsheet

```python
import numpy as np, pandas as pd
from scipy.io import loadmat
from pathlib import Path

SORT_DIR = Path('ephys_data/sort')
SCALING  = 9.155273e-8   # V/ADU
NOISE_ID = 99999999

# ── XLSX (Tier 0) ──────────────────────────────────────────────────────────── #
raw = pd.read_excel(SORT_DIR / 'A00_5-16_2_cells_final copy.xlsx',
                    sheet_name='1', header=9)
# quality_code is in raw['Notes'] column

# ── NPZ curation log (Tier 0 machine-readable) ─────────────────────────────── #
log = np.load(SORT_DIR / 'curation_log.npz', allow_pickle=True)
active_ch = log['channels'][log['has_neurons']]  # [8, 11, 12, 13]

# ── Single .mat file (Tier 1 or Tier 2) ────────────────────────────────────── #
mat = loadmat(str(SORT_DIR / '5.1' / 'A_ss8_sorted_new.mat'), simplify_cells=True)
waves_uv   = mat['newSpikesNegative'] * SCALING * 1e6  # (N, 256) µV
timestamps = mat['newTimestampsNegative'].ravel()
cluster_ids = mat['assignedNegative'].ravel().astype(np.int32)
su_ids     = mat['useNegative'].ravel().tolist()

# ── Pipeline dataset (waveforms_real.npz) ──────────────────────────────────── #
ds = np.load('data/real_units/waveforms_real.npz')
X, y, unit_ids = ds['waveforms'], ds['class_labels'], ds['unit_ids']
# X: (96006, 64) float32 µV  |  y: 0=noise, 1=spike  |  unit_ids: 0..10
```

---

*Document generated from `osort_file_extraction.ipynb` — all numbers verified against real data.*  
*Session date: 2025-05-16 | Recording duration: ~1950 s (~32 min) | Fs = 30 kHz*
