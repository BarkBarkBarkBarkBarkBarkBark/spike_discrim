"""
osort_loader.py — Load osort MATLAB (.mat) files into the canonical Python schema.

osort (Rutishauser et al. 2006) outputs .mat files containing sorted single-unit
spike data.  This adapter supports both:
  - MATLAB v5/v7 format  (scipy.io.loadmat)
  - MATLAB v7.3 / HDF5   (h5py)

The adapter extracts:
  - Per-unit spike times (samples)
  - Per-unit waveform snippets (float32[N_spikes, N_samples])
  - Noise / unassigned threshold crossings
  - Channel and session metadata

Output format (canonical schema from docs/spike_feature_validation.yaml §dataset_schema)
---------
  {
    "units": {
      "unit_001": {
        "spike_times_samples": array[N],
        "waveforms":           float32[N, T],
        "mean_waveform":       float32[T],
        "channel_id":          str,
        "n_spikes":            int,
        "metadata":            dict,
      },
      ...
    },
    "noise": {
      "noise_001": {
        "event_times_samples": array[N],
        "waveforms":           float32[N, T],
        "channel_id":          str,
        "metadata":            dict,
      }
    },
    "session_metadata": { ... },
  }

Saved outputs (to output_dir):
  canonical_units.npz     — unit waveforms + spike times
  canonical_noise.npz     — noise waveforms + event times
  labels.parquet          — per-snippet metadata DataFrame
  session_metadata.json   — session-level info
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# File format detection                                                       #
# --------------------------------------------------------------------------- #

def _is_hdf5(path: Path) -> bool:
    """Check if .mat file is HDF5 (v7.3) by inspecting magic bytes."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
        return header[:4] == b"\x89HDF"
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Low-level loaders                                                           #
# --------------------------------------------------------------------------- #

def _load_scipy(path: Path) -> dict:
    """Load MATLAB v5/v7 .mat file with scipy.io."""
    import scipy.io
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return mat


def _load_h5py(path: Path) -> "h5py.File":
    """Open MATLAB v7.3 .mat file as HDF5."""
    import h5py
    return h5py.File(str(path), "r")


# --------------------------------------------------------------------------- #
# osort field-name heuristics                                                 #
# osort .mat files use various naming conventions; we probe common names.     #
# --------------------------------------------------------------------------- #

_SPIKE_TIME_CANDIDATES = [
    "newSpikeTimes", "spikeTimes", "times", "timestamps",
    "spike_times", "ts",
]
_WAVEFORM_CANDIDATES = [
    "waveforms", "waves", "snippets", "spikeWaveforms",
    "Waveforms", "Waves", "snips",
]
_UNIT_ID_CANDIDATES = [
    "unitIDs", "unit_ids", "unitIds", "clusterIDs",
    "cluster_ids", "labels", "assigns",
]
_NOISE_CANDIDATES = [
    "noiseWaveforms", "noiseSnippets", "noise_waveforms",
    "rejected", "noise",
]


def _find_key(d: dict, candidates: list[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    # Case-insensitive fallback
    dl = {k.lower(): k for k in d}
    for c in candidates:
        if c.lower() in dl:
            return dl[c.lower()]
    return None


def _to_float32_array(v: Any) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr.astype(np.float32)


def _to_int64_array(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=np.int64).ravel()


# --------------------------------------------------------------------------- #
# Main adapter                                                                #
# --------------------------------------------------------------------------- #

def load_osort_mat(
    mat_path:   str | Path,
    output_dir: Optional[str | Path] = None,
    channel_id: str = "unknown",
    session_id: str = "unknown",
    verbose:    bool = True,
) -> dict:
    """Load an osort .mat file into the canonical spike_discrim schema.

    Parameters
    ----------
    mat_path   : Path to the .mat file.
    output_dir : If provided, saves NPZ + Parquet + JSON to this directory.
    channel_id : Channel identifier (e.g. "CSC1" or "ch01").
    session_id : Session identifier (e.g. date or experiment ID).
    verbose    : Print field-discovery log.

    Returns
    -------
    dict following the canonical schema described in module docstring.
    """
    import pandas as pd

    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"osort .mat file not found: {mat_path}")

    # ── Load raw mat ──────────────────────────────────────────────────────── #
    is_hdf5 = _is_hdf5(mat_path)
    if verbose:
        fmt = "HDF5 (v7.3)" if is_hdf5 else "scipy (v5/v7)"
        print(f"Loading {mat_path.name} [{fmt}]")

    if is_hdf5:
        mat_raw = _load_h5py(mat_path)
    else:
        mat_raw = _load_scipy(mat_path)

    if verbose:
        print(f"  Keys: {list(mat_raw.keys())[:20]}")

    # ── Discover field names ──────────────────────────────────────────────── #
    keys = dict(mat_raw)

    spike_time_key = _find_key(keys, _SPIKE_TIME_CANDIDATES)
    waveform_key   = _find_key(keys, _WAVEFORM_CANDIDATES)
    unit_id_key    = _find_key(keys, _UNIT_ID_CANDIDATES)
    noise_key      = _find_key(keys, _NOISE_CANDIDATES)

    if verbose:
        print(f"  spike_times → {spike_time_key}")
        print(f"  waveforms   → {waveform_key}")
        print(f"  unit_ids    → {unit_id_key}")
        print(f"  noise       → {noise_key}")

    if spike_time_key is None and waveform_key is None:
        warnings.warn(
            f"Could not identify spike-time or waveform fields in {mat_path.name}. "
            f"Available keys: {list(mat_raw.keys())[:30]}\n"
            "Set the correct field names manually and call _parse_fields().",
            stacklevel=2,
        )

    # ── Extract arrays ────────────────────────────────────────────────────── #
    spike_times_raw = mat_raw[spike_time_key] if spike_time_key else None
    waveforms_raw   = mat_raw[waveform_key]   if waveform_key   else None
    unit_ids_raw    = mat_raw[unit_id_key]    if unit_id_key    else None
    noise_raw       = mat_raw[noise_key]      if noise_key      else None

    # Convert
    spike_times = _to_int64_array(spike_times_raw) if spike_times_raw is not None else np.array([], dtype=np.int64)
    waveforms   = _to_float32_array(waveforms_raw) if waveforms_raw   is not None else np.empty((0, 64), dtype=np.float32)
    unit_ids    = _to_int64_array(unit_ids_raw)    if unit_ids_raw    is not None else np.zeros(len(spike_times), dtype=np.int64)

    if verbose:
        print(f"  spike_times : {spike_times.shape}  [{spike_times.dtype}]")
        print(f"  waveforms   : {waveforms.shape}  [{waveforms.dtype}]")
        print(f"  unit_ids    : {unit_ids.shape}")

    # Ensure waveforms and spike_times are aligned
    n_spikes = min(len(spike_times), len(waveforms))
    spike_times = spike_times[:n_spikes]
    waveforms   = waveforms[:n_spikes]
    unit_ids    = unit_ids[:n_spikes]

    # ── Build per-unit dict ───────────────────────────────────────────────── #
    unique_units = np.unique(unit_ids)
    # Filter out noise units (often labelled 0, -1, or very large IDs)
    valid_units  = unique_units[(unique_units > 0) & (unique_units < 1000)]

    units_out: dict[str, dict] = {}
    for uid in valid_units:
        mask = unit_ids == uid
        u_times  = spike_times[mask]
        u_waves  = waveforms[mask]
        key      = f"unit_{uid:03d}"
        units_out[key] = {
            "spike_times_samples": u_times,
            "waveforms":           u_waves,
            "mean_waveform":       u_waves.mean(axis=0).astype(np.float32),
            "channel_id":          channel_id,
            "n_spikes":            int(mask.sum()),
            "metadata": {
                "unit_id":    int(uid),
                "session_id": session_id,
                "source":     str(mat_path),
            },
        }

    if verbose:
        print(f"  Units found: {list(units_out.keys())}")

    # ── Extract noise events ──────────────────────────────────────────────── #
    noise_out: dict[str, dict] = {}
    if noise_raw is not None:
        n_waves = _to_float32_array(noise_raw)
        noise_out["noise_001"] = {
            "event_times_samples": np.array([], dtype=np.int64),
            "waveforms":           n_waves,
            "channel_id":          channel_id,
            "metadata": {"source": "osort_rejected", "session_id": session_id},
        }
    else:
        # Use unit_ids == 0 as noise if available
        noise_mask = unit_ids == 0
        if noise_mask.any():
            noise_out["noise_001"] = {
                "event_times_samples": spike_times[noise_mask],
                "waveforms":           waveforms[noise_mask],
                "channel_id":          channel_id,
                "metadata": {"source": "unassigned_threshold_crossings",
                             "session_id": session_id},
            }

    # ── Build per-snippet label DataFrame ────────────────────────────────── #
    meta_rows: list[dict] = []
    for uid, udata in units_out.items():
        for i in range(udata["n_spikes"]):
            meta_rows.append({
                "unit_label":   uid,
                "unit_id":      udata["metadata"]["unit_id"],
                "class_label":  1,
                "spike_time_samples": int(udata["spike_times_samples"][i]),
                "channel_id":   channel_id,
                "session_id":   session_id,
                "source":       "osort_sorted",
            })
    for nid, ndata in noise_out.items():
        for i in range(len(ndata["waveforms"])):
            meta_rows.append({
                "unit_label":   "noise",
                "unit_id":      0,
                "class_label":  0,
                "spike_time_samples": int(ndata["event_times_samples"][i]) if i < len(ndata["event_times_samples"]) else -1,
                "channel_id":   channel_id,
                "session_id":   session_id,
                "source":       ndata["metadata"].get("source", "noise"),
            })

    labels_df = pd.DataFrame(meta_rows)

    result = {
        "units":            units_out,
        "noise":            noise_out,
        "session_metadata": {
            "channel_id":  channel_id,
            "session_id":  session_id,
            "mat_path":    str(mat_path),
            "n_units":     len(units_out),
            "n_noise":     sum(len(v["waveforms"]) for v in noise_out.values()),
        },
        "labels_df": labels_df,
    }

    # ── Save if output_dir provided ───────────────────────────────────────── #
    if output_dir is not None:
        _save_canonical(result, output_dir, verbose=verbose)

    return result


def _save_canonical(result: dict, output_dir: str | Path, verbose: bool = True) -> None:
    """Save canonical dataset to output_dir as NPZ + Parquet + JSON."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unit waveforms and labels
    unit_waveforms_list: list[np.ndarray] = []
    unit_labels_list:    list[int]        = []
    unit_ids_list:       list[int]        = []

    for uid, udata in result["units"].items():
        unit_waveforms_list.append(udata["waveforms"])
        unit_labels_list.extend([1] * udata["n_spikes"])
        unit_ids_list.extend([udata["metadata"]["unit_id"]] * udata["n_spikes"])

    # Noise
    noise_waveforms_list: list[np.ndarray] = []
    for _, ndata in result["noise"].items():
        noise_waveforms_list.append(ndata["waveforms"])

    if unit_waveforms_list:
        spike_waves = np.vstack(unit_waveforms_list)
        spike_labels = np.array(unit_labels_list, dtype=np.int32)
        spike_uids   = np.array(unit_ids_list,    dtype=np.int32)
        np.savez_compressed(
            output_dir / "canonical_units.npz",
            waveforms    = spike_waves,
            class_labels = spike_labels,
            unit_ids     = spike_uids,
        )

    if noise_waveforms_list:
        noise_waves = np.vstack(noise_waveforms_list)
        np.savez_compressed(
            output_dir / "canonical_noise.npz",
            waveforms    = noise_waves,
            class_labels = np.zeros(len(noise_waves), dtype=np.int32),
        )

    result["labels_df"].to_parquet(output_dir / "labels.parquet", index=False)

    with open(output_dir / "session_metadata.json", "w") as fh:
        json.dump(result["session_metadata"], fh, indent=2)

    if verbose:
        print(f"  Canonical dataset saved to: {output_dir}")
