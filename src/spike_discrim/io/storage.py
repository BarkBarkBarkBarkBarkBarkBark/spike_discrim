"""
storage.py — NPZ + Parquet I/O with full provenance tracking.

Every save operation writes:
  - The primary data (NPZ or Parquet)
  - A sidecar provenance JSON with timestamp, shape, dtype, and caller info

This ensures that every file in data/ can be traced back to the run that
created it without any external database.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Waveform NPZ I/O                                                            #
# --------------------------------------------------------------------------- #

def save_waveforms(
    path:        str | Path,
    waveforms:   np.ndarray,              # float32[N, T]
    class_labels: Optional[np.ndarray] = None,  # int32[N]
    unit_ids:    Optional[np.ndarray]  = None,  # int32[N]
    metadata:    Optional[dict]         = None,
) -> Path:
    """Save waveforms to a compressed NPZ file with sidecar JSON.

    Returns the saved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {"waveforms": np.asarray(waveforms, dtype=np.float32)}
    if class_labels is not None:
        arrays["class_labels"] = np.asarray(class_labels, dtype=np.int32)
    if unit_ids is not None:
        arrays["unit_ids"] = np.asarray(unit_ids, dtype=np.int32)

    np.savez_compressed(path, **arrays)

    # Sidecar provenance
    prov = {
        "saved_at":     datetime.now(timezone.utc).isoformat(),
        "path":         str(path),
        "waveforms_shape": list(waveforms.shape),
        "dtype":        str(waveforms.dtype),
        "has_labels":   class_labels is not None,
        "has_unit_ids": unit_ids is not None,
        "n_spikes":     int((class_labels == 1).sum()) if class_labels is not None else None,
        "n_noise":      int((class_labels == 0).sum()) if class_labels is not None else None,
        "metadata":     metadata or {},
    }
    prov_path = path.with_suffix(".json")
    with open(prov_path, "w") as fh:
        json.dump(prov, fh, indent=2)

    return path


def load_waveforms(path: str | Path) -> dict[str, np.ndarray]:
    """Load waveforms NPZ.  Returns dict with all saved arrays."""
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# Feature table Parquet I/O                                                  #
# --------------------------------------------------------------------------- #

def save_features_parquet(
    path:          str | Path,
    feature_df:    Any,   # pandas DataFrame
    metadata:      Optional[dict] = None,
) -> Path:
    """Save feature DataFrame to Parquet with sidecar JSON."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(feature_df)
    df.to_parquet(path, index=False)

    prov = {
        "saved_at":   datetime.now(timezone.utc).isoformat(),
        "path":       str(path),
        "n_rows":     len(df),
        "columns":    list(df.columns),
        "metadata":   metadata or {},
    }
    prov_path = path.with_suffix(".json")
    with open(prov_path, "w") as fh:
        json.dump(prov, fh, indent=2)

    return path


def load_features_parquet(path: str | Path) -> Any:
    """Load feature table from Parquet.  Returns pandas DataFrame."""
    import pandas as pd
    return pd.read_parquet(str(path))


# --------------------------------------------------------------------------- #
# Generic JSON save                                                           #
# --------------------------------------------------------------------------- #

def save_results_json(path: str | Path, data: dict) -> Path:
    """Save any dict as a human-readable JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# --------------------------------------------------------------------------- #
# Run directory creation                                                      #
# --------------------------------------------------------------------------- #

def make_run_dir(base_dir: str | Path = "data/results") -> Path:
    """Create a timestamped run directory and return its Path."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = Path(base_dir) / ts
    run.mkdir(parents=True, exist_ok=True)
    return run
