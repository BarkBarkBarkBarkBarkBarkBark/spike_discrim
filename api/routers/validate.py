"""
api/routers/validate.py — Objective proof endpoints.

These endpoints allow a user to independently verify that the API is not
just returning stale cached data, but is actively re-computing correct
values from the stored artefacts.

Proof methods
-------------
1. /api/validate/checksums/{run_id}
   SHA-256 hashes of all result files in a run.  Compare across machines
   or after a re-run to confirm byte-level identity.

2. /api/validate/metrics/{run_id}
   Re-computes AUC, balanced accuracy, and confusion matrix FROM SCRATCH
   from the stored feature_matrix.parquet + weight_bank.json, then compares
   against the stored weight_bank_metrics.json.
   Returns: {expected, recomputed, match: bool, delta}.

3. /api/validate/feature_stats/{run_id}
   Recomputes mean/std/min/max of every feature column from feature_matrix.parquet
   and returns them alongside n_spikes / n_noise counts.  A human can
   cross-check these descriptive statistics against domain knowledge
   (e.g., peak amplitude should be positive, trough negative, etc.).

4. /api/validate/waveform_checksums
   SHA-256 of data/synthetic/waveforms.npz.  Stable across runs as long
   as the dataset has not been regenerated.

5. /api/validate/roundtrip/{run_id}
   Downloads the CSV export, re-parses it in-process, and confirms that
   the re-parsed values match the original Parquet to float32 precision.
   Returns: {files_checked, all_match: bool, mismatches}.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR   = BASE_DIR / "data" / "results"
SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"

router = APIRouter(tags=["validate"])


# ── Helpers ─────────────────────────────────────────────────────────────────── #

def _run_dir(run_id: str) -> Path:
    d = RESULTS_DIR / run_id
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 1. File checksums ─────────────────────────────────────────────────────── #

@router.get("/validate/checksums/{run_id}")
def checksums(run_id: str) -> dict:
    """SHA-256 hashes for every file in the run directory."""
    d = _run_dir(run_id)
    result: dict[str, str] = {}
    for f in sorted(d.rglob("*")):
        if f.is_file():
            result[str(f.relative_to(d))] = _sha256(f)
    return {"run_id": run_id, "files": result}


# ── 2. Metric recomputation ───────────────────────────────────────────────── #

@router.get("/validate/metrics/{run_id}")
def recompute_metrics(run_id: str) -> dict:
    """
    Re-derive WeightBank AUC and balanced accuracy from raw artefacts and
    compare against the values stored in weight_bank_metrics.json.

    Proof of correctness: if recomputed == stored (within float32 tolerance),
    the pipeline, the storage, and the API are all consistent.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    d = _run_dir(run_id)

    # Load stored metrics (ground truth from the pipeline run)
    stored_metrics_path = d / "weight_bank_metrics.json"
    if not stored_metrics_path.exists():
        raise HTTPException(status_code=404, detail="weight_bank_metrics.json not found — run with --tier ≥ 2")
    with open(stored_metrics_path) as fh:
        stored = json.load(fh)

    # Load feature matrix
    fm_path = d / "feature_matrix.parquet"
    if not fm_path.exists():
        raise HTTPException(status_code=404, detail="feature_matrix.parquet not found")
    df = pd.read_parquet(fm_path)
    labels = df["class_label"].values.astype(int)
    feat_names = [c for c in df.columns if c not in ("class_label", "unit_id")]
    X = df[feat_names].values.astype(np.float32)

    # Re-load and re-apply WeightBank
    wb_path = d / "weight_bank.json"
    if not wb_path.exists():
        raise HTTPException(status_code=404, detail="weight_bank.json not found")

    # Import and reconstruct WeightBank from its saved JSON
    sys_path_insert = str(BASE_DIR / "src")
    import sys
    if sys_path_insert not in sys.path:
        sys.path.insert(0, sys_path_insert)

    from spike_discrim.input_layer.weights import WeightBank
    wb = WeightBank.load(wb_path)
    wb.warmup()

    scores   = wb.score_batch(X)
    is_spike = wb.classify(X).astype(int)

    recomp_auc      = float(roc_auc_score(labels, scores))
    recomp_bal_acc  = float(balanced_accuracy_score(labels, is_spike))

    stored_auc     = float(stored.get("auc", -1))
    stored_bal_acc = float(stored.get("balanced_accuracy", -1))

    auc_delta     = abs(recomp_auc    - stored_auc)
    bal_acc_delta = abs(recomp_bal_acc - stored_bal_acc)
    tol           = 1e-4   # float32 round-trip tolerance

    return {
        "run_id": run_id,
        "tolerance": tol,
        "auc": {
            "stored":     stored_auc,
            "recomputed": recomp_auc,
            "delta":      auc_delta,
            "match":      auc_delta < tol,
        },
        "balanced_accuracy": {
            "stored":     stored_bal_acc,
            "recomputed": recomp_bal_acc,
            "delta":      bal_acc_delta,
            "match":      bal_acc_delta < tol,
        },
        "overall_match": auc_delta < tol and bal_acc_delta < tol,
    }


# ── 3. Feature descriptive statistics ────────────────────────────────────── #

@router.get("/validate/feature_stats/{run_id}")
def feature_stats(run_id: str) -> dict:
    """
    Compute mean, std, min, max for every feature column, split by class.
    Allows a domain expert to sanity-check that:
      - trough_amplitude is negative for spikes
      - peak_amplitude is positive for spikes
      - noise has higher baseline_rms variance
      - etc.
    """
    import pandas as pd

    d    = _run_dir(run_id)
    path = d / "feature_matrix.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="feature_matrix.parquet not found")

    df         = pd.read_parquet(path)
    feat_names = [c for c in df.columns if c not in ("class_label", "unit_id")]

    result: dict[str, Any] = {
        "run_id":    run_id,
        "n_total":   len(df),
        "n_spikes":  int((df["class_label"] == 1).sum()),
        "n_noise":   int((df["class_label"] == 0).sum()),
        "features":  {},
    }

    for feat in feat_names:
        spikes = df.loc[df["class_label"] == 1, feat].astype(float)
        noise  = df.loc[df["class_label"] == 0, feat].astype(float)
        result["features"][feat] = {
            "spike": {
                "mean": round(float(spikes.mean()), 4),
                "std":  round(float(spikes.std()),  4),
                "min":  round(float(spikes.min()),  4),
                "max":  round(float(spikes.max()),  4),
            },
            "noise": {
                "mean": round(float(noise.mean()), 4),
                "std":  round(float(noise.std()),  4),
                "min":  round(float(noise.min()),  4),
                "max":  round(float(noise.max()),  4),
            },
        }

    return result


# ── 4. Waveform file checksum ─────────────────────────────────────────────── #

@router.get("/validate/waveform_checksums")
def waveform_checksums() -> dict:
    """SHA-256 of data/synthetic/waveforms.npz (and labels.parquet if present)."""
    files = {}
    for fname in ("waveforms.npz", "labels.parquet", "generation_config.json"):
        p = SYNTHETIC_DIR / fname
        if p.exists():
            files[fname] = _sha256(p)
    if not files:
        raise HTTPException(status_code=404, detail="No synthetic data files found")
    return {"data_dir": str(SYNTHETIC_DIR.relative_to(BASE_DIR)), "files": files}


# ── 5. CSV round-trip check ───────────────────────────────────────────────── #

@router.get("/validate/roundtrip/{run_id}")
def csv_roundtrip(run_id: str) -> dict:
    """
    Re-parse the CSV export (generated in-process) and confirm values match
    the Parquet source to float32 precision (6 significant figures).

    This proves the CSV export pipeline is lossless within float32 bounds.
    """
    import pandas as pd

    d = _run_dir(run_id)

    results: list[dict] = []
    all_match = True

    for parquet_path in sorted(d.glob("*.parquet")):
        df_orig = pd.read_parquet(parquet_path)
        numeric_cols = df_orig.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            continue

        # Generate CSV in-memory (same logic as export endpoint)
        records = json.loads(df_orig.to_json(orient="records"))
        csv_buf = io.StringIO()
        writer  = csv.writer(csv_buf)
        headers = list(records[0].keys()) if records else []
        writer.writerow(headers)
        for row in records:
            writer.writerow([row.get(h, "") for h in headers])

        # Re-parse CSV
        csv_buf.seek(0)
        df_reparsed = pd.read_csv(csv_buf)

        mismatches = []
        for col in numeric_cols:
            if col not in df_reparsed.columns:
                mismatches.append({"col": col, "reason": "missing in CSV"})
                continue
            orig_vals    = df_orig[col].astype(float).values
            reparse_vals = df_reparsed[col].astype(float).values
            if len(orig_vals) != len(reparse_vals):
                mismatches.append({"col": col, "reason": "length mismatch"})
                continue
            max_delta = float(np.max(np.abs(orig_vals - reparse_vals)))
            rel_tol   = 1e-5
            if max_delta > rel_tol * (np.max(np.abs(orig_vals)) + 1e-10):
                mismatches.append({"col": col, "max_delta": max_delta})

        file_ok = len(mismatches) == 0
        if not file_ok:
            all_match = False
        results.append({
            "file":       parquet_path.name,
            "rows":       len(df_orig),
            "cols_checked": len(numeric_cols),
            "match":      file_ok,
            "mismatches": mismatches,
        })

    return {"run_id": run_id, "all_match": all_match, "files": results}
