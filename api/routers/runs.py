"""
api/routers/runs.py — Read-only endpoints that surface saved benchmark results.

All data is read from data/results/<run_id>/*.json and *.parquet.
No computation is performed here — the API is a thin filesystem reader.

Endpoints
---------
GET /api/runs                            → list of run IDs (sorted newest-first)
GET /api/runs/{run_id}/summary           → run_summary.json
GET /api/runs/{run_id}/profiling         → profiling_summary.json
GET /api/runs/{run_id}/features/single   → single_feature_ranks.json (CSV-convertible)
GET /api/runs/{run_id}/features/sets     → feature_set_ranks.json (CSV-convertible)
GET /api/runs/{run_id}/weightbank        → weight_bank_metrics.json
GET /api/runs/{run_id}/waveforms         → first N waveforms as lists (visualisation)
GET /api/runs/{run_id}/export/csv        → ZIP of all .json results as .csv files
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

BASE_DIR     = Path(__file__).resolve().parent.parent.parent  # project root
RESULTS_DIR  = BASE_DIR / "data" / "results"
SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"

router = APIRouter(tags=["runs"])


# ── Helpers ─────────────────────────────────────────────────────────────────── #

def _run_dir(run_id: str) -> Path:
    d = RESULTS_DIR / run_id
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return d


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    with open(path) as fh:
        return json.load(fh)


def _parquet_to_records(path: Path) -> list[dict]:
    """Read a Parquet file and return as a list of row dicts (JSON-serialisable)."""
    import pandas as pd
    df = pd.read_parquet(path)
    # Convert numpy scalars to plain Python types
    return json.loads(df.to_json(orient="records"))


# ── Endpoints ──────────────────────────────────────────────────────────────── #

@router.get("/runs")
def list_runs() -> list[str]:
    """Return all run IDs sorted newest-first."""
    if not RESULTS_DIR.exists():
        return []
    runs = sorted(
        [d.name for d in RESULTS_DIR.iterdir()
         if d.is_dir() and not d.name.startswith(".")],
        reverse=True,
    )
    return runs


@router.get("/runs/{run_id}/summary")
def get_summary(run_id: str) -> Any:
    d = _run_dir(run_id)
    return _read_json(d / "run_summary.json")


@router.get("/runs/{run_id}/profiling")
def get_profiling(run_id: str) -> Any:
    d = _run_dir(run_id)
    return _read_json(d / "profiling_summary.json")


@router.get("/runs/{run_id}/features/single")
def get_single_feature_ranks(run_id: str) -> list[dict]:
    d = _run_dir(run_id)
    parquet = d / "single_feature_ranks.parquet"
    if parquet.exists():
        return _parquet_to_records(parquet)
    # Fallback to JSON sidecar
    return _read_json(d / "single_feature_ranks.json")


@router.get("/runs/{run_id}/features/sets")
def get_feature_set_ranks(run_id: str) -> list[dict]:
    d = _run_dir(run_id)
    parquet = d / "feature_set_ranks.parquet"
    if parquet.exists():
        return _parquet_to_records(parquet)
    return _read_json(d / "feature_set_ranks.json")


@router.get("/runs/{run_id}/weightbank")
def get_weightbank(run_id: str) -> Any:
    d = _run_dir(run_id)
    metrics = _read_json(d / "weight_bank_metrics.json")
    # Merge top-level weight_bank.json (centers/widths omitted — too large)
    wb_path = d / "weight_bank.json"
    if wb_path.exists():
        with open(wb_path) as fh:
            wb = json.load(fh)
        metrics["n_bins"]       = wb.get("n_bins")
        metrics["sigma_scale"]  = wb.get("sigma_scale")
        metrics["feature_names"] = wb.get("feature_names", [])
        metrics["n_features"]   = len(wb.get("feature_names", []))
    return metrics


@router.get("/runs/{run_id}/waveforms")
def get_waveforms(run_id: str, n: int = 200) -> dict:
    """
    Return up to `n` waveforms (default 200) sampled from data/synthetic/waveforms.npz.
    Intended for visualisation only — data is lightly subsampled.
    """
    npz_path = SYNTHETIC_DIR / "waveforms.npz"
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail="waveforms.npz not found in data/synthetic")

    data   = np.load(str(npz_path), allow_pickle=False)
    waves  = data["waveforms"].astype(float)   # (N, T)
    labels = data.get("class_labels", np.ones(len(waves), dtype=int))

    # Balanced sample: up to n//2 spikes + n//2 noise
    half   = max(1, n // 2)
    spike_idx = np.where(labels == 1)[0]
    noise_idx = np.where(labels == 0)[0]
    rng    = np.random.default_rng(42)
    s_sel  = rng.choice(spike_idx, size=min(half, len(spike_idx)), replace=False)
    n_sel  = rng.choice(noise_idx, size=min(half, len(noise_idx)), replace=False)
    idx    = np.concatenate([s_sel, n_sel])
    rng.shuffle(idx)

    return {
        "waveforms": waves[idx].tolist(),
        "labels":    labels[idx].astype(int).tolist(),
        "n_samples": waves.shape[1],
        "total_returned": len(idx),
    }


@router.get("/runs/{run_id}/export/csv")
def export_csv(run_id: str) -> StreamingResponse:
    """
    Download a ZIP archive containing CSV versions of all JSON result files
    for the given run.  Allows objective offline inspection without any
    special tooling.
    """
    d = _run_dir(run_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for json_file in sorted(d.rglob("*.json")):
            rel   = json_file.relative_to(d)
            csv_name = str(rel).replace(".json", ".csv").replace("/", "__")
            try:
                with open(json_file) as fh:
                    raw = json.load(fh)
            except Exception:
                continue

            csv_buf = io.StringIO()
            writer  = csv.writer(csv_buf)

            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                # List of row dicts → proper CSV table
                headers = list(raw[0].keys())
                writer.writerow(headers)
                for row in raw:
                    writer.writerow([row.get(h, "") for h in headers])
            elif isinstance(raw, dict):
                # Flat dict → two-column key/value CSV
                writer.writerow(["key", "value"])
                for k, v in raw.items():
                    writer.writerow([k, json.dumps(v) if isinstance(v, (dict, list)) else v])
            else:
                writer.writerow(["value"])
                writer.writerow([str(raw)])

            zf.writestr(csv_name, csv_buf.getvalue())

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="spike_discrim_{run_id}.zip"'},
    )
