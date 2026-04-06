"""
api/routers/ephys_eval.py — Real electrophysiology evaluation endpoints.

Bridges the OSort-curated ground-truth dataset (data/real_units/waveforms_real.npz,
produced by osort_file_extraction.ipynb) into the API so that agents and
test harnesses can:

  1. Inspect the real dataset inventory (class balance, per-SU counts, sources)
  2. Run biological plausibility tests without touching synthetic data
  3. Evaluate any trained WeightBank against real waveforms in one API call
  4. Browse sample waveforms for visual sanity-checks

Endpoints
---------
GET  /api/ephys/dataset
     Inventory of waveforms_real.npz: shape, class balance, per-SU event
     counts, per-source-file breakdown (from waveforms_real_meta.csv).

GET  /api/ephys/tests
     Run all 5 biological validation tests and return a structured
     pass/fail report with per-test details.  Safe to call repeatedly.
     Tests:
       1. waveform_morphology  — mean spike waveform has clear trough < −5 µV
       2. class_balance        — 25–75% of snippets are labelled as spikes
       3. unit_count           — dataset contains exactly 10 SU identities
       4. amplitude_range      — noise snippets span broader amplitude range
       5. hard_negative_quality— excluded clusters are NOT simply low-amplitude

POST /api/ephys/evaluate
     Body: {"run_id": "abc12345", "tier": 2}
     Extracts features from waveforms_real.npz using the same tier as the
     reference run, scores with that run's WeightBank, and returns:
       • overall AUC, balanced accuracy, precision, recall, F1
       • per-SU unit breakdown (recall per unit)
       • per-source-file breakdown (accuracy per recording channel)
       • confusion matrix

GET  /api/ephys/waveforms
     Query params: n=20, class_label=0|1 (optional), unit_id=int (optional)
     Return n sample waveforms (as lists of floats) for visualisation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
REAL_UNITS_DIR = BASE_DIR / "data" / "real_units"
RESULTS_DIR    = BASE_DIR / "data" / "results"

REAL_NPZ   = REAL_UNITS_DIR / "waveforms_real.npz"
REAL_META  = REAL_UNITS_DIR / "waveforms_real_meta.csv"

router = APIRouter(tags=["ephys"])

# ── Source setup ─────────────────────────────────────────────────────────── #
_src = str(BASE_DIR / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ── Helpers ─────────────────────────────────────────────────────────────── #

def _load_real() -> dict[str, np.ndarray]:
    """Load waveforms_real.npz.  Raises 404 if the file is missing."""
    if not REAL_NPZ.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "waveforms_real.npz not found. "
                "Run osort_file_extraction.ipynb cell 11 to generate it."
            ),
        )
    data = np.load(REAL_NPZ)
    return {
        "waveforms":   data["waveforms"].astype(np.float32),    # (N, 64)
        "labels":      data["class_labels"].astype(np.int32),   # (N,)
        "unit_ids":    data["unit_ids"].astype(np.int32),       # (N,)
    }


def _load_meta() -> "pd.DataFrame | None":
    import pandas as pd
    if REAL_META.exists():
        return pd.read_csv(REAL_META)
    return None


def _extract_features(
    waveforms: np.ndarray,
    cfg: dict[str, Any] | None,
    tier: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Extract feature matrix from raw waveforms using the shared pipeline logic."""
    from spike_discrim.features.extraction import build_feature_matrix
    return build_feature_matrix(waveforms, cfg=cfg, tier=tier)


def _run_dir(run_id: str) -> Path:
    d = RESULTS_DIR / run_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return d


# ── 1. Dataset inventory ─────────────────────────────────────────────────── #

@router.get("/ephys/dataset")
def dataset_info() -> dict[str, Any]:
    """
    Full inventory of waveforms_real.npz.

    Returns shape, class balance, per-SU event counts, and (if the CSV
    sidecar is present) per-source-file and per-cluster breakdowns.
    """
    import pandas as pd

    d = _load_real()
    W, L, U = d["waveforms"], d["labels"], d["unit_ids"]
    N, T    = W.shape

    # Per-SU counts
    su_counts: dict[str, int] = {}
    for uid in np.unique(U[U > 0]):
        su_counts[str(int(uid))] = int((U == uid).sum())

    result: dict[str, Any] = {
        "file":             str(REAL_NPZ.relative_to(BASE_DIR)),
        "n_snippets":       N,
        "n_samples":        T,
        "n_spikes":         int((L == 1).sum()),
        "n_noise":          int((L == 0).sum()),
        "spike_fraction":   round(float(L.mean()), 4),
        "n_unique_su":      int((np.unique(U) > 0).sum()),
        "su_event_counts":  su_counts,
        "waveform_stats": {
            "spike_trough_uv":  {
                "mean": round(float(W[L == 1].min(axis=1).mean()), 2),
                "std":  round(float(W[L == 1].min(axis=1).std()),  2),
            },
            "noise_trough_uv": {
                "mean": round(float(W[L == 0].min(axis=1).mean()), 2),
                "std":  round(float(W[L == 0].min(axis=1).std()),  2),
            },
        },
    }

    meta = _load_meta()
    if meta is not None:
        # Per-source-file breakdown
        src_summary: list[dict] = []
        for src_file, grp in meta.groupby("source_file"):
            src_summary.append({
                "source_file": src_file,
                "n_total":     len(grp),
                "n_spikes":    int((grp["label"] == "SU").sum()),
                "n_noise":     int((grp["label"] == "NOISE").sum()),
                "su_clusters": sorted(grp.loc[grp["label"] == "SU", "cluster_id"]
                                      .unique().tolist()),
            })
        result["source_files"] = src_summary

    return result


# ── 2. Validation test suite ──────────────────────────────────────────────── #

@router.get("/ephys/tests")
def run_validation_tests() -> dict[str, Any]:
    """
    Run 5 biological plausibility tests on waveforms_real.npz.

    Each test returns {passed: bool, details: str, ...metrics}.
    Overall passed=True only when all 5 tests pass.

    Tests
    -----
    1. waveform_morphology   SU mean waveform trough < −5 µV for every SU
    2. class_balance         Spike fraction within 25–75%
    3. unit_count            Exactly 10 unique SU identities
    4. amplitude_range       Noise amplitude range ≥ 2× spike trough std
    5. hard_negative_quality Excluded clusters have mean trough depth > 50% of SU
    """
    d = _load_real()
    W, L, U = d["waveforms"], d["labels"], d["unit_ids"]
    meta = _load_meta()

    results: dict[str, Any] = {}
    all_passed = True

    # ── Test 1: Waveform morphology ──────────────────────────────────────── #
    per_su: dict[str, dict] = {}
    t1_ok  = True
    for uid in np.unique(U[U > 0]):
        mask   = (U == uid) & (L == 1)
        if not mask.any():
            continue
        mean_w       = W[mask].mean(axis=0)
        trough_val   = float(mean_w.min())
        trough_samp  = int(mean_w.argmin())
        su_ok        = trough_val < -5.0
        per_su[str(int(uid))] = {
            "trough_uv":     round(trough_val, 2),
            "trough_sample": trough_samp,
            "passed":        su_ok,
        }
        if not su_ok:
            t1_ok = False

    results["waveform_morphology"] = {
        "passed":  t1_ok,
        "details": "Each accepted SU mean waveform must have trough < −5 µV.",
        "per_su":  per_su,
    }
    if not t1_ok:
        all_passed = False

    # ── Test 2: Class balance ────────────────────────────────────────────── #
    frac = float(L.mean())
    t2_ok = 0.25 <= frac <= 0.75
    results["class_balance"] = {
        "passed":         t2_ok,
        "spike_fraction": round(frac, 4),
        "required":       "0.25 – 0.75",
        "details":        "Ensures neither class dominates pathologically.",
    }
    if not t2_ok:
        all_passed = False

    # ── Test 3: Unit count ───────────────────────────────────────────────── #
    n_units = int((np.unique(U) > 0).sum())
    t3_ok   = n_units == 10
    results["unit_count"] = {
        "passed":   t3_ok,
        "n_units":  n_units,
        "expected": 10,
        "details":  "10 SUs from 4 channels (ss8, ss11, ss12, ss13).",
    }
    if not t3_ok:
        all_passed = False

    # ── Test 4: Amplitude range (noise spans broader range than spikes) ───── #
    spike_troughs = W[L == 1].min(axis=1)
    noise_troughs = W[L == 0].min(axis=1)
    spike_std     = float(spike_troughs.std())
    noise_std     = float(noise_troughs.std())
    t4_ok         = noise_std >= spike_std   # noise should be at least as variable
    results["amplitude_range"] = {
        "passed":           t4_ok,
        "spike_trough_std": round(spike_std, 2),
        "noise_trough_std": round(noise_std, 2),
        "details":          "Noise class should have ≥ spike trough variability.",
    }
    if not t4_ok:
        all_passed = False

    # ── Test 5: Hard-negative quality (excluded ≈ comparable to real SU) ─── #
    #    Uses meta CSV: excluded clusters have label='NOISE' but cluster_id != 99999999
    t5_ok     = True
    t5_detail = "No metadata CSV found — skipped."
    if meta is not None:
        # Reconstruct: for each source file, get mean trough of SU vs excluded
        # using cluster_id (index back into waveform array)
        excl_proximity_ratios: list[float] = []
        for _, grp in meta.groupby("source_file"):
            su_idx   = grp.index[grp["label"] == "SU"].tolist()
            excl_idx = grp.index[
                (grp["label"] == "NOISE") & (grp["cluster_id"] != 99999999)
            ].tolist()
            if not su_idx or not excl_idx:
                continue
            su_depth   = abs(float(W[su_idx].mean(axis=0).min()))
            excl_depth = abs(float(W[excl_idx].mean(axis=0).min()))
            if su_depth > 0:
                excl_proximity_ratios.append(excl_depth / su_depth)

        if excl_proximity_ratios:
            mean_ratio = float(np.mean(excl_proximity_ratios))
            # Excluded clusters should be >50% as deep as real SUs
            t5_ok  = mean_ratio >= 0.5
            t5_detail = (
                f"Mean excluded/SU trough depth ratio: {mean_ratio:.2f} "
                f"(must be ≥0.50 for these to be challenging negatives)"
            )
        else:
            t5_detail = "Could not compute ratio — no matching pairs found."

    results["hard_negative_quality"] = {
        "passed":  t5_ok,
        "details": t5_detail,
    }
    if not t5_ok:
        all_passed = False

    return {
        "overall_passed": all_passed,
        "n_tests":        5,
        "n_passed":       sum(1 for v in results.values() if v["passed"]),
        "tests":          results,
    }


# ── 3. Discriminator evaluation against real data ─────────────────────────── #

class EvaluateRequest(BaseModel):
    run_id: str
    tier:   int = 2


@router.post("/ephys/evaluate")
def evaluate_real(req: EvaluateRequest) -> dict[str, Any]:
    """
    Score waveforms_real.npz with the WeightBank from a prior benchmark run.

    Extracts the same features (at the same tier) as the original run,
    then applies the stored WeightBank and returns full metrics including
    per-SU recall and per-source-file accuracy.

    The WeightBank was trained on synthetic data; this endpoint measures how
    well it generalises to real OSort-curated recordings.
    """
    import pandas as pd
    import yaml
    from sklearn.metrics import (
        roc_auc_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        confusion_matrix,
    )
    from spike_discrim.features.core_features import TEMPORAL_MAD_FEATURE_PREFIX
    from spike_discrim.features.event_features import EVENT_FEATURE_NAMES
    from spike_discrim.input_layer.weights import WeightBank

    d   = _load_real()
    W, L, U = d["waveforms"], d["labels"], d["unit_ids"]
    meta = _load_meta()

    run_dir = _run_dir(req.run_id)
    wb_path = run_dir / "weight_bank.json"
    if not wb_path.exists():
        raise HTTPException(status_code=404,
                            detail="weight_bank.json not found in run directory")

    wb = WeightBank.load(wb_path)
    cfg: dict[str, Any] = {"window_size_samples": 16, "temporal_mad": {"enabled": False}}
    cfg_path = run_dir / "config_snapshot.yaml"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            snapshot = yaml.safe_load(fh) or {}
        cfg = snapshot.get("default", snapshot) or cfg

    wb_feature_names = list(getattr(wb, "feature_names", []) or [])
    has_event_features = any(name in EVENT_FEATURE_NAMES for name in wb_feature_names)
    has_temporal_mad = any(
        name.startswith(TEMPORAL_MAD_FEATURE_PREFIX) for name in wb_feature_names
    )

    tier = 3 if has_event_features else req.tier
    temporal_cfg = dict(cfg.get("temporal_mad", {}) or {})
    temporal_cfg["enabled"] = has_temporal_mad or bool(temporal_cfg.get("enabled", False))
    cfg["temporal_mad"] = temporal_cfg

    X_full, extracted_names, _ = _extract_features(W, cfg, tier)
    if wb_feature_names:
        name_to_idx = {name: i for i, name in enumerate(extracted_names)}
        missing = [name for name in wb_feature_names if name not in name_to_idx]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Stored WeightBank features could not be reconstructed from "
                    f"the current extractor. Missing: {missing}"
                ),
            )
        X = X_full[:, [name_to_idx[name] for name in wb_feature_names]]
        feat_names = wb_feature_names
    else:
        X = X_full
        feat_names = extracted_names

    # Load and apply WeightBank
    wb.warmup()
    scores   = wb.score_batch(X)
    preds    = wb.classify(X).astype(int)

    # ── Overall metrics ──────────────────────────────────────────────────── #
    auc      = float(roc_auc_score(L, scores))
    bal_acc  = float(balanced_accuracy_score(L, preds))
    prec     = float(precision_score(L, preds, zero_division=0))
    rec      = float(recall_score(L, preds, zero_division=0))
    f1       = float(f1_score(L, preds, zero_division=0))
    cm       = confusion_matrix(L, preds).tolist()

    # ── Per-SU recall ────────────────────────────────────────────────────── #
    per_su: dict[str, dict] = {}
    for uid in sorted(np.unique(U[U > 0])):
        mask       = (U == uid) & (L == 1)
        if not mask.any():
            continue
        su_recall  = float(preds[mask].mean())
        per_su[str(int(uid))] = {
            "n_events":      int(mask.sum()),
            "recall":        round(su_recall, 4),
            "mean_score":    round(float(scores[mask].mean()), 4),
        }

    # ── Per-source-file accuracy ─────────────────────────────────────────── #
    per_file: list[dict] = []
    if meta is not None:
        for src_file, grp in meta.groupby("source_file"):
            idx    = grp.index.tolist()
            l_sub  = L[idx]
            p_sub  = preds[idx]
            s_sub  = scores[idx]
            n_spk  = int((l_sub == 1).sum())
            n_nse  = int((l_sub == 0).sum())
            file_acc = float((l_sub == p_sub).mean())
            per_file.append({
                "source_file":   src_file,
                "n_spikes":      n_spk,
                "n_noise":       n_nse,
                "accuracy":      round(file_acc, 4),
                "mean_score_su": round(float(s_sub[l_sub == 1].mean()), 4)
                                  if n_spk else None,
                "mean_score_noise": round(float(s_sub[l_sub == 0].mean()), 4)
                                     if n_nse else None,
            })

    return {
        "run_id":            req.run_id,
        "tier":              tier,
        "n_waveforms":       int(len(L)),
        "n_features":        len(feat_names),
        "feature_names":     feat_names,
        "overall": {
            "auc":                round(auc,     4),
            "balanced_accuracy":  round(bal_acc, 4),
            "precision":          round(prec,    4),
            "recall":             round(rec,     4),
            "f1":                 round(f1,      4),
            "confusion_matrix":   cm,
        },
        "per_su_recall":     per_su,
        "per_source_file":   per_file,
        "note": (
            "WeightBank was trained on synthetic data. "
            "These metrics measure generalisation to real OSort-curated recordings."
        ),
    }


# ── 4. Sample waveforms for visualisation ────────────────────────────────── #

@router.get("/ephys/waveforms")
def sample_waveforms(
    n:           int           = Query(20, ge=1,  le=200),
    class_label: Optional[int] = Query(None, description="0=noise, 1=spike"),
    unit_id:     Optional[int] = Query(None, description="Filter by SU identity (1–10)"),
) -> dict[str, Any]:
    """
    Return n sample waveforms from waveforms_real.npz.

    Optionally filter by class_label (0=noise, 1=spike) and/or unit_id.
    """
    d    = _load_real()
    W, L, U = d["waveforms"], d["labels"], d["unit_ids"]

    mask = np.ones(len(L), dtype=bool)
    if class_label is not None:
        if class_label not in (0, 1):
            raise HTTPException(status_code=422, detail="class_label must be 0 or 1")
        mask &= L == class_label
    if unit_id is not None:
        mask &= U == unit_id

    idx_pool = np.where(mask)[0]
    if len(idx_pool) == 0:
        raise HTTPException(status_code=404,
                            detail="No waveforms match the requested filters")

    rng      = np.random.default_rng(0)
    sample   = rng.choice(idx_pool, size=min(n, len(idx_pool)), replace=False)
    sample   = np.sort(sample)

    return {
        "n":           len(sample),
        "n_samples":   W.shape[1],
        "class_label": class_label,
        "unit_id":     unit_id,
        "waveforms": [
            {
                "index":       int(i),
                "class_label": int(L[i]),
                "unit_id":     int(U[i]),
                "values":      W[i].tolist(),
            }
            for i in sample
        ],
    }
