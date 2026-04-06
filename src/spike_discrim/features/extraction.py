from __future__ import annotations

from typing import Any

import numpy as np

from spike_discrim.features.core_features import (
    SCALAR_FEATURE_NAMES,
    batch_extract_scalar_features,
    extract_temporal_mad_features,
)
from spike_discrim.features.event_features import (
    EVENT_FEATURE_NAMES,
    N_EVENT_FEATURES,
    batch_event_features,
)


def _normalise_temporal_mad_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(cfg or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "n_time_bins": int(raw.get("n_time_bins", 8)),
        "overlap_fraction": float(raw.get("overlap_fraction", 0.5)),
        "noise_mad_mode": str(raw.get("noise_mad_mode", "none")),
        "global_noise_mad": raw.get("global_noise_mad"),
        "edge_samples": int(raw.get("edge_samples", 8)),
        "mad_scale_factor": float(raw.get("mad_scale_factor", 1.4826)),
    }


def build_feature_matrix(
    waveforms: np.ndarray,
    cfg: dict[str, Any] | None = None,
    tier: int = 2,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build the feature matrix used by the CLI pipeline and API endpoints."""
    cfg = dict(cfg or {})
    waveforms = np.ascontiguousarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2:
        raise ValueError("waveforms must have shape [N, T]")

    n_waveforms, n_samples = waveforms.shape
    window = int(cfg.get("window_size_samples", 16))

    d1_buf = np.empty((n_waveforms, n_samples), dtype=np.float32)
    d2_buf = np.empty((n_waveforms, n_samples), dtype=np.float32)
    aws_buf = np.empty((n_waveforms, n_samples), dtype=np.float32)
    scalar_out = np.empty((n_waveforms, len(SCALAR_FEATURE_NAMES)), dtype=np.float32)
    batch_extract_scalar_features(
        waveforms,
        d1_buf,
        d2_buf,
        aws_buf,
        scalar_out,
        window,
    )

    feature_blocks: list[np.ndarray] = [scalar_out]
    feature_names: list[str] = list(SCALAR_FEATURE_NAMES)
    metadata: dict[str, Any] = {
        "scalar_features": {
            "feature_names": list(SCALAR_FEATURE_NAMES),
            "window_size_samples": int(window),
        }
    }

    temporal_cfg = _normalise_temporal_mad_config(cfg.get("temporal_mad"))
    if temporal_cfg["enabled"]:
        temporal_out, temporal_names, temporal_meta = extract_temporal_mad_features(
            waveforms=waveforms,
            n_bins=temporal_cfg["n_time_bins"],
            overlap_fraction=temporal_cfg["overlap_fraction"],
            noise_mad_mode=temporal_cfg["noise_mad_mode"],
            global_noise_mad=temporal_cfg["global_noise_mad"],
            edge_samples=temporal_cfg["edge_samples"],
            mad_scale_factor=temporal_cfg["mad_scale_factor"],
        )
        feature_blocks.append(temporal_out.astype(np.float32, copy=False))
        feature_names.extend(temporal_names)
        metadata["temporal_mad"] = temporal_meta

    if tier >= 3:
        event_out = np.empty((n_waveforms, N_EVENT_FEATURES), dtype=np.float32)
        batch_event_features(waveforms, event_out)
        feature_blocks.append(event_out)
        feature_names.extend(list(EVENT_FEATURE_NAMES))
        metadata["event_features"] = {
            "feature_names": list(EVENT_FEATURE_NAMES),
            "n_features": int(N_EVENT_FEATURES),
        }

    feature_matrix = np.hstack(feature_blocks).astype(np.float32, copy=False)
    metadata["n_features"] = int(len(feature_names))
    return feature_matrix, feature_names, metadata
