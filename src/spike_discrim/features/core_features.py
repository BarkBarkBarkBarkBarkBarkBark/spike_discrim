"""
core_features.py — Tier 1 and Tier 2 spike waveform features.

All kernels are Numba-JIT compiled for zero-copy, cache-friendly computation.
Each function writes into a pre-allocated output buffer; no heap allocation
occurs in the hot path.

Feature equations are taken from docs/features.yaml and
docs/spike_feature_validation.yaml.  Static op-counts (adds / muls /
mem-accesses per sample) are stored in OP_COUNTS for use by the profiling
layer without measurement overhead.

Causal flag convention
----------------------
  causal=True  → output[t] depends only on input[0..t]   (streaming-safe)
  causal=False → requires future samples; boundary filled with 0.0

Reference tiers (docs/features.yaml §feature_priority)
-------------------------------------------------------
  Tier 1:  amplitude, first_derivative, second_derivative
  Tier 2:  absolute_window_sum, short_window_energy, teager_energy
"""
from __future__ import annotations

import numpy as np
import numba


TEMPORAL_MAD_FEATURE_PREFIX = "mad_wta_bin"

# --------------------------------------------------------------------------- #
# Static operation counts per sample                                          #
# Source: docs/features.yaml §features[*].operations_per_sample               #
# Used by profiling.op_counter — do NOT modify without updating YAML          #
# --------------------------------------------------------------------------- #
OP_COUNTS: dict[str, dict[str, int]] = {
    "amplitude":           {"add": 0, "mul": 0, "mem_read": 1, "mem_write": 1,
                            "total_arith": 0, "total_mem": 2},
    "first_derivative":    {"add": 1, "mul": 0, "mem_read": 2, "mem_write": 1,
                            "total_arith": 1, "total_mem": 3},
    "second_derivative":   {"add": 2, "mul": 1, "mem_read": 3, "mem_write": 1,
                            "total_arith": 3, "total_mem": 4},
    "absolute_window_sum": {"add": 1, "mul": 0, "mem_read": 1, "mem_write": 1,
                            "total_arith": 1, "total_mem": 2},
    "short_window_energy": {"add": 1, "mul": 1, "mem_read": 1, "mem_write": 1,
                            "total_arith": 2, "total_mem": 2},
    "teager_energy":       {"add": 1, "mul": 2, "mem_read": 3, "mem_write": 1,
                            "total_arith": 3, "total_mem": 4},
}


# =========================================================================== #
# Single-snippet kernels — operate on 1-D float32 arrays                      #
# =========================================================================== #

@numba.njit(cache=True, fastmath=True)
def amplitude(waveform: np.ndarray, out: np.ndarray) -> None:
    """Tier 1 — Identity (amplitude trace).

    Equation : out[t] = x[t]
    Causal   : yes
    Ops/sample: 0 arith, 1 read, 1 write
    """
    n = waveform.shape[0]
    for t in range(n):
        out[t] = waveform[t]


@numba.njit(cache=True, fastmath=True)
def first_derivative(waveform: np.ndarray, out: np.ndarray) -> None:
    """Tier 1 — First derivative (slope / velocity).

    Equation : d1[t] = x[t] - x[t-1]
    Causal   : yes  |  Memory: 1 sample
    Ops/sample: 1 add, 2 reads, 1 write
    """
    n = waveform.shape[0]
    out[0] = 0.0
    for t in range(1, n):
        out[t] = waveform[t] - waveform[t - 1]


@numba.njit(cache=True, fastmath=True)
def second_derivative(waveform: np.ndarray, out: np.ndarray) -> None:
    """Tier 1 — Second derivative (curvature / acceleration).

    Equation : d2[t] = x[t] - 2·x[t-1] + x[t-2]
    Causal   : yes  |  Memory: 2 samples
    Ops/sample: 2 adds, 1 mul, 3 reads, 1 write
    """
    n = waveform.shape[0]
    out[0] = 0.0
    if n > 1:
        out[1] = 0.0
    for t in range(2, n):
        out[t] = waveform[t] - 2.0 * waveform[t - 1] + waveform[t - 2]


@numba.njit(cache=True, fastmath=True)
def absolute_window_sum(waveform: np.ndarray, out: np.ndarray, window: int) -> None:
    """Tier 2 — Absolute window sum (energy proxy, no multiplication).

    Equation : A[t] = Σ |x[i]|  for i in [t-window+1, t]
    Causal   : yes  |  Memory: window samples
    Ops/sample: 1 add (running sum update), 1 read, 1 write

    Uses a sliding-window running sum to maintain O(1) per step.
    """
    n = waveform.shape[0]
    running = 0.0
    for t in range(n):
        running += abs(waveform[t])
        if t >= window:
            running -= abs(waveform[t - window])
        out[t] = running


@numba.njit(cache=True, fastmath=True)
def short_window_energy(waveform: np.ndarray, out: np.ndarray, window: int) -> None:
    """Tier 2 — Short-window energy (signal power estimate).

    Equation : E[t] = Σ x[i]²  for i in [t-window+1, t]
    Causal   : yes  |  Memory: window samples
    Ops/sample: 1 add + 1 mul (running sum update via ring buffer)

    The ring buffer stores the squared value of each sample so the evicted
    term can be subtracted without re-accessing the waveform.
    """
    n = waveform.shape[0]
    ring = np.zeros(window, dtype=np.float64)
    running = 0.0
    for t in range(n):
        sq = waveform[t] * waveform[t]
        idx = t % window
        running -= ring[idx]
        ring[idx] = sq
        running += sq
        out[t] = running


@numba.njit(cache=True, fastmath=True)
def teager_energy(waveform: np.ndarray, out: np.ndarray) -> None:
    """Tier 2 — Teager-Kaiser energy operator.

    Equation : Ψ[t] = x[t]² - x[t-1]·x[t+1]
    Causal   : NO (requires x[t+1])  |  Memory: 3 samples
    Ops/sample: 1 add, 2 muls, 3 reads, 1 write

    Boundaries (t=0, t=N-1) are set to 0.  One-sample latency is acceptable
    when used as a post-detection shape feature.
    """
    n = waveform.shape[0]
    out[0] = 0.0
    for t in range(1, n - 1):
        out[t] = (waveform[t] * waveform[t]
                  - waveform[t - 1] * waveform[t + 1])
    if n > 1:
        out[n - 1] = 0.0


# =========================================================================== #
# Scalar summary reducers — compress feature time-series to a single value    #
# =========================================================================== #

@numba.njit(cache=True, fastmath=True)
def _max_val(arr: np.ndarray) -> float:
    best = arr[0]
    for v in arr:
        if v > best:
            best = v
    return best


@numba.njit(cache=True, fastmath=True)
def _min_val(arr: np.ndarray) -> float:
    best = arr[0]
    for v in arr:
        if v < best:
            best = v
    return best


@numba.njit(cache=True, fastmath=True)
def _abs_max(arr: np.ndarray) -> float:
    best = 0.0
    for v in arr:
        a = abs(v)
        if a > best:
            best = a
    return best


@numba.njit(cache=True)
def _median_1d(arr: np.ndarray) -> float:
    sorted_arr = np.sort(arr.copy())
    n = sorted_arr.shape[0]
    mid = n // 2
    if n % 2 == 0:
        return 0.5 * (sorted_arr[mid - 1] + sorted_arr[mid])
    return sorted_arr[mid]


@numba.njit(cache=True)
def _mad_1d(arr: np.ndarray) -> float:
    med = _median_1d(arr)
    dev = np.empty(arr.shape[0], dtype=np.float32)
    for i in range(arr.shape[0]):
        dev[i] = abs(arr[i] - med)
    return _median_1d(dev)


@numba.njit(cache=True)
def _window_robust_wta(waveform: np.ndarray, start: int, end: int) -> float:
    """Winner-take-all amplitude relative to a median-centered baseline."""
    n = end - start
    window = np.empty(n, dtype=np.float32)
    for i in range(n):
        window[i] = waveform[start + i]

    med = _median_1d(window)
    best = 0.0
    for i in range(n):
        dev = abs(window[i] - med)
        if dev > best:
            best = dev
    return best


@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_estimate_snippet_edge_noise_mad(
    waveforms: np.ndarray,
    edge_samples: int,
    out: np.ndarray,
) -> None:
    """Estimate per-snippet baseline noise from leading and trailing edges."""
    n_waveforms, n_samples = waveforms.shape
    use_edge = min(edge_samples, max(1, n_samples // 2))
    baseline_len = use_edge * 2

    for i in numba.prange(n_waveforms):
        baseline = np.empty(baseline_len, dtype=np.float32)
        for j in range(use_edge):
            baseline[j] = waveforms[i, j]
            baseline[use_edge + j] = waveforms[i, n_samples - use_edge + j]
        out[i] = _mad_1d(baseline)


@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_extract_temporal_mad_features(
    waveforms: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    noise_scales: np.ndarray,
    out: np.ndarray,
) -> None:
    """Extract flattened overlapping-window robust amplitude features.

    Each output column is the winner-take-all absolute deviation from the
    median within one overlapping temporal window. If ``noise_scales[i] > 0``,
    the feature is normalised by that snippet's MAD-derived noise estimate.
    """
    n_waveforms = waveforms.shape[0]
    n_bins = window_starts.shape[0]

    for i in numba.prange(n_waveforms):
        scale = noise_scales[i]
        for b in range(n_bins):
            value = _window_robust_wta(waveforms[i], window_starts[b], window_ends[b])
            if scale > 1e-8:
                out[i, b] = value / scale
            else:
                out[i, b] = value


def compute_temporal_window_bounds(
    n_samples: int,
    n_bins: int,
    overlap_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return start/end indices for overlapping temporal bins.

    The windows are spread across the snippet so the first bin starts at 0 and
    the last bin ends at ``n_samples``. ``overlap_fraction`` should be in
    ``[0.0, 0.95)``; larger values create more overlap and longer windows.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if not 0.0 <= overlap_fraction < 0.95:
        raise ValueError("overlap_fraction must be in [0.0, 0.95)")

    if n_bins == 1:
        return np.array([0], dtype=np.int32), np.array([n_samples], dtype=np.int32)

    coverage = 1.0 + (n_bins - 1) * (1.0 - overlap_fraction)
    window_len = int(np.ceil(n_samples / coverage))
    window_len = max(1, min(window_len, n_samples))

    max_start = max(0, n_samples - window_len)
    starts = np.round(np.linspace(0.0, float(max_start), n_bins)).astype(np.int32)
    ends = np.minimum(starts + window_len, n_samples).astype(np.int32)
    starts = np.maximum(0, ends - window_len).astype(np.int32)

    starts[0] = 0
    ends[-1] = n_samples
    return starts, ends


def build_temporal_mad_feature_names(
    n_bins: int,
    prefix: str = TEMPORAL_MAD_FEATURE_PREFIX,
) -> list[str]:
    """Build flattened column names for temporal MAD/WTA features."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return [f"{prefix}_{i:02d}" for i in range(n_bins)]


def extract_temporal_mad_features(
    waveforms: np.ndarray,
    n_bins: int,
    overlap_fraction: float = 0.5,
    noise_mad_mode: str = "none",
    global_noise_mad: float | None = None,
    edge_samples: int = 8,
    mad_scale_factor: float = 1.4826,
) -> tuple[np.ndarray, list[str], dict]:
    """Extract flattened overlapping-window afferent features.

    Parameters
    ----------
    waveforms : float32[N, T]
        Input snippets.
    n_bins : int
        Number of temporal bins / timesteps to emit.
    overlap_fraction : float
        Temporal overlap between adjacent bins.
    noise_mad_mode : {'none', 'global', 'snippet_edges'}
        How to obtain the MAD-derived noise scale used for normalisation.
    global_noise_mad : float, optional
        External channel-noise MAD from a calibration phase.
    edge_samples : int
        Number of samples to take from each snippet edge when
        ``noise_mad_mode='snippet_edges'``.
    mad_scale_factor : float
        Optional conversion from MAD to sigma-equivalent scale.
    """
    waveforms = np.ascontiguousarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2:
        raise ValueError("waveforms must have shape [N, T]")
    if edge_samples <= 0:
        raise ValueError("edge_samples must be positive")
    if mad_scale_factor <= 0.0:
        raise ValueError("mad_scale_factor must be positive")

    noise_mode = noise_mad_mode.strip().lower()
    if noise_mode not in {"none", "global", "snippet_edges"}:
        raise ValueError("noise_mad_mode must be one of: none, global, snippet_edges")
    if noise_mode == "global" and (global_noise_mad is None or global_noise_mad <= 0.0):
        raise ValueError("global_noise_mad must be provided and positive for global mode")

    n_waveforms, n_samples = waveforms.shape
    starts, ends = compute_temporal_window_bounds(
        n_samples=n_samples,
        n_bins=n_bins,
        overlap_fraction=overlap_fraction,
    )

    if noise_mode == "none":
        noise_scales = np.zeros(n_waveforms, dtype=np.float32)
    elif noise_mode == "global":
        scale = np.float32(global_noise_mad * mad_scale_factor)
        noise_scales = np.full(n_waveforms, scale, dtype=np.float32)
    else:
        noise_scales = np.empty(n_waveforms, dtype=np.float32)
        batch_estimate_snippet_edge_noise_mad(waveforms, int(edge_samples), noise_scales)
        noise_scales *= np.float32(mad_scale_factor)

    out = np.empty((n_waveforms, n_bins), dtype=np.float32)
    batch_extract_temporal_mad_features(
        waveforms,
        starts.astype(np.int32),
        ends.astype(np.int32),
        noise_scales,
        out,
    )

    feature_names = build_temporal_mad_feature_names(n_bins)
    metadata = {
        "feature_family": "temporal_mad_wta",
        "feature_prefix": TEMPORAL_MAD_FEATURE_PREFIX,
        "n_time_bins": int(n_bins),
        "overlap_fraction": float(overlap_fraction),
        "window_starts": starts.tolist(),
        "window_ends": ends.tolist(),
        "window_lengths": (ends - starts).tolist(),
        "noise_mad_mode": noise_mode,
        "global_noise_mad": None if global_noise_mad is None else float(global_noise_mad),
        "edge_samples": int(edge_samples),
        "mad_scale_factor": float(mad_scale_factor),
        "winner_take_all": True,
    }
    return out, feature_names, metadata


# =========================================================================== #
# Batch kernels — process N snippets in parallel                               #
# waveforms: float32[N, T] (C-contiguous)                                     #
# out:       float32[N, T] pre-allocated                                      #
# =========================================================================== #

@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_amplitude(waveforms: np.ndarray, out: np.ndarray) -> None:
    """Batch amplitude for N snippets."""
    for i in numba.prange(waveforms.shape[0]):
        amplitude(waveforms[i], out[i])


@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_first_derivative(waveforms: np.ndarray, out: np.ndarray) -> None:
    """Batch first derivative for N snippets."""
    for i in numba.prange(waveforms.shape[0]):
        first_derivative(waveforms[i], out[i])


@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_second_derivative(waveforms: np.ndarray, out: np.ndarray) -> None:
    """Batch second derivative for N snippets."""
    for i in numba.prange(waveforms.shape[0]):
        second_derivative(waveforms[i], out[i])


@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_absolute_window_sum(waveforms: np.ndarray, out: np.ndarray,
                               window: int) -> None:
    """Batch absolute window sum for N snippets."""
    for i in numba.prange(waveforms.shape[0]):
        absolute_window_sum(waveforms[i], out[i], window)


@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_short_window_energy(waveforms: np.ndarray, out: np.ndarray,
                               window: int) -> None:
    """Batch short-window energy for N snippets."""
    for i in numba.prange(waveforms.shape[0]):
        short_window_energy(waveforms[i], out[i], window)


@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_teager_energy(waveforms: np.ndarray, out: np.ndarray) -> None:
    """Batch Teager energy for N snippets."""
    for i in numba.prange(waveforms.shape[0]):
        teager_energy(waveforms[i], out[i])


# =========================================================================== #
# Combined scalar feature extraction — single parallel pass over all snippets #
#                                                                              #
# Returns float32[N, 6] scalar feature matrix:                                #
#   [0] peak_amplitude      = max(x)                                          #
#   [1] trough_amplitude    = min(x)                                          #
#   [2] max_slope           = max(d1)                                         #
#   [3] min_slope           = min(d1)                                         #
#   [4] max_abs_curvature   = max(|d2|)                                       #
#   [5] abs_window_sum_peak = max(AWS)                                        #
# =========================================================================== #

@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_extract_scalar_features(
    waveforms: np.ndarray,   # float32[N, T]
    d1_buf:    np.ndarray,   # float32[N, T]  pre-allocated scratch
    d2_buf:    np.ndarray,   # float32[N, T]  pre-allocated scratch
    aws_buf:   np.ndarray,   # float32[N, T]  pre-allocated scratch
    out:       np.ndarray,   # float32[N, 6]  pre-allocated output
    window:    int,
) -> None:
    """Extract 6 scalar Tier-1/2 features per snippet in one parallel pass.

    Caller must pre-allocate all buffers.  No copies are made inside.
    Each prange iteration owns a private row of every buffer — no races.
    """
    for i in numba.prange(waveforms.shape[0]):
        w   = waveforms[i]
        d1  = d1_buf[i]
        d2  = d2_buf[i]
        aws = aws_buf[i]

        first_derivative(w, d1)
        second_derivative(w, d2)
        absolute_window_sum(w, aws, window)

        out[i, 0] = _max_val(w)
        out[i, 1] = _min_val(w)
        out[i, 2] = _max_val(d1)
        out[i, 3] = _min_val(d1)
        out[i, 4] = _abs_max(d2)
        out[i, 5] = _max_val(aws)


# ── Column name mapping for the scalar feature matrix ─────────────────────
SCALAR_FEATURE_NAMES: list[str] = [
    "peak_amplitude",
    "trough_amplitude",
    "max_slope",
    "min_slope",
    "max_abs_curvature",
    "abs_window_sum_peak",
]
