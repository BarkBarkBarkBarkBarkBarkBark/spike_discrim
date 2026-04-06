"""
event_features.py — Tier 3 event-based spike waveform features.

These features require a complete waveform snippet (event window) and are
computed after detection.  They provide stronger unit-identity discrimination
than Tier 1/2 but add latency proportional to snippet length.

All kernels are Numba-JIT compiled and operate on 1-D float32 arrays.
Batch variants accept float32[N, T] and write into pre-allocated float32[N, K].

Batch output column mapping (EVENT_FEATURE_NAMES):
  [0]  ev_trough_amplitude           — min(x)
  [1]  ev_peak_amplitude             — max(x)
  [2]  ev_trough_to_peak_time_samples— samples from trough to subsequent peak
  [3]  ev_half_width_samples         — samples ≤ 0.5 × trough_amplitude
  [4]  ev_full_width_samples         — samples ≤ 0.1 × trough_amplitude
  [5]  ev_biphasic_ratio             — |peak| / (|trough| + ε)
  [6]  ev_signed_area                — Σ x[t]
  [7]  ev_absolute_area              — Σ |x[t]|
  [8]  ev_max_rising_slope           — max(d1)
  [9]  ev_max_falling_slope          — min(d1)
  [10] ev_baseline_rms               — RMS of first n_baseline samples
  [11] ev_zero_crossing_count        — number of sign changes
"""
from __future__ import annotations

import numpy as np
import numba

N_EVENT_FEATURES: int = 12

EVENT_FEATURE_NAMES: list[str] = [
    "ev_trough_amplitude",
    "ev_peak_amplitude",
    "ev_trough_to_peak_time_samples",
    "ev_half_width_samples",
    "ev_full_width_samples",
    "ev_biphasic_ratio",
    "ev_signed_area",
    "ev_absolute_area",
    "ev_max_rising_slope",
    "ev_max_falling_slope",
    "ev_baseline_rms",
    "ev_zero_crossing_count",
]


# =========================================================================== #
# Single-snippet scalar functions                                              #
# =========================================================================== #

@numba.njit(cache=True, fastmath=True)
def trough_amplitude(waveform: np.ndarray) -> float:
    """Minimum (most negative) value — trough of the extracellular spike."""
    return np.min(waveform)


@numba.njit(cache=True, fastmath=True)
def peak_amplitude(waveform: np.ndarray) -> float:
    """Maximum (most positive) value — post-spike positive deflection."""
    return np.max(waveform)


@numba.njit(cache=True, fastmath=True)
def trough_to_peak_time(waveform: np.ndarray) -> int:
    """Samples from trough to the subsequent peak.

    The trough is the global minimum.  The peak is the maximum value
    *after* the trough (captures the repolarisation peak, not pre-spike
    positive deflection).  This is a stable unit-identity feature because
    it reflects the action potential waveform duration.
    """
    n = waveform.shape[0]
    t_trough = 0
    mn = waveform[0]
    for t in range(n):
        if waveform[t] < mn:
            mn = waveform[t]
            t_trough = t

    t_peak = t_trough
    mx = waveform[t_trough]
    for t in range(t_trough, n):
        if waveform[t] > mx:
            mx = waveform[t]
            t_peak = t

    return t_peak - t_trough


@numba.njit(cache=True, fastmath=True)
def half_width(waveform: np.ndarray) -> int:
    """Samples where waveform ≤ 0.5 × trough_amplitude.

    trough_amplitude is negative, so 0.5× is less negative — we count
    how many samples fall deeper than the half-depth mark.  Broader spikes
    (slower units) have larger half-width values.
    """
    n = waveform.shape[0]
    mn = np.min(waveform)
    half = mn * 0.5
    count = 0
    for t in range(n):
        if waveform[t] <= half:
            count += 1
    return count


@numba.njit(cache=True, fastmath=True)
def full_width(waveform: np.ndarray) -> int:
    """Samples where waveform ≤ 0.1 × trough_amplitude (10% threshold)."""
    n = waveform.shape[0]
    mn = np.min(waveform)
    thr = mn * 0.1
    count = 0
    for t in range(n):
        if waveform[t] <= thr:
            count += 1
    return count


@numba.njit(cache=True, fastmath=True)
def biphasic_ratio(waveform: np.ndarray) -> float:
    """Ratio |peak_amplitude| / (|trough_amplitude| + ε).

    < 1  → trough-dominant (typical extracellular spike)
    ≈ 1  → symmetric biphasic
    > 1  → positive peak dominates (uncommon; may indicate artifact)
    """
    pk = np.max(waveform)
    tr = np.min(waveform)
    return abs(pk) / (abs(tr) + 1e-12)


@numba.njit(cache=True, fastmath=True)
def signed_area(waveform: np.ndarray) -> float:
    """Sum of all sample values (proportional to net charge)."""
    total = 0.0
    for v in waveform:
        total += v
    return total


@numba.njit(cache=True, fastmath=True)
def absolute_area(waveform: np.ndarray) -> float:
    """Sum of |x[t]| — total energy without sign cancellation."""
    total = 0.0
    for v in waveform:
        total += abs(v)
    return total


@numba.njit(cache=True, fastmath=True)
def max_rising_slope(waveform: np.ndarray) -> float:
    """Maximum positive first derivative (fastest rising edge)."""
    n = waveform.shape[0]
    best = 0.0
    for t in range(1, n):
        d = waveform[t] - waveform[t - 1]
        if d > best:
            best = d
    return best


@numba.njit(cache=True, fastmath=True)
def max_falling_slope(waveform: np.ndarray) -> float:
    """Maximum negative first derivative (fastest falling edge, returned as negative)."""
    n = waveform.shape[0]
    worst = 0.0
    for t in range(1, n):
        d = waveform[t] - waveform[t - 1]
        if d < worst:
            worst = d
    return worst


@numba.njit(cache=True, fastmath=True)
def baseline_rms(waveform: np.ndarray, n_baseline: int = 8) -> float:
    """RMS of the first n_baseline samples (pre-event baseline).

    n_baseline=8 samples = 0.27 ms at 30 kHz.  The first 8 samples of a
    properly extracted snippet should be near baseline.  Large baseline_rms
    indicates a noisy recording or incorrectly aligned snippet.
    """
    n = min(n_baseline, waveform.shape[0])
    total = 0.0
    for t in range(n):
        total += waveform[t] * waveform[t]
    return (total / n) ** 0.5


@numba.njit(cache=True, fastmath=True)
def zero_crossing_count(waveform: np.ndarray) -> int:
    """Number of times the waveform crosses zero.

    A typical extracellular spike crosses zero 1–2 times.
    Noise and multi-unit hash typically have > 4 zero crossings.
    This is a strong noise-veto feature.
    """
    n = waveform.shape[0]
    count = 0
    for t in range(1, n):
        prev = waveform[t - 1]
        curr = waveform[t]
        if (prev >= 0.0 and curr < 0.0) or (prev < 0.0 and curr >= 0.0):
            count += 1
    return count


# =========================================================================== #
# Batch extraction — writes all 12 scalars per snippet in one parallel pass   #
# =========================================================================== #

@numba.njit(parallel=True, cache=True, fastmath=True)
def batch_event_features(
    waveforms: np.ndarray,  # float32[N, T]
    out:       np.ndarray,  # float32[N, 12]  pre-allocated
) -> None:
    """Extract all 12 Tier-3 scalar event features for N snippets.

    Column order matches EVENT_FEATURE_NAMES exactly.
    No scratch buffers needed — each feature is a pure scalar reduction.
    """
    for i in numba.prange(waveforms.shape[0]):
        w = waveforms[i]
        out[i, 0]  = trough_amplitude(w)
        out[i, 1]  = peak_amplitude(w)
        out[i, 2]  = np.float32(trough_to_peak_time(w))
        out[i, 3]  = np.float32(half_width(w))
        out[i, 4]  = np.float32(full_width(w))
        out[i, 5]  = biphasic_ratio(w)
        out[i, 6]  = signed_area(w)
        out[i, 7]  = absolute_area(w)
        out[i, 8]  = max_rising_slope(w)
        out[i, 9]  = max_falling_slope(w)
        out[i, 10] = baseline_rms(w)
        out[i, 11] = np.float32(zero_crossing_count(w))
