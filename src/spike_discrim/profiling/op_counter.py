"""
op_counter.py — Operation counting and wall-clock profiling for feature kernels.

Two complementary measurements are combined:

1. **Static op count** — exact arithmetic and memory-access counts per sample,
   sourced from docs/features.yaml §features[*].operations_per_sample.
   These are architecture-independent and useful for FPGA / embedded projection.

2. **Wall-clock time** — nanosecond-resolution timing via time.perf_counter_ns,
   measured AFTER Numba JIT warmup.  Represents actual CPU throughput on the
   host machine.

Profiling results are saved as JSON to `data/profiling/` for full transparency.

Design notes
------------
- All timing is done OUTSIDE JIT-compiled kernels to avoid
  adding instrumentation to the hot path.
- Numba JIT compilation happens on the first call.  The profiler always warms
  up the kernel (n_warmup calls) before timing.
- Op counts are STATIC (compile-time constants) — do NOT update them without
  also updating docs/features.yaml.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np


# --------------------------------------------------------------------------- #
# Static operation count registry                                             #
# Sourced from docs/features.yaml §features[*].operations_per_sample         #
# --------------------------------------------------------------------------- #

FEATURE_OP_COUNTS: dict[str, dict[str, int]] = {
    "amplitude": {
        "add": 0, "mul": 0, "mem_read": 1, "mem_write": 1,
        "total_arith": 0, "total_mem": 2,
    },
    "first_derivative": {
        "add": 1, "mul": 0, "mem_read": 2, "mem_write": 1,
        "total_arith": 1, "total_mem": 3,
    },
    "second_derivative": {
        "add": 2, "mul": 1, "mem_read": 3, "mem_write": 1,
        "total_arith": 3, "total_mem": 4,
    },
    "absolute_window_sum": {
        "add": 1, "mul": 0, "mem_read": 1, "mem_write": 1,
        "total_arith": 1, "total_mem": 2,
    },
    "short_window_energy": {
        "add": 1, "mul": 1, "mem_read": 1, "mem_write": 1,
        "total_arith": 2, "total_mem": 2,
    },
    "teager_energy": {
        "add": 1, "mul": 2, "mem_read": 3, "mem_write": 1,
        "total_arith": 3, "total_mem": 4,
    },
    "trough_amplitude":       {"total_arith": 0, "total_mem": 1},
    "peak_amplitude":         {"total_arith": 0, "total_mem": 1},
    "trough_to_peak_time":    {"total_arith": 0, "total_mem": 2},
    "half_width":             {"total_arith": 1, "total_mem": 1},
    "biphasic_ratio":         {"total_arith": 2, "total_mem": 1},
    "baseline_rms":           {"total_arith": 2, "total_mem": 1},
    "zero_crossing_count":    {"total_arith": 0, "total_mem": 1},
    "signed_area":            {"total_arith": 1, "total_mem": 1},
    "absolute_area":          {"total_arith": 1, "total_mem": 1},
    "max_rising_slope":       {"total_arith": 1, "total_mem": 2},
    "max_falling_slope":      {"total_arith": 1, "total_mem": 2},
}


# --------------------------------------------------------------------------- #
# Result dataclass                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class ProfileResult:
    """Complete profiling result for one kernel + dataset pair."""
    # Identity
    feature_name:             str
    backend:                  str   = "numba_jit"
    # Dataset dimensions
    n_snippets:               int   = 0
    n_samples_per_snippet:    int   = 0
    total_samples:            int   = 0
    # Static cost (ops / sample, from YAML)
    ops_add_per_sample:       int   = -1
    ops_mul_per_sample:       int   = -1
    ops_mem_read_per_sample:  int   = -1
    ops_mem_write_per_sample: int   = -1
    total_arith_ops:          int   = -1
    total_mem_ops:            int   = -1
    # Wall-clock timing
    wall_time_ns:             int   = 0
    wall_time_ms:             float = 0.0
    throughput_snippets_per_sec: float = 0.0
    throughput_samples_per_sec:  float = 0.0
    # Environment
    run_timestamp:            str   = ""
    python_version:           str   = ""
    numba_version:            str   = ""
    notes:                    str   = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def __str__(self) -> str:
        return (
            f"ProfileResult({self.feature_name}): "
            f"{self.n_snippets} snippets × {self.n_samples_per_snippet} samples | "
            f"wall={self.wall_time_ms:.2f} ms | "
            f"{self.throughput_snippets_per_sec:,.0f} snippets/s | "
            f"static_arith={self.ops_add_per_sample + self.ops_mul_per_sample} ops/sample"
        )


# --------------------------------------------------------------------------- #
# Context manager for nanosecond wall-clock timing                            #
# --------------------------------------------------------------------------- #

@dataclass
class _TimingContext:
    elapsed_ns: int   = 0
    elapsed_ms: float = 0.0
    _start: int       = 0


@contextmanager
def timer() -> Iterator[_TimingContext]:
    """Nanosecond-resolution wall-clock timer.

    Usage::

        with timer() as t:
            some_numba_kernel(data, out)
        print(t.elapsed_ms)

    IMPORTANT: For Numba JIT functions, always warm up the kernel
    (call it once with small input) BEFORE entering this context.
    """
    ctx = _TimingContext()
    ctx._start = time.perf_counter_ns()
    try:
        yield ctx
    finally:
        ctx.elapsed_ns = time.perf_counter_ns() - ctx._start
        ctx.elapsed_ms = ctx.elapsed_ns / 1_000_000.0


# --------------------------------------------------------------------------- #
# High-level profiling helper                                                 #
# --------------------------------------------------------------------------- #

def profile_feature(
    feature_name:  str,
    kernel_fn:     Callable[..., Any],
    waveforms:     np.ndarray,          # float32[N, T]
    n_warmup:      int  = 3,
    profiling_dir: str | Path = "data/profiling",
    notes:         str  = "",
    **kernel_kwargs,
) -> ProfileResult:
    """Profile a batch feature kernel end-to-end.

    Steps:
    1. Allocate output buffer (same shape as waveforms).
    2. Warm up JIT (n_warmup calls on a small slice).
    3. Time the kernel on the full waveforms array.
    4. Compute static op counts and throughput.
    5. Save JSON to profiling_dir.
    6. Return ProfileResult.

    Parameters
    ----------
    feature_name  : Key in FEATURE_OP_COUNTS (e.g. "first_derivative").
    kernel_fn     : Batch Numba kernel, signature fn(waveforms, out, **kwargs).
    waveforms     : float32[N, T] batch of waveform snippets.
    n_warmup      : Warm-up calls before timing.
    profiling_dir : Directory for JSON output.
    notes         : Optional annotation (e.g. hardware description).
    **kernel_kwargs : Extra kwargs forwarded to kernel_fn (e.g. window=16).
    """
    n, t = waveforms.shape
    out_buf = np.empty_like(waveforms)

    # Warm up
    small     = waveforms[:min(4, n)]
    small_out = np.empty_like(small)
    for _ in range(n_warmup):
        kernel_fn(small, small_out, **kernel_kwargs)

    # Timed run
    with timer() as t_ctx:
        kernel_fn(waveforms, out_buf, **kernel_kwargs)

    elapsed_ns = t_ctx.elapsed_ns
    elapsed_ms = t_ctx.elapsed_ms
    sec        = elapsed_ns / 1e9 if elapsed_ns > 0 else 1e-9
    thr_snip   = n   / sec
    thr_samp   = n*t / sec

    # Static counts
    op = FEATURE_OP_COUNTS.get(feature_name, {})

    # Environment
    try:
        python_ver = sys.version.split()[0]
    except Exception:
        python_ver = "unknown"
    try:
        import numba as _nb
        numba_ver = _nb.__version__
    except Exception:
        numba_ver = "unknown"

    result = ProfileResult(
        feature_name             = feature_name,
        backend                  = "numba_jit",
        n_snippets               = n,
        n_samples_per_snippet    = t,
        total_samples            = n * t,
        ops_add_per_sample       = op.get("add",       -1),
        ops_mul_per_sample       = op.get("mul",       -1),
        ops_mem_read_per_sample  = op.get("mem_read",  -1),
        ops_mem_write_per_sample = op.get("mem_write", -1),
        total_arith_ops          = op.get("total_arith", -1) * n * t,
        total_mem_ops            = op.get("total_mem",   -1) * n * t,
        wall_time_ns             = elapsed_ns,
        wall_time_ms             = elapsed_ms,
        throughput_snippets_per_sec = thr_snip,
        throughput_samples_per_sec  = thr_samp,
        run_timestamp            = datetime.now(timezone.utc).isoformat(),
        python_version           = python_ver,
        numba_version            = numba_ver,
        notes                    = notes,
    )

    profiling_dir = Path(profiling_dir)
    profiling_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = profiling_dir / f"{feature_name}_{ts}.json"
    result.save(fname)

    return result


def profile_all_features(
    waveforms:     np.ndarray,
    window:        int  = 16,
    n_warmup:      int  = 3,
    profiling_dir: str | Path = "data/profiling",
    notes:         str  = "",
) -> dict[str, ProfileResult]:
    """Profile all Tier 1 + 2 batch kernels in sequence.

    Returns a dict mapping feature_name → ProfileResult.
    All results are also saved as JSON to profiling_dir.
    """
    from spike_discrim.features.core_features import (
        batch_amplitude, batch_first_derivative, batch_second_derivative,
        batch_absolute_window_sum, batch_short_window_energy, batch_teager_energy,
    )

    kernels = {
        "amplitude":           (batch_amplitude,           {}),
        "first_derivative":    (batch_first_derivative,    {}),
        "second_derivative":   (batch_second_derivative,   {}),
        "absolute_window_sum": (batch_absolute_window_sum, {"window": window}),
        "short_window_energy": (batch_short_window_energy, {"window": window}),
        "teager_energy":       (batch_teager_energy,       {}),
    }

    results: dict[str, ProfileResult] = {}
    for name, (fn, kwargs) in kernels.items():
        r = profile_feature(
            feature_name  = name,
            kernel_fn     = fn,
            waveforms     = waveforms,
            n_warmup      = n_warmup,
            profiling_dir = profiling_dir,
            notes         = notes,
            **kwargs,
        )
        results[name] = r
        print(r)

    return results
