"""
weights.py — Quantile-initialized population-code WeightBank for the SNN input layer.

Design rationale: docs/MANIFEST.md §4

The WeightBank maps each scalar feature to a bank of N_bins Gaussian-tuned
neurons.  Bin centers are set from quantiles of a calibration spike set, so
each neuron covers an equal fraction of the spike feature distribution.

  Real spike  → feature values fall within learned bins → high activation
  Noise event → wrong shape/amplitude → sparse activation → low score → rejected

Scoring is performed by a Numba-JIT kernel (_batch_score_jit) that operates
on pre-allocated float32 arrays — no Python-level loops or copies in the hot path.

Usage
-----
    from spike_discrim.input_layer.weights import WeightBank
    import numpy as np

    # feature_matrix: float32[N_spikes, n_features]
    wb = WeightBank(n_bins=10, sigma_scale=1.0, threshold=0.5)
    wb.fit(feature_matrix, feature_names=["peak_amplitude", "max_slope", ...])
    wb.warmup()                          # pre-compile JIT kernels

    scores = wb.score_batch(test_features)   # float32[N_test]
    is_spike = wb.classify(test_features)    # bool[N_test]
    wb.save("data/results/weight_bank.json") # transparent inspection
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numba


# =========================================================================== #
# Numba JIT scoring kernels                                                   #
# =========================================================================== #

@numba.njit(cache=True, fastmath=True)
def _score_snippet_jit(
    fv:      np.ndarray,   # float32[n_features]
    centers: np.ndarray,   # float32[n_features, n_bins]
    widths:  np.ndarray,   # float32[n_features, n_bins]
    fw:      np.ndarray,   # float32[n_features]  — per-feature weights
) -> float:
    """Population-code discriminant score for one feature vector.

    For each feature f:
        a_f = max_b  exp(-0.5 · ((x_f - c_fb) / w_fb)²)

    Total score = Σ_f (fw_f · a_f) / Σ_f fw_f

    Real spikes activate most feature banks → score ≈ 1.
    Noise produces off-center activations → score ≪ 1.
    """
    n_features = fv.shape[0]
    n_bins     = centers.shape[1]
    total        = 0.0
    total_weight = 0.0

    for f in range(n_features):
        x = fv[f]
        max_act = 0.0
        for b in range(n_bins):
            dx = (x - centers[f, b]) / (widths[f, b] + 1e-10)
            act = np.exp(-0.5 * dx * dx)
            if act > max_act:
                max_act = act
        total        += fw[f] * max_act
        total_weight += fw[f]

    if total_weight > 0.0:
        return total / total_weight
    return 0.0


@numba.njit(parallel=True, cache=True, fastmath=True)
def _batch_score_jit(
    fvs:     np.ndarray,   # float32[N, n_features]
    centers: np.ndarray,   # float32[n_features, n_bins]
    widths:  np.ndarray,   # float32[n_features, n_bins]
    fw:      np.ndarray,   # float32[n_features]
    out:     np.ndarray,   # float32[N]  pre-allocated
) -> None:
    """Batch scoring for N feature vectors (parallel over snippets)."""
    for i in numba.prange(fvs.shape[0]):
        out[i] = _score_snippet_jit(fvs[i], centers, widths, fw)


@numba.njit(cache=True, fastmath=True)
def _project_snippet_jit(
    fv:      np.ndarray,   # float32[n_features]
    centers: np.ndarray,   # float32[n_features, n_bins]
    widths:  np.ndarray,   # float32[n_features, n_bins]
    out:     np.ndarray,   # float32[n_features, n_bins]
) -> None:
    """Project one feature vector into afferent activation space."""
    n_features = fv.shape[0]
    n_bins = centers.shape[1]

    for f in range(n_features):
        x = fv[f]
        for b in range(n_bins):
            dx = (x - centers[f, b]) / (widths[f, b] + 1e-10)
            out[f, b] = np.exp(-0.5 * dx * dx)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _batch_project_jit(
    fvs:     np.ndarray,   # float32[N, n_features]
    centers: np.ndarray,   # float32[n_features, n_bins]
    widths:  np.ndarray,   # float32[n_features, n_bins]
    out:     np.ndarray,   # float32[N, n_features, n_bins]
) -> None:
    """Project N feature vectors into afferent activation space."""
    for i in numba.prange(fvs.shape[0]):
        _project_snippet_jit(fvs[i], centers, widths, out[i])


# =========================================================================== #
# WeightBank class                                                             #
# =========================================================================== #

@dataclass
class WeightBank:
    """Population-coded input layer weight bank.

    Parameters
    ----------
    n_bins : int
        Number of Gaussian tuning neurons per feature.  10 gives a good
        resolution/parameter tradeoff (see docs/MANIFEST.md §4.2).
    sigma_scale : float
        Scaling factor for bin widths.  1.0 = width equals half the
        inter-quantile spacing, ensuring smooth coverage with minimal overlap.
        Increase for a more tolerant gate; decrease for a sharper gate.
    threshold : float
        Score threshold for spike/noise binary decision.  Default 0.5.
        Tune from ROC analysis on a labelled validation set.
    """
    n_bins:      int   = 10
    sigma_scale: float = 1.0
    threshold:   float = 0.5

    # Set by fit()
    feature_names:     list          = field(default_factory=list, repr=False)
    centers_:          Optional[np.ndarray] = field(default=None,  repr=False)
    widths_:           Optional[np.ndarray] = field(default=None,  repr=False)
    fw_:               Optional[np.ndarray] = field(default=None,  repr=False)
    quantile_levels_:  Optional[np.ndarray] = field(default=None,  repr=False)

    # ── Fit ──────────────────────────────────────────────────────────────── #

    def fit(
        self,
        feature_matrix:  np.ndarray,              # float32[N, n_features]
        feature_names:   Optional[list]     = None,
        feature_weights: Optional[np.ndarray] = None,
    ) -> "WeightBank":
        """Fit bin centers and widths from quantiles of calibration spikes.

        Parameters
        ----------
        feature_matrix : float32[N_spikes, n_features]
            Scalar feature vectors from known-good spike waveforms.
        feature_names : list[str], optional
            Labels for each feature column.
        feature_weights : float32[n_features], optional
            Per-feature importance weights.  Default: uniform (all ones).

        Returns self for chaining.
        """
        n_spikes, n_features = feature_matrix.shape
        self.feature_names = feature_names or [f"f{i}" for i in range(n_features)]

        q_levels = np.linspace(0.0, 1.0, self.n_bins)
        self.quantile_levels_ = q_levels

        centers = np.empty((n_features, self.n_bins), dtype=np.float32)
        widths  = np.empty((n_features, self.n_bins), dtype=np.float32)

        for f in range(n_features):
            col = feature_matrix[:, f].astype(np.float64)
            c   = np.quantile(col, q_levels).astype(np.float32)
            centers[f] = c

            # Width = sigma_scale × half the inter-center spacing.
            # Edge bins use the spacing of the nearest interior pair so that
            # the first and last neurons are not artificially narrow.
            spacing     = np.diff(c.astype(np.float64))
            if len(spacing) == 0:
                spacing = np.array([1e-6])
            left_w      = np.concatenate([[spacing[0]],   spacing])
            right_w     = np.concatenate([spacing,        [spacing[-1]]])
            w           = 0.5 * self.sigma_scale * (left_w + right_w) / 2.0
            w           = np.maximum(w, 1e-6).astype(np.float32)
            widths[f]   = w

        self.centers_ = centers
        self.widths_  = widths
        self.fw_ = (
            np.array(feature_weights, dtype=np.float32)
            if feature_weights is not None
            else np.ones(n_features, dtype=np.float32)
        )
        return self

    # ── Scoring ──────────────────────────────────────────────────────────── #

    def score_snippet(self, feature_vector: np.ndarray) -> float:
        """Score a single snippet.  feature_vector: float32[n_features]."""
        self._assert_fitted()
        fv = np.asarray(feature_vector, dtype=np.float32)
        centers = self.centers_
        widths = self.widths_
        fw = self.fw_
        assert centers is not None and widths is not None and fw is not None
        return float(_score_snippet_jit(fv, centers, widths, fw))

    def score_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Score N snippets.

        Parameters
        ----------
        feature_matrix : float32[N, n_features]

        Returns
        -------
        scores : float32[N]  — discriminant scores in [0, 1]
        """
        self._assert_fitted()
        fvs = np.ascontiguousarray(feature_matrix, dtype=np.float32)
        centers = self.centers_
        widths = self.widths_
        fw = self.fw_
        assert centers is not None and widths is not None and fw is not None
        out = np.empty(fvs.shape[0], dtype=np.float32)
        _batch_score_jit(fvs, centers, widths, fw, out)
        return out

    def classify(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return bool mask: True = real spike, False = noise/fail."""
        scores = self.score_batch(feature_matrix)
        return scores >= np.float32(self.threshold)

    def project_snippet(self, feature_vector: np.ndarray) -> np.ndarray:
        """Return afferent activations with shape [n_features, n_bins]."""
        self._assert_fitted()
        fv = np.asarray(feature_vector, dtype=np.float32)
        centers = self.centers_
        widths = self.widths_
        assert centers is not None and widths is not None
        out = np.empty((fv.shape[0], centers.shape[1]), dtype=np.float32)
        _project_snippet_jit(fv, centers, widths, out)
        return out

    def project_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return afferent activations with shape [N, n_features, n_bins]."""
        self._assert_fitted()
        fvs = np.ascontiguousarray(feature_matrix, dtype=np.float32)
        centers = self.centers_
        widths = self.widths_
        assert centers is not None and widths is not None
        out = np.empty((fvs.shape[0], fvs.shape[1], centers.shape[1]), dtype=np.float32)
        _batch_project_jit(fvs, centers, widths, out)
        return out

    # ── JIT warmup ───────────────────────────────────────────────────────── #

    def warmup(self) -> None:
        """Pre-compile Numba kernels with a zero-cost dummy call.

        Call once at startup *before* any timing measurements.
        After warmup, JIT overhead is eliminated from subsequent calls.
        """
        self._assert_fitted()
        centers = self.centers_
        widths = self.widths_
        fw = self.fw_
        assert centers is not None and widths is not None and fw is not None
        n_f = centers.shape[0]
        dummy_fv  = np.zeros(n_f, dtype=np.float32)
        dummy_fvs = np.zeros((2, n_f), dtype=np.float32)
        dummy_out = np.empty(2, dtype=np.float32)
        dummy_proj = np.empty((n_f, centers.shape[1]), dtype=np.float32)
        dummy_proj_batch = np.empty((2, n_f, centers.shape[1]), dtype=np.float32)
        _score_snippet_jit(dummy_fv, centers, widths, fw)
        _batch_score_jit(dummy_fvs, centers, widths, fw, dummy_out)
        _project_snippet_jit(dummy_fv, centers, widths, dummy_proj)
        _batch_project_jit(dummy_fvs, centers, widths, dummy_proj_batch)

    # ── Serialisation ────────────────────────────────────────────────────── #

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (human-readable)."""
        self._assert_fitted()
        quantile_levels = self.quantile_levels_
        centers = self.centers_
        widths = self.widths_
        fw = self.fw_
        assert quantile_levels is not None and centers is not None
        assert widths is not None and fw is not None
        return {
            "n_bins":           self.n_bins,
            "sigma_scale":      self.sigma_scale,
            "threshold":        self.threshold,
            "feature_names":    self.feature_names,
            "quantile_levels":  quantile_levels.tolist(),
            "centers":          centers.tolist(),
            "widths":           widths.tolist(),
            "feature_weights":  fw.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightBank":
        """Reconstruct WeightBank from a dict (e.g. loaded from JSON)."""
        wb = cls(
            n_bins=d["n_bins"],
            sigma_scale=d["sigma_scale"],
            threshold=d["threshold"],
        )
        wb.feature_names    = d["feature_names"]
        wb.quantile_levels_ = np.array(d["quantile_levels"], dtype=np.float64)
        wb.centers_         = np.array(d["centers"],         dtype=np.float32)
        wb.widths_          = np.array(d["widths"],          dtype=np.float32)
        wb.fw_              = np.array(d["feature_weights"], dtype=np.float32)
        return wb

    def save(self, path: str | Path) -> None:
        """Save WeightBank configuration to JSON (human-inspectable)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "WeightBank":
        """Load WeightBank from JSON file."""
        with open(path, "r") as fh:
            return cls.from_dict(json.load(fh))

    def describe(self) -> str:
        """Return a human-readable summary of the weight bank."""
        centers = self.centers_
        widths = self.widths_
        if centers is None or widths is None:
            return "WeightBank(unfitted)"
        lines = [
            f"WeightBank  n_bins={self.n_bins}  σ_scale={self.sigma_scale}"
            f"  threshold={self.threshold}",
            f"  {'Feature':<30} {'Bin-min':>10} {'Bin-max':>10}"
            f"  {'Avg width':>10}",
            "  " + "-" * 64,
        ]
        for f, name in enumerate(self.feature_names):
            c_min = float(centers[f, 0])
            c_max = float(centers[f, -1])
            w_avg = float(widths[f].mean())
            lines.append(f"  {name:<30} {c_min:>10.4f} {c_max:>10.4f}"
                         f"  {w_avg:>10.4f}")
        return "\n".join(lines)

    # ── Internal ─────────────────────────────────────────────────────────── #

    def _assert_fitted(self) -> None:
        if self.centers_ is None:
            raise RuntimeError(
                "WeightBank must be fitted before scoring.  Call .fit() first."
            )
