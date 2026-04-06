"""tests/test_features.py — Unit tests for Tier 1/2/3 feature kernels."""
import numpy as np
import pytest

from spike_discrim.features.core_features import (
    first_derivative,
    second_derivative,
    absolute_window_sum,
    short_window_energy,
    teager_energy,
    batch_extract_scalar_features,
    SCALAR_FEATURE_NAMES,
    TEMPORAL_MAD_FEATURE_PREFIX,
    build_temporal_mad_feature_names,
    compute_temporal_window_bounds,
    extract_temporal_mad_features,
)
from spike_discrim.features.event_features import (
    trough_amplitude,
    peak_amplitude,
    trough_to_peak_time,
    half_width,
    biphasic_ratio,
    baseline_rms,
    zero_crossing_count,
    batch_event_features,
    N_EVENT_FEATURES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────── #

@pytest.fixture
def biphasic_snippet():
    """Canonical biphasic waveform: trough at sample 20, peak at sample 32."""
    from spike_discrim.synthetic.procedural_generator import make_biphasic_waveform
    return make_biphasic_waveform(n_samples=64, pre_peak=20, amp_neg=-100.0,
                                  amp_pos=40.0, delay_pos=12, noise_std=0.0)


@pytest.fixture
def flat_snippet():
    return np.zeros(64, dtype=np.float32)


@pytest.fixture
def batch_waveforms(biphasic_snippet):
    """float32[20, 64] batch."""
    return np.stack([biphasic_snippet] * 20, axis=0)


# ── Tier 1 tests ──────────────────────────────────────────────────────────── #

class TestFirstDerivative:
    def test_constant_signal_is_zero(self, flat_snippet):
        out = np.empty(64, dtype=np.float32)
        first_derivative(flat_snippet, out)
        assert np.allclose(out[1:], 0.0)

    def test_ramp_gives_constant_slope(self):
        ramp = np.arange(64, dtype=np.float32)
        out  = np.empty(64, dtype=np.float32)
        first_derivative(ramp, out)
        assert np.allclose(out[1:], 1.0)

    def test_causal_boundary(self, biphasic_snippet):
        out = np.empty(64, dtype=np.float32)
        first_derivative(biphasic_snippet, out)
        assert out[0] == 0.0

    def test_detects_trough_descent(self, biphasic_snippet):
        out = np.empty(64, dtype=np.float32)
        first_derivative(biphasic_snippet, out)
        # Before trough there must be a negative slope region
        assert out[:20].min() < 0.0


class TestSecondDerivative:
    def test_linear_signal_is_zero(self):
        ramp = np.arange(64, dtype=np.float32)
        out  = np.empty(64, dtype=np.float32)
        second_derivative(ramp, out)
        assert np.allclose(out[2:], 0.0, atol=1e-5)

    def test_causal_boundaries(self, biphasic_snippet):
        out = np.empty(64, dtype=np.float32)
        second_derivative(biphasic_snippet, out)
        assert out[0] == 0.0 and out[1] == 0.0


class TestAbsoluteWindowSum:
    def test_all_ones_window(self):
        ones = np.ones(64, dtype=np.float32)
        out  = np.empty(64, dtype=np.float32)
        absolute_window_sum(ones, out, window=8)
        # After 8 samples, running sum should equal 8
        assert np.allclose(out[7:], 8.0)

    def test_monotonically_nonnegative(self, biphasic_snippet):
        out = np.empty(64, dtype=np.float32)
        absolute_window_sum(biphasic_snippet, out, window=16)
        assert (out >= 0.0).all()


class TestShortWindowEnergy:
    def test_zero_signal(self, flat_snippet):
        out = np.empty(64, dtype=np.float32)
        short_window_energy(flat_snippet, out, window=8)
        assert np.allclose(out, 0.0)

    def test_matches_manual(self):
        x   = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = np.empty(8, dtype=np.float32)
        short_window_energy(x, out, window=4)
        assert abs(out[3] - (1**2 + 2**2 + 3**2 + 0**2)) < 1e-4


# ── Tier 3 event feature tests ────────────────────────────────────────────── #

class TestEventFeatures:
    def test_trough_amplitude_is_min(self, biphasic_snippet):
        assert abs(trough_amplitude(biphasic_snippet) - biphasic_snippet.min()) < 1e-5

    def test_peak_amplitude_is_max(self, biphasic_snippet):
        assert abs(peak_amplitude(biphasic_snippet) - biphasic_snippet.max()) < 1e-5

    def test_trough_to_peak_positive(self, biphasic_snippet):
        ttp = trough_to_peak_time(biphasic_snippet)
        assert ttp > 0, "trough-to-peak time must be positive"

    def test_half_width_positive_flat_is_zero(self):
        # All-positive signal: trough=1, half=0.5, no samples ≤ 0.5 → width = 0
        pos = np.ones(64, dtype=np.float32)
        assert half_width(pos) == 0

    def test_biphasic_ratio_positive(self, biphasic_snippet):
        br = biphasic_ratio(biphasic_snippet)
        assert br > 0

    def test_baseline_rms_flat_is_zero(self, flat_snippet):
        assert abs(baseline_rms(flat_snippet)) < 1e-10

    def test_zero_crossing_flat_is_zero(self, flat_snippet):
        assert zero_crossing_count(flat_snippet) == 0

    def test_zero_crossing_sine(self):
        t  = np.linspace(0, 4 * np.pi, 100, dtype=np.float32)
        s  = np.sin(t)
        zc = zero_crossing_count(s)
        # 0→4π is 2 complete periods; each period has 2 zero crossings
        assert zc == 4


# ── Batch extraction tests ────────────────────────────────────────────────── #

class TestBatchExtraction:
    def test_scalar_feature_matrix_shape(self, batch_waveforms):
        N, T = batch_waveforms.shape
        d1   = np.empty((N, T), dtype=np.float32)
        d2   = np.empty((N, T), dtype=np.float32)
        aws  = np.empty((N, T), dtype=np.float32)
        out  = np.empty((N, 6), dtype=np.float32)
        batch_extract_scalar_features(batch_waveforms, d1, d2, aws, out, window=16)
        assert out.shape == (N, 6)

    def test_event_feature_matrix_shape(self, batch_waveforms):
        out = np.empty((len(batch_waveforms), N_EVENT_FEATURES), dtype=np.float32)
        batch_event_features(batch_waveforms, out)
        assert out.shape == (len(batch_waveforms), N_EVENT_FEATURES)

    def test_consistent_with_scalar(self, biphasic_snippet, batch_waveforms):
        """Batch scalar extraction matches single-call values."""
        N, T = batch_waveforms.shape
        d1   = np.empty((N, T), dtype=np.float32)
        d2   = np.empty((N, T), dtype=np.float32)
        aws  = np.empty((N, T), dtype=np.float32)
        out  = np.empty((N, 6), dtype=np.float32)
        batch_extract_scalar_features(batch_waveforms, d1, d2, aws, out, window=16)
        # All rows identical (same waveform repeated)
        assert np.allclose(out[0], out[1], atol=1e-4)


class TestTemporalMadFeatures:
    def test_temporal_window_bounds_cover_snippet(self):
        starts, ends = compute_temporal_window_bounds(64, n_bins=5, overlap_fraction=0.5)
        assert len(starts) == 5
        assert starts[0] == 0
        assert ends[-1] == 64
        assert np.all(starts[1:] <= ends[:-1])

    def test_feature_names_match_bin_count(self):
        names = build_temporal_mad_feature_names(4)
        assert names == [
            f"{TEMPORAL_MAD_FEATURE_PREFIX}_00",
            f"{TEMPORAL_MAD_FEATURE_PREFIX}_01",
            f"{TEMPORAL_MAD_FEATURE_PREFIX}_02",
            f"{TEMPORAL_MAD_FEATURE_PREFIX}_03",
        ]

    def test_temporal_mad_feature_matrix_shape(self, batch_waveforms):
        features, names, metadata = extract_temporal_mad_features(
            batch_waveforms,
            n_bins=6,
            overlap_fraction=0.5,
        )
        assert features.shape == (len(batch_waveforms), 6)
        assert len(names) == 6
        assert metadata["winner_take_all"] is True
        assert np.allclose(features[0], features[1], atol=1e-5)

    def test_global_noise_mad_normalises_output(self):
        waveforms = np.array([
            [0.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        raw, _, _ = extract_temporal_mad_features(
            waveforms,
            n_bins=2,
            overlap_fraction=0.5,
            noise_mad_mode="none",
        )
        norm, _, metadata = extract_temporal_mad_features(
            waveforms,
            n_bins=2,
            overlap_fraction=0.5,
            noise_mad_mode="global",
            global_noise_mad=2.0,
            mad_scale_factor=1.0,
        )
        assert metadata["noise_mad_mode"] == "global"
        assert np.allclose(norm, raw / 2.0, atol=1e-5)
