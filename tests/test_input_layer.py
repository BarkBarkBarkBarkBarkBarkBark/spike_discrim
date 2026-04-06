"""tests/test_input_layer.py — Tests for WeightBank fitting, scoring, and serialisation."""
import numpy as np
import pytest
from spike_discrim.input_layer.weights import WeightBank


@pytest.fixture
def calibration_data():
    """200 spikes × 6 features, drawn from a Normal distribution."""
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (200, 6)).astype(np.float32)
    return X


@pytest.fixture
def noise_data():
    """50 noise events that sit outside the calibration distribution."""
    rng = np.random.default_rng(1)
    X = rng.normal(5.0, 1.0, (50, 6)).astype(np.float32)
    return X


@pytest.fixture
def fitted_wb(calibration_data):
    wb = WeightBank(n_bins=10, sigma_scale=1.0, threshold=0.5)
    wb.fit(calibration_data, feature_names=[f"f{i}" for i in range(6)])
    wb.warmup()
    return wb


class TestWeightBankFit:
    def test_centers_shape(self, fitted_wb):
        assert fitted_wb.centers_.shape == (6, 10)

    def test_widths_positive(self, fitted_wb):
        assert (fitted_wb.widths_ > 0).all()

    def test_feature_names_stored(self, fitted_wb):
        assert len(fitted_wb.feature_names) == 6

    def test_unfitted_raises(self):
        wb = WeightBank()
        with pytest.raises(RuntimeError, match="must be fitted"):
            wb.score_snippet(np.zeros(6, dtype=np.float32))


class TestWeightBankScoring:
    def test_calibration_scores_high(self, fitted_wb, calibration_data):
        scores = fitted_wb.score_batch(calibration_data)
        # Median score on calibration data should be well above 0.3
        assert float(np.median(scores)) > 0.3

    def test_ood_scores_lower_than_calibration(self, fitted_wb,
                                                calibration_data, noise_data):
        cal_scores   = fitted_wb.score_batch(calibration_data)
        noise_scores = fitted_wb.score_batch(noise_data)
        assert cal_scores.mean() > noise_scores.mean()

    def test_scores_in_range(self, fitted_wb, calibration_data):
        scores = fitted_wb.score_batch(calibration_data)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_single_vs_batch_consistent(self, fitted_wb, calibration_data):
        single = fitted_wb.score_snippet(calibration_data[0])
        batch  = fitted_wb.score_batch(calibration_data[:1])[0]
        assert abs(single - float(batch)) < 1e-5


class TestWeightBankSerialisation:
    def test_round_trip_json(self, fitted_wb, tmp_path, calibration_data):
        path = tmp_path / "wb.json"
        fitted_wb.save(path)
        wb2 = WeightBank.load(path)
        s1 = fitted_wb.score_batch(calibration_data)
        s2 = wb2.score_batch(calibration_data)
        assert np.allclose(s1, s2, atol=1e-5)

    def test_json_is_human_readable(self, fitted_wb, tmp_path):
        path = tmp_path / "wb.json"
        fitted_wb.save(path)
        import json
        with open(path) as f:
            d = json.load(f)
        assert "centers" in d and "widths" in d and "feature_names" in d

    def test_describe_returns_string(self, fitted_wb):
        desc = fitted_wb.describe()
        assert "WeightBank" in desc and "n_bins" in desc
