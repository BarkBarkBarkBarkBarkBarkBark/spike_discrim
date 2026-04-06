"""tests/test_osort_adapter.py — Tests for the osort MATLAB loader."""
import numpy as np
import pytest
import tempfile
from pathlib import Path


def _make_fake_mat_v7(path: Path, n_spikes: int = 50,
                      n_samples: int = 64, n_units: int = 2) -> None:
    """Create a fake MATLAB v7 .mat file with osort-like structure."""
    import scipy.io
    rng = np.random.default_rng(42)
    unit_ids = np.repeat(np.arange(1, n_units + 1), n_spikes // n_units)
    spike_times = np.sort(rng.integers(1000, 100000, len(unit_ids)))
    waveforms   = rng.normal(0, 1, (len(unit_ids), n_samples)).astype(np.float32)
    scipy.io.savemat(str(path), {
        "newSpikeTimes": spike_times.astype(np.float64),
        "waveforms":     waveforms,
        "unitIDs":       unit_ids.astype(np.float64),
    })


@pytest.fixture
def fake_mat_path(tmp_path):
    p = tmp_path / "test_osort.mat"
    _make_fake_mat_v7(p, n_spikes=100, n_samples=64, n_units=3)
    return p


class TestOsortLoader:
    def test_loads_without_error(self, fake_mat_path):
        from spike_discrim.adapters.osort_loader import load_osort_mat
        result = load_osort_mat(fake_mat_path, verbose=False)
        assert "units" in result
        assert "noise" in result

    def test_units_have_waveforms(self, fake_mat_path):
        from spike_discrim.adapters.osort_loader import load_osort_mat
        result = load_osort_mat(fake_mat_path, verbose=False)
        for uid, udata in result["units"].items():
            assert "waveforms" in udata
            assert udata["waveforms"].ndim == 2
            assert udata["waveforms"].dtype == np.float32

    def test_correct_n_units(self, fake_mat_path):
        from spike_discrim.adapters.osort_loader import load_osort_mat
        result = load_osort_mat(fake_mat_path, verbose=False)
        assert len(result["units"]) == 3

    def test_labels_df_has_class_label(self, fake_mat_path):
        from spike_discrim.adapters.osort_loader import load_osort_mat
        result = load_osort_mat(fake_mat_path, verbose=False)
        df = result["labels_df"]
        assert "class_label" in df.columns
        assert set(df["class_label"].unique()).issubset({0, 1})

    def test_saves_npz_and_parquet(self, fake_mat_path, tmp_path):
        from spike_discrim.adapters.osort_loader import load_osort_mat
        out_dir = tmp_path / "canonical"
        load_osort_mat(fake_mat_path, output_dir=out_dir, verbose=False)
        assert (out_dir / "canonical_units.npz").exists()
        assert (out_dir / "labels.parquet").exists()
        assert (out_dir / "session_metadata.json").exists()

    def test_hdf5_detection(self, tmp_path):
        """Non-HDF5 file should not be flagged as HDF5."""
        from spike_discrim.adapters.osort_loader import _is_hdf5
        p = tmp_path / "dummy.mat"
        p.write_bytes(b"\x00" * 16)
        assert not _is_hdf5(p)
