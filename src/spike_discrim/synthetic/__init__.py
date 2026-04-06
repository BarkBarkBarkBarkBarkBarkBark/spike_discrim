"""Synthetic subpackage."""
from spike_discrim.synthetic.procedural_generator import (
    make_biphasic_waveform,
    make_noise_waveform,
    generate_dataset as generate_procedural_dataset,
)

try:
    from spike_discrim.synthetic.lfpy_generator import (
        generate_dataset as generate_lfpy_dataset,
        LFPY_AVAILABLE,
    )
except Exception:
    LFPY_AVAILABLE = False  # type: ignore[assignment]

    def generate_lfpy_dataset(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "LFPy/NEURON not available.  Install with: pip install -e '.[lfpy]'"
        )


__all__ = [
    "make_biphasic_waveform",
    "make_noise_waveform",
    "generate_procedural_dataset",
    "generate_lfpy_dataset",
    "LFPY_AVAILABLE",
]
