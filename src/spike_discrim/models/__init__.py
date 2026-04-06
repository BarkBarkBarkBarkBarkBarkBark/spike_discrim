"""Models subpackage."""
from spike_discrim.models.discriminants import (
    ThresholdDiscriminant,
    WeightBankDiscriminant,
    make_model,
)

__all__ = ["ThresholdDiscriminant", "WeightBankDiscriminant", "make_model"]
