"""Profiling subpackage — op counters and wall-clock timers."""
from spike_discrim.profiling.op_counter import (
    FEATURE_OP_COUNTS,
    ProfileResult,
    timer,
    profile_feature,
    profile_all_features,
)

__all__ = [
    "FEATURE_OP_COUNTS",
    "ProfileResult",
    "timer",
    "profile_feature",
    "profile_all_features",
]
