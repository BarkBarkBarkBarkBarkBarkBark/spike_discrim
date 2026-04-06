"""Config loader — YAML config with merge and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "default.yaml"
_BENCHMARK_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "benchmarks.yaml"


def load_config(path: str | Path = None) -> dict:
    """Load a YAML config file.  Falls back to configs/default.yaml."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_benchmark_config(path: str | Path = None) -> dict:
    """Load benchmark config.  Falls back to configs/benchmarks.yaml."""
    if path is None:
        path = _BENCHMARK_CONFIG_PATH
    return load_config(path)


def merge_configs(*configs: dict) -> dict:
    """Deep-merge multiple config dicts.  Later values override earlier ones."""
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
