"""Configuration loading.

The pipeline is driven entirely by YAML: ``configs/default.yaml`` holds the
baseline and experiment files override only what they change. Keeping every
parameter in configuration (not in code) is what makes a run reproducible from
its saved ``config.yaml`` alone.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> dict:
    """Set ``cfg['a']['b']['c'] = value`` from the key ``'a.b.c'`` (in place)."""
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return cfg


def load_config(
    experiment: str | Path | None = None,
    overrides: list[str] | None = None,
    base: str | Path = DEFAULT_CONFIG,
) -> dict:
    """Load the base config, apply an experiment file, then ``key=value`` overrides.

    Override values are parsed as YAML, so ``training.epochs=5`` yields an int and
    ``augmentation.enabled=false`` yields a bool.
    """
    with open(base) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("experiment", "default")
    if experiment:
        with open(experiment) as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def config_hash(cfg: dict, length: int = 10) -> str:
    """Stable short hash of a (sub-)config, used for cache keys and run IDs."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
