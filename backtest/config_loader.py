"""Global configuration loader.

Reads ``config.yaml`` (project root) and caches the result with mtime
invalidation — re-reads only when the file has been modified.  Config
files are small; the mtime check avoids the confusing stale-cache
behaviour while preventing redundant I/O (e.g. ~50 reads triggered by
a single ``StepThresholds()`` construction).
Any module that needs thresholds or knobs imports ``get_config()`` or
``get_section()`` rather than hard-coding values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH: Path | None = None
_CONFIG_MTIME: float = 0.0
_CONFIG_CACHE: dict[str, Any] | None = None


def _find_project_root() -> Path:
    """Walk up from this file until we find ``config.yaml``."""
    here = Path(__file__).resolve().parent
    for p in [here, here.parent, here.parent.parent]:
        candidate = p / "config.yaml"
        if candidate.exists():
            return p
    raise FileNotFoundError(
        "config.yaml not found in project root or parent directories"
    )


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the global YAML configuration.

    Parameters
    ----------
    path : Path | str | None
        Explicit path to a YAML file.  When ``None`` the default
        ``<project_root>/config.yaml`` is used.

    Returns
    -------
    dict
    """
    global _CONFIG_PATH, _CONFIG_MTIME, _CONFIG_CACHE

    if path is None:
        path = _find_project_root() / "config.yaml"
    path = Path(path)

    # Cache with mtime invalidation: re-read only when the file has
    # changed on disk.  This prevents the ~50 redundant reads
    # triggered by StepThresholds() default_factory lambdas while
    # ensuring edits to config.yaml take effect immediately.
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0

    if _CONFIG_CACHE is not None and _CONFIG_PATH == path and _CONFIG_MTIME == mtime:
        return _CONFIG_CACHE.copy()

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML did not parse to dict: {path}")

    _CONFIG_PATH = path
    _CONFIG_MTIME = mtime
    _CONFIG_CACHE = data
    return data.copy()


def get_config() -> dict[str, Any]:
    """Return the full global config (cached with mtime invalidation)."""
    return load_config()


def get_section(*keys: str) -> Any:
    """Drill into the config by nested keys.

    Examples
    --------
    >>> get_section("thresholds", "admission", "min_rankicir")
    0.25

    >>> get_section("pipeline", "default_top_pct")
    0.1
    """
    cfg = get_config()
    for k in keys:
        if not isinstance(cfg, dict):
            raise KeyError(f"Cannot drill {keys!r}: {k!r} is not a dict")
        cfg = cfg[k]
    return cfg


def get_section_or(default: Any, *keys: str) -> Any:
    """Drill into the config by nested keys, returning *default* if missing.

    Examples
    --------
    >>> get_section_or("open", "pipeline", "ret_type")
    "open"   # when config.yaml lacks pipeline.ret_type
    """
    cfg = get_config()
    for k in keys:
        if not isinstance(cfg, dict) or k not in cfg:
            return default
        cfg = cfg[k]
    return cfg


def get_thresholds() -> dict[str, Any]:
    """Return the ``thresholds`` section."""
    return get_section("thresholds")


def get_admission_thresholds() -> dict[str, Any]:
    """Return ``thresholds.admission``."""
    return get_section("thresholds", "admission")


def get_pipeline_thresholds() -> dict[str, Any]:
    """Return ``thresholds.pipeline``."""
    return get_section("thresholds", "pipeline")


def get_agent_thresholds() -> dict[str, Any]:
    """Return ``thresholds.agent``."""
    return get_section("thresholds", "agent")


def load_yaml_file(path: Path | str) -> dict[str, Any]:
    """Load a standalone YAML file (no caching — distinct from ``load_config``).

    Used for per-factor configs that must not pollute the global config cache.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML did not parse to dict: {path}")
    return data
