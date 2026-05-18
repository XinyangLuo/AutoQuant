"""Factor registry: metadata tracking for named/numbered factors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from backtest.data.tushare_client import _find_project_root


_PROJECT_ROOT = _find_project_root()
_REGISTRY_PATH = _PROJECT_ROOT / "data" / "factor_library" / "registry.json"
_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

# In-memory cache
_REGISTRY_CACHE: dict | None = None
_FACTOR_FUNCTIONS: dict[str, Callable] = {}


def _load_registry() -> dict:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        if _REGISTRY_PATH.exists():
            with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
                _REGISTRY_CACHE = json.load(f)
        else:
            _REGISTRY_CACHE = {}
    return _REGISTRY_CACHE


def _save_registry(registry: dict) -> None:
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = registry
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def register(
    factor_id: str,
    *,
    name: str,
    category: str,
    data_sources: list[str],
    description: str = "",
    parameters: dict | None = None,
):
    """Decorator to register a factor compute function.

    Registration is in-memory only. Call ``sync_registry()`` to persist to disk.
    """

    def decorator(func: Callable):
        registry = _load_registry()

        existing = registry.get(factor_id, {})
        if existing.get("func_name") and existing["func_name"] != func.__name__:
            raise ValueError(
                f"factor_id '{factor_id}' already registered to "
                f"{existing.get('func_name')}"
            )

        # Preserve admission state across re-registration. @register runs every
        # time the module is imported; without this merge, a single import would
        # silently downgrade an admitted/rejected factor back to pending in the
        # in-memory cache.
        preserved = {
            k: existing[k]
            for k in ("status", "admission", "admission_history")
            if k in existing
        }

        registry[factor_id] = {
            "name": name,
            "category": category,
            "data_sources": data_sources,
            "description": description,
            "parameters": parameters or {},
            "func_name": func.__name__,
            "func_module": func.__module__,
            **preserved,
        }
        # In-memory only; disk sync is deferred to avoid race conditions
        _REGISTRY_CACHE = registry

        _FACTOR_FUNCTIONS[factor_id] = func
        func._factor_id = factor_id  # type: ignore[attr-defined]
        return func

    return decorator


def sync_registry() -> None:
    """Persist the in-memory registry to disk."""
    registry = _load_registry()
    _save_registry(registry)


def get_factor_function(factor_id: str) -> Callable:
    """Return the registered compute function for a factor_id."""
    if factor_id in _FACTOR_FUNCTIONS:
        return _FACTOR_FUNCTIONS[factor_id]
    registry = _load_registry()
    meta = registry.get(factor_id)
    if meta is None:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")
    import importlib

    mod = importlib.import_module(meta["func_module"])
    func = getattr(mod, meta["func_name"])
    _FACTOR_FUNCTIONS[factor_id] = func
    return func


def get_registry() -> dict:
    """Return the full registry dict."""
    return _load_registry().copy()


def get_factor_meta(factor_id: str) -> dict:
    """Return metadata for a single factor."""
    registry = _load_registry()
    if factor_id not in registry:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")
    return registry[factor_id].copy()


def list_factors(category: str | None = None) -> list[dict]:
    """List all registered factors, optionally filtered by category."""
    registry = _load_registry()
    result = []
    for factor_id, meta in registry.items():
        if category and meta.get("category") != category:
            continue
        result.append({"factor_id": factor_id, **meta})
    return result


def unregister(factor_id: str) -> None:
    """Remove a factor from the registry (useful for testing)."""
    registry = _load_registry()
    registry.pop(factor_id, None)
    _save_registry(registry)
    _FACTOR_FUNCTIONS.pop(factor_id, None)
