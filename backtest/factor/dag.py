"""Dependency DAG for residual-admitted factors.

When a factor is admitted in ``residual`` mode, its stored values are
per-date Ridge residuals against all already-admitted factors.  This
creates a dependency chain that backfill and daily update must honour:
dependencies must be computed before the factors that depend on them.

This module provides graph construction and topological sort utilities
used by ``backfill.py`` and ``update.py``.
"""

from __future__ import annotations

from graphlib import TopologicalSorter


def build_dependency_graph(registry: dict) -> dict[str, set[str]]:
    """Build adjacency dict ``{factor_id: {dependency_id, ...}}``.

    Only factors with ``depends_on`` populated and ``admission_mode ==
    'residual'`` contribute edges.  Raw-mode and pending factors have no
    outgoing dependency edges.
    """
    graph: dict[str, set[str]] = {}
    for fid, meta in registry.items():
        deps = meta.get("depends_on")
        if isinstance(deps, list) and deps and meta.get("admission_mode") == "residual":
            graph[fid] = set(deps)
    return graph


def topological_sort(factor_ids: list[str], registry: dict) -> list[str]:
    """Return *factor_ids* in topological order (dependencies first).

    Factors without ``depends_on`` are placed at the front in arbitrary
    order.  The sort is stable for nodes at the same depth when the
    underlying ``TopologicalSorter`` behaves deterministically.

    Raises ``ValueError`` if a dependency cycle is detected.
    """
    graph = build_dependency_graph(registry)

    # Ensure every requested factor_id is a node — factors not in the
    # graph become source nodes (no dependencies).
    for fid in factor_ids:
        graph.setdefault(fid, set())

    ts = TopologicalSorter(graph)
    try:
        ordered = list(ts.static_order())
    except Exception as exc:
        raise ValueError(
            f"Failed to topologically sort factors: {exc}"
        ) from exc

    # static_order returns ALL nodes in the graph, including factors not
    # in the original request.  Filter to only requested IDs.
    requested = set(factor_ids)
    return [fid for fid in ordered if fid in requested]


def get_admission_mode(factor_id: str, registry: dict) -> str | None:
    """Return ``"raw"``, ``"residual"``, or ``None`` if not set."""
    meta = registry.get(factor_id, {})
    mode = meta.get("admission_mode")
    if mode in ("raw", "residual"):
        return mode
    return None


def get_depends_on(factor_id: str, registry: dict) -> list[str]:
    """Return the list of factor IDs this factor is residualized against.

    Empty list for raw-mode or not-admitted factors.
    """
    meta = registry.get(factor_id, {})
    deps = meta.get("depends_on")
    if isinstance(deps, list):
        return deps
    return []
