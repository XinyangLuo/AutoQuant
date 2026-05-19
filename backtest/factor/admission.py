#!/usr/bin/env python3
"""Factor admission: promote a factor variant from the work DB into the stable library.

Pipeline split (post-redesign):

  ``backfill``  → write factor variants(每个声明的 ``(industry, cap)`` 组合)to the **work** DB.
  ``evaluation``→ read work DB at one variant, print metrics + reference thresholds (no gating).
  ``run_factor_pipeline.py`` → kick off factor eval + simple + detailed BT.
  ``admit``     → after a human looks at all three reports, run this once
                 for the chosen ``(factor_id, variant)``.
                 Moves that variant's data from work → library, marks
                 ``variant_status[variant]='admitted'`` in the registry.
                 Other variants are unaffected.
  ``reject``    → opposite path: clears the variant's rows from work,
                 marks ``variant_status[variant]='rejected'``.

每个 variant 都是独立的 admission 单位。同一因子的 ``raw`` 可以被 admit、
``swl2_capq5`` 可以被 reject,反之亦然 —— 它们是因子的不同"形态"。

The work DB is the temporary research playground. The library DB
(``factor_library.duckdb``) holds the stabilised factor variants and is the only
source consulted by the cross-factor correlation check during evaluation.

All status / history records live inside ``registry.json``.

CLI:
    python -m backtest.factor.admission admit  f_001                       # default variant
    python -m backtest.factor.admission admit  f_001 --variant swl2_capq5
    python -m backtest.factor.admission reject f_002 --variant raw
    python -m backtest.factor.admission status                              # list all
    python -m backtest.factor.admission status f_001                        # show one factor
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from backtest.factor.registry import (
    _load_registry,
    _save_registry,
    get_factor_variants,
    get_registry,
    sync_registry,
)
from backtest.factor.storage import FactorLibrary, FactorStorage
from backtest.factor.variants import BASELINE_VARIANT, canonicalize_variant

# Reference thresholds — purely informational. ``evaluation`` may print
# "passes / does not pass" relative to these to help a human decide, but
# nothing in admission gates against them. Adjust freely.
RECOMMENDED_THRESHOLDS: dict[str, float | int | str | bool] = {
    "min_rankicir": 0.25,
    "min_ic_positive_ratio": 0.52,
    "max_turnover": 0.5,
    "max_corr": 0.85,
    "primary_horizon": 20,
    "ret_type": "open",
    "exclude_limit_up": True,
}

STATUS_ADMITTED = "admitted"
STATUS_REJECTED = "rejected"
STATUS_PENDING = "pending"
_VALID_STATUSES = frozenset({STATUS_ADMITTED, STATUS_REJECTED})

Status = Literal["admitted", "rejected"]


@dataclass
class AdmissionAction:
    """Outcome of an ``admit()`` / ``reject()`` call."""

    factor_id: str
    variant: str
    action: Status
    rows_promoted: int
    rows_cleared: int
    timestamp: str

    def __repr__(self) -> str:
        return (
            f"AdmissionAction({self.factor_id}/{self.variant}, "
            f"{self.action.upper()}, "
            f"promoted={self.rows_promoted}, cleared={self.rows_cleared})"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_variant_state(meta: dict) -> None:
    """In-place migrate a registry entry to variant-aware state if needed.

    Old shape: ``meta["status"] = "admitted" | "rejected"``.
    New shape: ``meta["variant_status"] = {variant: status, ...}``.

    Migration applies the legacy single status to all currently declared
    variants. Legacy ``admission`` / ``admission_history`` are also folded
    into ``variant_admission_history[variant]`` for each declared variant.
    """
    if "variant_status" in meta:
        meta.setdefault("variant_admission_history", {})
        return

    declared = [
        d.get("industry") for d in meta.get("neutralizations", [])
    ]  # just to detect non-empty; real list comes from registry helper
    # We don't have factor_id here, so we'll fall back to the meta's own
    # neutralizations list expansion.
    from backtest.factor.variants import expand_variant_names

    variants = expand_variant_names(meta.get("neutralizations"))
    legacy_status = meta.get("status")
    legacy_admission = meta.get("admission")
    legacy_history = meta.get("admission_history", [])

    meta["variant_status"] = {}
    meta["variant_admission_history"] = {}
    if legacy_status in _VALID_STATUSES and variants:
        for v in variants:
            meta["variant_status"][v] = legacy_status
            if legacy_history:
                meta["variant_admission_history"][v] = list(legacy_history)


def _finalize_action(
    factor_id: str,
    variant: str,
    status: Status,
    *,
    rows_promoted: int,
    rows_cleared: int,
    notes: str | None,
    registry: dict,
    persist: bool,
    strategy_config: dict | None = None,
) -> AdmissionAction:
    """Stamp the (factor_id, variant) action onto the registry."""
    entry = {
        "variant": variant,
        "action": status,
        "rows_promoted": rows_promoted,
        "rows_cleared": rows_cleared,
        "timestamp": _now_iso(),
        "notes": notes,
    }
    if strategy_config is not None:
        entry["strategy_config"] = strategy_config

    meta = registry[factor_id]
    _ensure_variant_state(meta)
    meta["variant_status"][variant] = status
    history = meta["variant_admission_history"].setdefault(variant, [])
    history.append(entry)
    del history[:-20]

    # Maintain a derived top-level summary for human reading
    # (admitted = all variants admitted; rejected = all variants rejected;
    # pending = anything else, including mixed).
    statuses = set(meta["variant_status"].values())
    if statuses == {STATUS_ADMITTED}:
        meta["status"] = STATUS_ADMITTED
    elif statuses == {STATUS_REJECTED}:
        meta["status"] = STATUS_REJECTED
    elif statuses & _VALID_STATUSES:
        meta["status"] = "mixed"
    else:
        meta["status"] = STATUS_PENDING
    meta["admission"] = entry  # last action across any variant

    if persist:
        _save_registry(registry)
        sync_registry()

    return AdmissionAction(
        factor_id=factor_id,
        variant=variant,
        action=status,
        rows_promoted=rows_promoted,
        rows_cleared=rows_cleared,
        timestamp=entry["timestamp"],
    )


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------


def admit(
    factor_id: str,
    *,
    variant: str = BASELINE_VARIANT,
    notes: str | None = None,
    registry: dict | None = None,
    strategy_config: dict | None = None,
) -> AdmissionAction:
    """Promote one variant of a factor from the work DB into the stable library.

    Steps:
      1. Read ``(factor_id, variant)`` rows from ``FactorStorage`` (work DB).
      2. Upsert into ``FactorLibrary`` (library DB).
      3. Delete those rows from the work DB (other variants untouched).
      4. Mark ``registry[factor_id]["variant_status"][variant] = "admitted"``.

    Parameters
    ----------
    factor_id : str
        Factor to admit. Must already exist in ``registry.json``.
    variant : str
        Which neutralization variant to admit (default ``swl2_capq5``).
    notes : str, optional
        Free-form note attached to the admission history entry.
    registry : dict, optional
        In-memory registry dict to mutate (caller persists). When ``None``,
        load / save ``registry.json`` directly.
    strategy_config : dict, optional
        Snapshot of the strategy config used to validate this variant
        (top_n / top_pct / rebalance / decay / direction / index_members / ...).
        Stored verbatim in the admission history entry for reproducibility.

    Raises
    ------
    KeyError
        If ``factor_id`` is not registered.
    ValueError
        If the work DB has no data for this (factor_id, variant).
    """
    variant = canonicalize_variant(variant)
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    with FactorStorage() as work, FactorLibrary() as lib:
        rows_promoted = lib.promote_from_work(factor_id, work, variant=variant)
        if rows_promoted == 0:
            raise ValueError(
                f"No data in work DB for {factor_id}/{variant}. "
                f"Did you run `python -m backtest.factor.backfill {factor_id}` first?"
            )
        rows_cleared = work.delete_factor(factor_id, variant=variant)

    return _finalize_action(
        factor_id, variant, STATUS_ADMITTED,
        rows_promoted=rows_promoted, rows_cleared=rows_cleared,
        notes=notes, registry=target, persist=persist,
        strategy_config=strategy_config,
    )


def reject(
    factor_id: str,
    *,
    variant: str = BASELINE_VARIANT,
    notes: str | None = None,
    registry: dict | None = None,
    strategy_config: dict | None = None,
) -> AdmissionAction:
    """Discard one variant of a factor — clears its work rows, marks
    ``variant_status[variant]='rejected'``. Other variants untouched.

    The library DB is **not** touched. If the (factor, variant) was previously
    admitted, this raises ``ValueError``.

    See :func:`admit` for the ``strategy_config`` parameter semantics.
    """
    variant = canonicalize_variant(variant)
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    _ensure_variant_state(target[factor_id])
    if target[factor_id]["variant_status"].get(variant) == STATUS_ADMITTED:
        raise ValueError(
            f"{factor_id}/{variant} is already admitted. De-admission is not supported "
            f"via this CLI — remove from library manually if you must."
        )

    with FactorStorage() as work:
        rows_cleared = work.delete_factor(factor_id, variant=variant)

    return _finalize_action(
        factor_id, variant, STATUS_REJECTED,
        rows_promoted=0, rows_cleared=rows_cleared,
        notes=notes, registry=target, persist=persist,
        strategy_config=strategy_config,
    )


def _discover_strategy_config(
    factor_id: str,
    variant: str,
    *,
    results_root: str | Path = "results",
    tag: str | None = None,
) -> dict | None:
    """读 ``results/<factor_id>/<variant>/<tag>/pipeline.json`` 中的 strategy_config 块。

    未指定 ``tag`` 且存在多个 tag 子目录时抛 ``ValueError``;variant 目录或
    pipeline.json 不存在则返回 None,允许用户跳过 pipeline 直接手动 admit。
    """
    variant_dir = Path(results_root) / factor_id / variant
    if not variant_dir.exists():
        return None

    if tag is not None:
        target = variant_dir / tag / "pipeline.json"
        if not target.exists():
            raise FileNotFoundError(
                f"pipeline.json not found at {target}. Available tags: "
                f"{sorted(p.name for p in variant_dir.iterdir() if p.is_dir() and p.name != 'factor_eval')}"
            )
        return _load_pipeline_strategy_config(target)

    candidates = sorted(
        p for p in variant_dir.iterdir()
        if p.is_dir() and p.name != "factor_eval" and (p / "pipeline.json").exists()
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        names = [p.name for p in candidates]
        raise ValueError(
            f"Found multiple pipeline.json under {variant_dir}: {names}. "
            f"Pass --tag <tag> to pick one (or --no-strategy-config to skip)."
        )
    return _load_pipeline_strategy_config(candidates[0] / "pipeline.json")


def _load_pipeline_strategy_config(path: Path) -> dict | None:
    """Read ``pipeline.json``,返回 ``strategy_config`` 块,缺失返回 None。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload.get("strategy_config")
    if not isinstance(cfg, dict):
        return None
    return cfg


# ---------------------------------------------------------------------------
# Registry queries
# ---------------------------------------------------------------------------


def _variant_status_map(meta: dict) -> dict[str, str]:
    """Return ``{variant: status}`` for a factor meta, with on-the-fly migration."""
    _ensure_variant_state(meta)
    return dict(meta["variant_status"])


def get_factor_variant_statuses(factor_id: str) -> dict[str, str]:
    """Return ``{variant: status}`` for a single factor.

    Variants without an explicit status are reported as ``"pending"``.
    """
    meta = get_registry()[factor_id]
    declared = get_factor_variants(factor_id)
    explicit = _variant_status_map(meta)
    return {v: explicit.get(v, STATUS_PENDING) for v in declared}


def get_admitted_factor_variants() -> list[tuple[str, str]]:
    """Return list of ``(factor_id, variant)`` whose status is admitted."""
    out: list[tuple[str, str]] = []
    for fid, meta in get_registry().items():
        for v, st in _variant_status_map(meta).items():
            if st == STATUS_ADMITTED:
                out.append((fid, v))
    return out


def get_rejected_factor_variants() -> list[tuple[str, str]]:
    """Return list of ``(factor_id, variant)`` whose status is rejected."""
    out: list[tuple[str, str]] = []
    for fid, meta in get_registry().items():
        for v, st in _variant_status_map(meta).items():
            if st == STATUS_REJECTED:
                out.append((fid, v))
    return out


def get_admitted_factor_ids() -> list[str]:
    """Factor IDs with at least one admitted variant (compat shim for callers)."""
    seen: set[str] = set()
    for fid, _ in get_admitted_factor_variants():
        seen.add(fid)
    return sorted(seen)


def get_rejected_factor_ids() -> list[str]:
    """Factor IDs whose every declared variant is rejected (compat shim)."""
    rejected: list[str] = []
    for fid, meta in get_registry().items():
        declared = get_factor_variants(fid)
        statuses = _variant_status_map(meta)
        if declared and all(statuses.get(v) == STATUS_REJECTED for v in declared):
            rejected.append(fid)
    return rejected


def get_pending_factor_ids() -> list[str]:
    """Factor IDs with **any** declared variant still pending."""
    pending: list[str] = []
    for fid, meta in get_registry().items():
        declared = get_factor_variants(fid)
        if not declared:
            pending.append(fid)
            continue
        statuses = _variant_status_map(meta)
        if any(statuses.get(v) not in _VALID_STATUSES for v in declared):
            pending.append(fid)
    return pending


# ---------------------------------------------------------------------------
# Threshold reference helper (purely informational)
# ---------------------------------------------------------------------------


def check_recommended_thresholds(
    metrics: dict,
    config: dict | None = None,
) -> dict[str, bool]:
    """Compare a factor's metrics against the reference thresholds.

    Returns a dict of ``{check_name: passed_bool}``. **Does not gate
    anything** — purely a convenience for evaluation output and Agent
    decision-making.

    Expected keys in ``metrics`` (typically taken from
    :class:`~backtest.factor.evaluation.EvaluationResult`):

    - ``rankicir``  — RankICIR at primary_horizon
    - ``ic_positive_ratio`` — positive-IC day ratio at primary_horizon
    - ``turnover`` — factor rank turnover
    - ``max_corr`` — max ``|corr|`` against library factors
    """
    cfg = {**RECOMMENDED_THRESHOLDS, **(config or {})}
    return {
        "rankicir": metrics.get("rankicir", float("-inf")) >= cfg["min_rankicir"],
        "ic_positive_ratio": metrics.get("ic_positive_ratio", 0.0)
            >= cfg["min_ic_positive_ratio"],
        "turnover": metrics.get("turnover", float("inf")) < cfg["max_turnover"],
        "max_corr": metrics.get("max_corr", float("inf")) < cfg["max_corr"],
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_action(action: AdmissionAction) -> None:
    """Pretty-print the outcome of admit() / reject()."""
    print(f"\n{'=' * 60}")
    print(f"Admission: {action.factor_id}/{action.variant}  ->  {action.action.upper()}")
    print(f"{'=' * 60}")
    print(f"  rows promoted to library: {action.rows_promoted:,}")
    print(f"  rows cleared from work  : {action.rows_cleared:,}")
    print(f"  timestamp              : {action.timestamp}")
    print(f"{'=' * 60}\n")


def print_status(factor_id: str | None = None) -> None:
    """Show admission status broken down by (factor_id, variant)."""
    registry = get_registry()
    if factor_id is not None:
        if factor_id not in registry:
            print(f"factor_id '{factor_id}' not found")
            return
        rows = [(factor_id, registry[factor_id])]
    else:
        rows = sorted(registry.items())

    if not rows:
        print("(no factors registered)")
        return

    print(f"\n{'factor_id':<14}{'variant':<14}{'status':<12}{'name'}")
    print("-" * 78)
    for fid, meta in rows:
        statuses = _variant_status_map(meta)
        declared = get_factor_variants(fid) or sorted(statuses.keys())
        for v in declared:
            st = statuses.get(v, STATUS_PENDING)
            print(f"{fid:<14}{v:<14}{st:<12}{meta.get('name', '')}")
        if not declared:
            print(f"{fid:<14}{'(no variants)':<14}{'-':<12}{meta.get('name', '')}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Factor admission")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_meta_flags(sp):
        sp.add_argument("--tag", default=None,
                        help="results/<fid>/<variant>/<tag>/ 下的 tag。指定则 admit/reject "
                             "时读对应 pipeline.json 的 strategy_config 写入 history。")
        sp.add_argument("--results-root", default="results",
                        help="results 根目录,默认 'results'。")
        sp.add_argument("--no-strategy-config", action="store_true",
                        help="不读 pipeline.json,history 不写 strategy_config 块。")

    p_admit = sub.add_parser("admit", help="Promote factor variant from work DB to library DB")
    p_admit.add_argument("factor_id")
    p_admit.add_argument("--variant", default=BASELINE_VARIANT,
                         help=f"Variant to admit (default: {BASELINE_VARIANT})")
    p_admit.add_argument("--notes", default=None, help="Free-form note for history")
    _add_meta_flags(p_admit)

    p_reject = sub.add_parser("reject", help="Discard factor variant — clear work DB, mark rejected")
    p_reject.add_argument("factor_id")
    p_reject.add_argument("--variant", default=BASELINE_VARIANT,
                          help=f"Variant to reject (default: {BASELINE_VARIANT})")
    p_reject.add_argument("--notes", default=None, help="Free-form note for history")
    _add_meta_flags(p_reject)

    p_status = sub.add_parser("status", help="Show admission status broken down by variant")
    p_status.add_argument("factor_id", nargs="?", default=None)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd in ("admit", "reject"):
        strategy_config: dict | None = None
        if not args.no_strategy_config:
            try:
                strategy_config = _discover_strategy_config(
                    args.factor_id, args.variant,
                    results_root=args.results_root, tag=args.tag,
                )
            except (ValueError, FileNotFoundError) as exc:
                parser.exit(2, f"{args.cmd} failed: {exc}\n")
        try:
            fn = admit if args.cmd == "admit" else reject
            action = fn(
                args.factor_id, variant=args.variant, notes=args.notes,
                strategy_config=strategy_config,
            )
        except (KeyError, ValueError) as exc:
            parser.exit(2, f"{args.cmd} failed: {exc}\n")
        print_action(action)
        if strategy_config:
            print(f"  strategy_config recorded ({len(strategy_config)} fields)\n")
        return

    if args.cmd == "status":
        print_status(args.factor_id)
        return


if __name__ == "__main__":
    main()
