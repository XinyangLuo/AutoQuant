#!/usr/bin/env python3
"""Factor admission: promote a factor from the work DB into the stable library.

Pipeline split (post-redesign):

  ``backfill``  → write factor values to the **work** DB.
  ``evaluation``→ read work DB, print metrics + reference thresholds (no gating).
  ``run_factor_pipeline.py`` → kick off factor eval + simple + detailed BT.
  ``admit``     → after a human looks at all three reports, run this once.
                 Moves the factor's data from work → library, clears work,
                 and stamps ``status=admitted`` in the registry.
  ``reject``    → opposite path: clears work without writing to library,
                 stamps ``status=rejected``.

The work DB is the temporary research playground. The library DB
(``factor_library.duckdb``) holds the stabilised factors and is the only
source consulted by the cross-factor correlation check during evaluation.

All status / history records live inside ``registry.json``.

CLI:
    python -m backtest.factor.admission admit  f_001
    python -m backtest.factor.admission reject f_002
    python -m backtest.factor.admission status                  # list all
    python -m backtest.factor.admission status f_001            # show one
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from backtest.factor.registry import (
    _load_registry,
    _save_registry,
    get_registry,
    sync_registry,
)
from backtest.factor.storage import FactorLibrary, FactorStorage

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
_VALID_STATUSES = frozenset({STATUS_ADMITTED, STATUS_REJECTED})

Status = Literal["admitted", "rejected"]


@dataclass
class AdmissionAction:
    """Outcome of an ``admit()`` / ``reject()`` call."""

    factor_id: str
    action: Status
    rows_promoted: int
    rows_cleared: int
    timestamp: str

    def __repr__(self) -> str:
        return (
            f"AdmissionAction({self.factor_id}, {self.action.upper()}, "
            f"promoted={self.rows_promoted}, cleared={self.rows_cleared})"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finalize_action(
    factor_id: str,
    status: Status,
    *,
    rows_promoted: int,
    rows_cleared: int,
    notes: str | None,
    registry: dict,
    persist: bool,
) -> AdmissionAction:
    """Stamp the action onto the registry and return the AdmissionAction.

    Shared between admit() and reject() so the entry shape, history cap, and
    persistence path stay in one place.
    """
    entry = {
        "action": status,
        "rows_promoted": rows_promoted,
        "rows_cleared": rows_cleared,
        "timestamp": _now_iso(),
        "notes": notes,
    }

    meta = registry[factor_id]
    meta["status"] = status
    meta["admission"] = entry
    history = meta.setdefault("admission_history", [])
    history.append(entry)
    del history[:-20]

    if persist:
        _save_registry(registry)
        sync_registry()

    return AdmissionAction(
        factor_id=factor_id,
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
    notes: str | None = None,
    registry: dict | None = None,
) -> AdmissionAction:
    """Promote a factor from the work DB into the stable library.

    Steps:
      1. Read the factor's full data from ``FactorStorage`` (work DB).
      2. Upsert into ``FactorLibrary`` (library DB).
      3. Delete the factor's rows from the work DB.
      4. Mark ``registry[factor_id]["status"] = "admitted"`` and append to
         ``admission_history``.

    Parameters
    ----------
    factor_id : str
        Factor to admit. Must already exist in ``registry.json`` and have
        data in the work DB.
    notes : str, optional
        Free-form note attached to the admission history entry (e.g.,
        "Sharpe 1.45 detailed BT, IR 0.92 vs 000300").
    registry : dict, optional
        In-memory registry dict to mutate (caller persists). When ``None``,
        load / save ``registry.json`` directly.

    Raises
    ------
    KeyError
        If ``factor_id`` is not registered.
    ValueError
        If the work DB has no data for this factor.
    """
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    with FactorStorage() as work, FactorLibrary() as lib:
        rows_promoted = lib.promote_from_work(factor_id, work)
        if rows_promoted == 0:
            raise ValueError(
                f"No data in work DB for {factor_id}. "
                f"Did you run `python -m backtest.factor.backfill {factor_id}` first?"
            )
        rows_cleared = work.delete_factor(factor_id)

    return _finalize_action(
        factor_id, STATUS_ADMITTED,
        rows_promoted=rows_promoted, rows_cleared=rows_cleared,
        notes=notes, registry=target, persist=persist,
    )


def reject(
    factor_id: str,
    *,
    notes: str | None = None,
    registry: dict | None = None,
) -> AdmissionAction:
    """Discard a factor — clears the work DB, stamps ``status=rejected``.

    The library DB is **not** touched. If the factor was previously admitted
    and is being demoted, this raises ``ValueError`` (deliberate de-admission
    is out of scope and has no API surface yet).
    """
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    if target[factor_id].get("status") == STATUS_ADMITTED:
        raise ValueError(
            f"{factor_id} is already admitted. De-admission is not supported "
            f"via this CLI — remove from library manually if you must."
        )

    with FactorStorage() as work:
        rows_cleared = work.delete_factor(factor_id)

    return _finalize_action(
        factor_id, STATUS_REJECTED,
        rows_promoted=0, rows_cleared=rows_cleared,
        notes=notes, registry=target, persist=persist,
    )


# ---------------------------------------------------------------------------
# Registry queries
# ---------------------------------------------------------------------------


def _get_factor_ids_by_status(status: str | None) -> list[str]:
    """Return factor_ids filtered by status. ``None`` = pending (no status)."""
    registry = get_registry()
    if status is None:
        return [
            fid for fid, meta in registry.items()
            if meta.get("status") not in _VALID_STATUSES
        ]
    return [
        fid for fid, meta in registry.items()
        if meta.get("status") == status
    ]


def get_admitted_factor_ids() -> list[str]:
    """Factor IDs with ``status='admitted'``."""
    return _get_factor_ids_by_status(STATUS_ADMITTED)


def get_rejected_factor_ids() -> list[str]:
    """Factor IDs with ``status='rejected'``."""
    return _get_factor_ids_by_status(STATUS_REJECTED)


def get_pending_factor_ids() -> list[str]:
    """Factor IDs without a status (neither admitted nor rejected)."""
    return _get_factor_ids_by_status(None)


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
    print(f"Admission: {action.factor_id}  ->  {action.action.upper()}")
    print(f"{'=' * 60}")
    print(f"  rows promoted to library: {action.rows_promoted:,}")
    print(f"  rows cleared from work  : {action.rows_cleared:,}")
    print(f"  timestamp              : {action.timestamp}")
    print(f"{'=' * 60}\n")


def print_status(factor_id: str | None = None) -> None:
    """Show admission status for one or all registered factors."""
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

    print(f"\n{'factor_id':<14}{'status':<12}{'last action':<28}{'name'}")
    print("-" * 78)
    for fid, meta in rows:
        status = meta.get("status", "pending")
        last = (meta.get("admission") or {}).get("timestamp", "-")
        print(f"{fid:<14}{status:<12}{last:<28}{meta.get('name', '')}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Factor admission")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_admit = sub.add_parser("admit", help="Promote factor from work DB to library DB")
    p_admit.add_argument("factor_id")
    p_admit.add_argument("--notes", default=None, help="Free-form note for history")

    p_reject = sub.add_parser("reject", help="Discard factor — clear work DB, mark rejected")
    p_reject.add_argument("factor_id")
    p_reject.add_argument("--notes", default=None, help="Free-form note for history")

    p_status = sub.add_parser("status", help="Show admission status for one or all factors")
    p_status.add_argument("factor_id", nargs="?", default=None)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "admit":
        try:
            action = admit(args.factor_id, notes=args.notes)
        except (KeyError, ValueError) as exc:
            parser.exit(2, f"admit failed: {exc}\n")
        print_action(action)
        return

    if args.cmd == "reject":
        try:
            action = reject(args.factor_id, notes=args.notes)
        except (KeyError, ValueError) as exc:
            parser.exit(2, f"reject failed: {exc}\n")
        print_action(action)
        return

    if args.cmd == "status":
        print_status(args.factor_id)
        return


if __name__ == "__main__":
    main()
