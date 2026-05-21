#!/usr/bin/env python3
"""Factor admission: promote a factor from the work DB into the stable library.

Pipeline:

  ``backfill``  → write factor values to the **work** DB.
  ``evaluation``→ read work DB, print metrics + reference thresholds (no gating).
  ``run_factor_pipeline.py`` → kick off factor eval + simple + detailed BT.
  ``admit``     → after a human looks at all three reports, run this.
                 Moves the factor's data from work → library, marks
                 ``status='admitted'`` in the registry.
  ``reject``    → opposite: clears the factor's column from work,
                 marks ``status='rejected'``.

Each factor has a single neutralization variant (recorded in the registry's
``variant`` field, see :mod:`backtest.factor.variants`). To compare a factor
under a different variant, re-register it with a new ``factor_id`` or
re-backfill (which overwrites the column).

The work DB is the temporary research playground. The library DB
(``factor_library.duckdb``) holds the stabilised factors and is the only
source consulted by the cross-factor correlation check during evaluation.

All status / history records live inside ``registry.json``.

CLI:
    python -m backtest.factor.admission admit  f_001
    python -m backtest.factor.admission reject f_002
    python -m backtest.factor.admission status                # list all
    python -m backtest.factor.admission status f_001          # show one factor
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from backtest.factor.admission_check import RidgeCheckResult, ridge_r2_check
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
STATUS_PENDING = "pending"
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


# Categories that bootstrap the library — they ARE the regressors used by
# the ridge R² check, so they're admitted before the check exists for anything
# else. Always-skip these from the gate.
_BOOTSTRAP_CATEGORIES: frozenset[str] = frozenset({"barra_l3", "barra_l1"})


def _finalize_action(
    factor_id: str,
    status: Status,
    *,
    rows_promoted: int,
    rows_cleared: int,
    notes: str | None,
    registry: dict,
    persist: bool,
    strategy_config: dict | None = None,
    ridge_check: RidgeCheckResult | None = None,
) -> AdmissionAction:
    """Stamp the action onto the registry."""
    entry = {
        "action": status,
        "rows_promoted": rows_promoted,
        "rows_cleared": rows_cleared,
        "timestamp": _now_iso(),
        "notes": notes,
    }
    if strategy_config is not None:
        entry["strategy_config"] = strategy_config
    if ridge_check is not None:
        entry["ridge_check"] = ridge_check.as_meta()

    meta = registry[factor_id]
    meta["status"] = status
    meta["admission"] = entry
    if ridge_check is not None:
        meta["tier"] = ridge_check.tier
        meta["r2"] = float(ridge_check.r2)
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
    strategy_config: dict | None = None,
    force: bool = False,
    skip_ridge_check: bool = False,
) -> AdmissionAction:
    """Promote a factor from the work DB into the stable library.

    Steps:
      1. Read factor column from ``FactorStorage`` (work DB).
      2. Run :func:`ridge_r2_check` against the 6 Barra L1 styles in the
         library (skipped for ``barra_l3`` / ``barra_l1`` categories — they
         ARE the regressors). A ``reject`` tier blocks promotion unless
         ``force=True``; the tier + R² are stamped onto meta either way.
      3. Upsert into ``FactorLibrary`` (library DB).
      4. Drop the column from the work DB.
      5. Mark ``registry[factor_id]["status"] = "admitted"``.

    Raises
    ------
    KeyError
        If ``factor_id`` is not registered.
    ValueError
        If the work DB has no data for this factor, or the ridge check
        returned ``reject`` and ``force`` is False.
    """
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    meta = target[factor_id]
    category = str(meta.get("category", ""))
    should_check = (
        not skip_ridge_check and category not in _BOOTSTRAP_CATEGORIES
    )

    ridge_result: RidgeCheckResult | None = None
    if should_check:
        ridge_result = ridge_r2_check(factor_id)
        if ridge_result.tier == "reject" and not force:
            raise ValueError(
                f"{factor_id} blocked by ridge_r2_check: R²={ridge_result.r2:.3f} "
                f"-> tier=reject. Override with force=True if you really want "
                f"this style-clone in the library."
            )

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
        strategy_config=strategy_config,
        ridge_check=ridge_result,
    )


def reject(
    factor_id: str,
    *,
    notes: str | None = None,
    registry: dict | None = None,
    strategy_config: dict | None = None,
) -> AdmissionAction:
    """Discard a factor — clear its work-DB column, mark ``status='rejected'``.

    The library DB is **not** touched. If the factor was previously admitted,
    this raises ``ValueError``.
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
        strategy_config=strategy_config,
    )


def _discover_strategy_config(
    factor_id: str,
    *,
    results_root: str | Path = "results",
    tag: str | None = None,
) -> dict | None:
    """Read ``results/<factor_id>/<tag>/pipeline.json`` strategy_config block.

    Returns ``None`` if the factor's results directory or pipeline.json is
    missing (skipping the strategy_config record is fine). Raises
    ``ValueError`` if multiple tag subdirectories exist and ``tag`` is
    unspecified.
    """
    factor_dir = Path(results_root) / factor_id
    if not factor_dir.exists():
        return None

    if tag is not None:
        target = factor_dir / tag / "pipeline.json"
        if not target.exists():
            available = sorted(
                p.name for p in factor_dir.iterdir()
                if p.is_dir() and p.name != "factor_eval"
            )
            raise FileNotFoundError(
                f"pipeline.json not found at {target}. Available tags: {available}"
            )
        return _load_pipeline_strategy_config(target)

    candidates = sorted(
        p for p in factor_dir.iterdir()
        if p.is_dir() and p.name != "factor_eval" and (p / "pipeline.json").exists()
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        names = [p.name for p in candidates]
        raise ValueError(
            f"Found multiple pipeline.json under {factor_dir}: {names}. "
            f"Pass --tag <tag> to pick one (or --no-strategy-config to skip)."
        )
    return _load_pipeline_strategy_config(candidates[0] / "pipeline.json")


def _load_pipeline_strategy_config(path: Path) -> dict | None:
    """Read ``pipeline.json``, return ``strategy_config`` block or None."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload.get("strategy_config")
    if not isinstance(cfg, dict):
        return None
    return cfg


# ---------------------------------------------------------------------------
# Registry queries
# ---------------------------------------------------------------------------


def _factor_ids_where(predicate) -> list[str]:
    return sorted(
        fid for fid, meta in get_registry().items()
        if predicate(meta.get("status"))
    )


def get_admitted_factor_ids() -> list[str]:
    """Factor IDs whose status is admitted."""
    return _factor_ids_where(lambda st: st == STATUS_ADMITTED)


def get_rejected_factor_ids() -> list[str]:
    """Factor IDs whose status is rejected."""
    return _factor_ids_where(lambda st: st == STATUS_REJECTED)


def get_pending_factor_ids() -> list[str]:
    """Factor IDs that have not been admitted or rejected."""
    return _factor_ids_where(lambda st: st not in _VALID_STATUSES)


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
    print(f"  work column cleared    : {action.rows_cleared}")
    print(f"  timestamp              : {action.timestamp}")
    print(f"{'=' * 60}\n")


def print_status(factor_id: str | None = None) -> None:
    """Show admission status for one factor or all factors."""
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

    print(f"\n{'factor_id':<14}{'variant':<18}{'freq':<6}{'status':<12}{'name'}")
    print("-" * 78)
    for fid, meta in rows:
        st = meta.get("status", STATUS_PENDING)
        variant = meta.get("variant", "-")
        freq = meta.get("frequency", "-")
        print(f"{fid:<14}{variant:<18}{freq:<6}{st:<12}{meta.get('name', '')}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Factor admission")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_meta_flags(sp):
        sp.add_argument("--tag", default=None,
                        help="results/<fid>/<tag>/ subdirectory. When set, "
                             "admit/reject reads pipeline.json's strategy_config "
                             "block into history.")
        sp.add_argument("--results-root", default="results",
                        help="results root directory, default 'results'.")
        sp.add_argument("--no-strategy-config", action="store_true",
                        help="Skip reading pipeline.json; history won't carry "
                             "strategy_config block.")

    p_admit = sub.add_parser("admit", help="Promote factor from work DB to library DB")
    p_admit.add_argument("factor_id")
    p_admit.add_argument("--notes", default=None, help="Free-form note for history")
    p_admit.add_argument("--force", action="store_true",
                         help="Bypass the ridge R² reject gate. Use only when "
                              "you really want a style-clone (R²>=0.80) in the library.")
    p_admit.add_argument("--skip-ridge-check", action="store_true",
                         help="Don't run the ridge check at all (e.g. when "
                              "library doesn't yet contain the 6 Barra L1).")
    _add_meta_flags(p_admit)

    p_reject = sub.add_parser("reject", help="Discard factor — clear work DB, mark rejected")
    p_reject.add_argument("factor_id")
    p_reject.add_argument("--notes", default=None, help="Free-form note for history")
    _add_meta_flags(p_reject)

    p_status = sub.add_parser("status", help="Show admission status")
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
                    args.factor_id,
                    results_root=args.results_root, tag=args.tag,
                )
            except (ValueError, FileNotFoundError) as exc:
                parser.exit(2, f"{args.cmd} failed: {exc}\n")
        try:
            if args.cmd == "admit":
                action = admit(
                    args.factor_id, notes=args.notes,
                    strategy_config=strategy_config,
                    force=args.force,
                    skip_ridge_check=args.skip_ridge_check,
                )
            else:
                action = reject(
                    args.factor_id, notes=args.notes,
                    strategy_config=strategy_config,
                )
        except (KeyError, ValueError) as exc:
            parser.exit(2, f"{args.cmd} failed: {exc}\n")
        print_action(action)
        if strategy_config:
            print(f"  strategy_config recorded ({len(strategy_config)} fields)")
        if args.cmd == "admit":
            entry = get_registry()[args.factor_id].get("admission", {})
            rc = entry.get("ridge_check")
            if rc is not None:
                print(f"  ridge check          : R²={rc['r2']:.3f} "
                      f"tier={rc['tier']} n_obs={rc['n_obs']:,}")
                if rc.get("residual_icir") is not None:
                    print(f"  residual ICIR        : {rc['residual_icir']:.3f}")
        print()
        return

    if args.cmd == "status":
        print_status(args.factor_id)
        return


if __name__ == "__main__":
    main()
