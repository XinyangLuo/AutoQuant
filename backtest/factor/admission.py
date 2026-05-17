#!/usr/bin/env python3
"""Factor admission gate: evaluate and decide whether a factor enters the library.

A factor must pass all configured thresholds to be "admitted". Once admitted,
it is eligible for automatic backfill and incremental updates.

All admission records live inside ``registry.json`` — each factor entry carries
its ``status``, latest ``admission`` snapshot, and ``admission_history`` list.
No separate admission log file.

Usage:
    python -m backtest.factor.admission f_001 --start 20210101 --end 20241231
    python -m backtest.factor.admission --all --start 20210101 --end 20241231
    python -m backtest.factor.admission --pending --start 20210101 --end 20241231
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from backtest.factor.evaluation import evaluate, EvaluationResult
from backtest.factor.registry import (
    get_registry,
    _load_registry,
    _save_registry,
    sync_registry,
)
from backtest.factor.storage import FactorStorage

DEFAULT_CONFIG: dict[str, object] = {
    "min_rankicir": 0.25,
    "min_ic_positive_ratio": 0.52,
    "max_turnover": 0.5,
    "max_corr": 0.85,
    "primary_horizon": 20,
    "ret_type": "open",
    "exclude_limit_up": True,
}

_STATUS_ADMITTED = "admitted"
_STATUS_REJECTED = "rejected"
_VALID_STATUSES = frozenset({_STATUS_ADMITTED, _STATUS_REJECTED})


@dataclass
class AdmissionResult:
    factor_id: str
    passed: bool
    checks: dict[str, bool]
    metrics: dict
    config: dict

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "REJECTED"
        return f"AdmissionResult({self.factor_id}, {status})"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def admit(
    factor_id: str,
    start: str,
    end: str,
    *,
    config: dict | None = None,
    horizons: list[int] | None = None,
    corr_top_k: int = 5,
    registry: dict | None = None,
    keep_data: bool = False,
) -> AdmissionResult:
    """Run evaluation and decide admission.

    Parameters
    ----------
    factor_id : str
        Factor to evaluate.
    start, end : str
        Evaluation date range (YYYYMMDD).
    config : dict, optional
        Override default admission thresholds.
    horizons : list[int], optional
        Forward return horizons. Defaults to [1, 5, 10, 20, 60].
    corr_top_k : int
        Top-K correlated factors to check (0 to skip).
    registry : dict, optional
        If provided, mutate this in-memory dict instead of reading/writing
        ``registry.json`` on disk. The caller must persist the dict.
    keep_data : bool, default False
        If ``False`` (default), rejected factors have their data automatically
        deleted from ``factors_daily`` to keep storage clean.

    Returns
    -------
    AdmissionResult
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if horizons is None:
        horizons = [1, 5, 10, 20, 60]

    primary_h = cfg["primary_horizon"]
    if primary_h not in horizons:
        horizons = sorted(set(horizons + [primary_h]))

    ret_type = cfg["ret_type"]
    exclude_limit_up = cfg.get("exclude_limit_up", True)

    result: EvaluationResult = evaluate(
        factor_id,
        start,
        end,
        horizons=horizons,
        ret_type=ret_type,
        corr_top_k=corr_top_k,
        exclude_limit_up=exclude_limit_up,
    )

    ric = result.rank_ic_metrics.get(primary_h, {})
    ic = result.ic_metrics.get(primary_h, {})

    rankicir = ric.get("icir", float("-inf"))
    pos_ratio = ic.get("ic_positive_ratio", 0.0)
    turnover = result.turnover
    max_corr_pair = result.max_corr()
    max_corr = abs(max_corr_pair[1]) if max_corr_pair else 0.0

    checks = {
        "rankicir": rankicir >= cfg["min_rankicir"],
        "ic_positive_ratio": pos_ratio >= cfg["min_ic_positive_ratio"],
        "turnover": turnover < cfg["max_turnover"],
        "max_corr": max_corr < cfg["max_corr"],
    }

    passed = all(checks.values())

    metrics = {
        "rankicir": round(rankicir, 4),
        "ic_positive_ratio": round(pos_ratio, 4),
        "turnover": round(turnover, 4),
        "max_corr": round(max_corr, 4),
        "primary_horizon": primary_h,
    }

    entry = {
        "passed": passed,
        "checks": {k: bool(v) for k, v in checks.items()},
        "metrics": metrics,
        "config": cfg,
        "eval_period": f"{start}~{end}",
        "timestamp": _now_iso(),
    }

    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    meta = target[factor_id]
    meta["status"] = _STATUS_ADMITTED if passed else _STATUS_REJECTED
    meta["admission"] = entry
    meta.setdefault("admission_history", []).append(entry)
    # cap history to last 20 entries to prevent unbounded growth
    meta["admission_history"] = meta["admission_history"][-20:]

    # Clean up rejected factor data from DuckDB unless explicitly kept
    if not passed and not keep_data:
        with FactorStorage() as fs:
            n_deleted = fs.delete_factor(factor_id)
            if n_deleted > 0:
                print(f"  [cleanup] Deleted {n_deleted:,} rows for rejected factor {factor_id}")

    if registry is None:
        _save_registry(target)
        sync_registry()

    return AdmissionResult(
        factor_id=factor_id,
        passed=passed,
        checks=checks,
        metrics=metrics,
        config=cfg,
    )


def _get_factor_ids_by_status(status: str | None) -> list[str]:
    """Return factor_ids filtered by status.

    status=None returns factors with no status (pending).
    """
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
    """Return all factor_ids with ``status='admitted'``."""
    return _get_factor_ids_by_status(_STATUS_ADMITTED)


def get_rejected_factor_ids() -> list[str]:
    """Return all factor_ids with ``status='rejected'``."""
    return _get_factor_ids_by_status(_STATUS_REJECTED)


def get_pending_factor_ids() -> list[str]:
    """Return all factor_ids without a status."""
    return _get_factor_ids_by_status(None)


def print_admission(result: AdmissionResult) -> None:
    """Pretty-print admission result."""
    status = "ADMITTED" if result.passed else "REJECTED"
    print(f"\n{'=' * 60}")
    print(f"Admission: {result.factor_id}  ->  {status}")
    print(f"{'=' * 60}")

    print(f"\nConfig thresholds:")
    for k, v in result.config.items():
        print(f"  {k}: {v}")

    print(f"\nMetrics vs thresholds:")
    m = result.metrics
    c = result.config
    print(f"  RankICIR      = {m['rankicir']:>+8.4f}  (>= {c['min_rankicir']})  {'PASS' if result.checks['rankicir'] else 'FAIL'}")
    print(f"  IC+ ratio     = {m['ic_positive_ratio']:>8.2%}  (>= {c['min_ic_positive_ratio']:.0%})  {'PASS' if result.checks['ic_positive_ratio'] else 'FAIL'}")
    print(f"  Turnover      = {m['turnover']:>8.4f}  (< {c['max_turnover']})  {'PASS' if result.checks['turnover'] else 'FAIL'}")
    print(f"  Max |corr|    = {m['max_corr']:>8.4f}  (< {c['max_corr']})  {'PASS' if result.checks['max_corr'] else 'FAIL'}")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Factor admission gate")
    parser.add_argument("factor_id", nargs="?", help="Factor ID to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all registered factors")
    parser.add_argument("--pending", action="store_true", help="Evaluate only pending (no status) factors")
    parser.add_argument("--start", required=True, help="Evaluation start YYYYMMDD")
    parser.add_argument("--end", required=True, help="Evaluation end YYYYMMDD")
    parser.add_argument("--dry-run", action="store_true", help="Do not write status to registry")
    parser.add_argument(
        "--config",
        type=json.loads,
        default="{}",
        help='JSON override for admission config, e.g. \'{"min_rankicir":3.0}\'',
    )
    parser.add_argument(
        "--ret-type",
        choices=["close", "open"],
        default="close",
        help="Return calculation type",
    )
    parser.add_argument(
        "--no-corr",
        action="store_true",
        help="Skip correlation check with existing factors",
    )
    parser.add_argument(
        "--no-exclude-limit-up",
        action="store_true",
        help="Do NOT exclude limit-up rows from evaluation",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep factor data in DuckDB even if rejected (default: delete)",
    )
    args = parser.parse_args()

    if not args.all and not args.pending and not args.factor_id:
        parser.error("Specify --all, --pending, or a factor_id")

    config = DEFAULT_CONFIG.copy()
    config["ret_type"] = args.ret_type
    config["exclude_limit_up"] = not args.no_exclude_limit_up
    config.update(args.config)

    corr_top_k = 0 if args.no_corr else 5

    if args.all:
        factor_ids = list(get_registry().keys())
    elif args.pending:
        factor_ids = get_pending_factor_ids()
    else:
        factor_ids = [args.factor_id]

    if not factor_ids:
        print("No factors to evaluate.")
        return

    print(f"Evaluating {len(factor_ids)} factor(s) ...")

    registry = None if args.dry_run else _load_registry()
    passed_count = 0
    for fid in factor_ids:
        try:
            result = admit(
                fid,
                args.start,
                args.end,
                config=config,
                corr_top_k=corr_top_k,
                registry=registry,
                keep_data=args.keep_data,
            )
            print_admission(result)
            if result.passed:
                passed_count += 1
        except Exception as exc:
            print(f"\nERROR evaluating {fid}: {exc}\n")
            continue

    if registry is not None:
        _save_registry(registry)
        sync_registry()

    print(f"\n{'=' * 60}")
    print(f"Summary: {passed_count}/{len(factor_ids)} passed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
