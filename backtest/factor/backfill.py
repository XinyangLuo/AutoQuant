#!/usr/bin/env python3
"""Backfill factor values into the **work** DB.

Use this while researching a new factor: it lands in ``factors.duckdb``
(work area). Once you decide to admit it, ``admit()`` moves the data to
``factor_library.duckdb`` and clears it from here. Until then it remains a
temporary research artefact.

The compute function emits raw values; this script then runs the
``variant``-specific neutralization pipeline (see
:func:`backtest.factor.compute.apply_variant_pipeline`) before insert.

Daily incremental refresh of *already-admitted* factors lives in
``update.py`` and writes to the library DB.

Usage:
    python -m backtest.factor.backfill f_001                   # single factor
    python -m backtest.factor.backfill --pending               # all pending factors, auto workers
    python -m backtest.factor.backfill --pending --workers 1   # force serial
    python -m backtest.factor.backfill f_001 --test-days 60    # last 60 trade days only
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.admission import get_pending_factor_ids
from backtest.factor.compute import apply_variant_pipeline, compute_factor
from backtest.factor.dag import topological_sort
from backtest.factor.registry import get_registry
from backtest.factor.storage import FactorStorage


def backfill_factor(
    factor_id: str,
    start_date: str,
    end_date: str,
    *,
    market_storage: MarketStorage | None = None,
    factor_storage: FactorStorage | None = None,
) -> int:
    """Backfill a single factor into the work DB. Returns rows written."""
    raw_df = compute_factor(
        factor_id,
        start_date,
        end_date,
        market_storage=market_storage,
        factor_storage=factor_storage,
    )
    if raw_df.empty:
        return 0

    own_market = market_storage is None
    own_factor = factor_storage is None
    try:
        if market_storage is None:
            market_storage = MarketStorage(read_only=True)
        if factor_storage is None:
            factor_storage = FactorStorage()

        df = apply_variant_pipeline(
            raw_df, factor_id,
            market_storage=market_storage,
            factor_storage=factor_storage,
        )
        if df.empty:
            return 0
        factor_storage.insert_factors(df)
        return len(df)
    finally:
        if own_factor and factor_storage is not None:
            factor_storage.close()
        if own_market and market_storage is not None:
            market_storage.close()


def _get_earliest_start_date(stock_list: pd.DataFrame) -> str:
    return str(stock_list["list_date"].min())


def _default_workers() -> int:
    """Bounded default for independent-factor backfill workers."""
    return max(1, min(4, os.cpu_count() or 1))


def _has_intra_request_dependencies(factor_ids: list[str], registry: dict) -> bool:
    """Return True when requested factors depend on each other.

    Parallel backfill is safe for independent factors. If a requested residual
    factor depends on another requested factor, preserve topological order by
    running serially.
    """
    requested = set(factor_ids)
    for fid in factor_ids:
        meta = registry.get(fid, {})
        deps = meta.get("depends_on")
        if isinstance(deps, list) and any(dep in requested for dep in deps):
            return True
    return False


def _backfill_one(factor_id: str, factor_start: str, end_date: str) -> tuple[str, int, str | None]:
    """Backfill a single factor in an isolated thread.  Returns (factor_id, rows, error)."""
    try:
        rows = backfill_factor(factor_id, factor_start, end_date)
        return factor_id, rows, None
    except Exception as exc:
        return factor_id, 0, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Backfill factor values into the work DB")
    parser.add_argument("factor_id", nargs="?", help="Factor ID to backfill (e.g. f_001)")
    parser.add_argument("--pending", action="store_true",
                        help="Backfill all pending (unadmitted, unrejected) factors")
    parser.add_argument("--test-days", type=int, default=None,
                        help="Only backfill the last N trade days (debugging)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto, bounded by 4). "
                             "Use --workers 1 to force serial execution. "
                             "Each worker opens its own DB connections.")
    args = parser.parse_args()
    if args.workers is None:
        args.workers = _default_workers()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if not args.pending and not args.factor_id:
        parser.error("Specify a factor_id or --pending")

    stock_list = fetch_stock_list()
    earliest_date = _get_earliest_start_date(stock_list)

    with MarketStorage(read_only=True) as market_storage:
        latest_date = market_storage.get_max_date()
        if latest_date is None:
            print("market_daily is empty. Run cold_start first.")
            return

        if args.test_days:
            all_dates = get_trade_dates(earliest_date, latest_date)
            start_date = (all_dates[-args.test_days]
                          if len(all_dates) >= args.test_days else earliest_date)
            end_date = latest_date
        else:
            start_date = earliest_date
            end_date = latest_date

        print(f"Backfill range: {start_date} ~ {end_date} (work DB)")

        if args.factor_id:
            factor_ids = [args.factor_id]
        else:
            factor_ids = get_pending_factor_ids()

        if not factor_ids:
            print("No factors to backfill.")
            return

        print(f"Factors: {factor_ids}")

        registry = get_registry()
        try:
            factor_ids = topological_sort(factor_ids, registry)
        except ValueError as e:
            print(f"WARNING: dependency cycle detected, "
                  f"falling back to original order: {e}")

        # Collect resume info serially (requires a single storage handle).
        backfill_tasks: list[tuple[str, str]] = []
        with FactorStorage() as factor_storage:
            for factor_id in factor_ids:
                existing_max = factor_storage.get_max_date(factor_id)
                if existing_max and existing_max >= end_date:
                    print(f"  {factor_id}: already up to date ({existing_max})")
                    continue
                if existing_max:
                    resume_dates = get_trade_dates(existing_max, end_date)
                    factor_start = resume_dates[1] if len(resume_dates) > 1 else end_date
                    print(f"  {factor_id}: resuming from {factor_start}")
                else:
                    factor_start = start_date
                backfill_tasks.append((factor_id, factor_start))

        if not backfill_tasks:
            print("\nBackfill complete (all factors up to date).")
            return

        effective_workers = min(args.workers, len(backfill_tasks))
        if effective_workers > 1 and _has_intra_request_dependencies(factor_ids, registry):
            print("  Dependency edges detected within requested factors; using serial backfill.")
            effective_workers = 1

        if effective_workers > 1:
            print(f"Backfilling with {effective_workers} workers ...")
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(_backfill_one, fid, fstart, end_date): fid
                    for fid, fstart in backfill_tasks
                }
                for future in as_completed(futures):
                    factor_id, rows, err = future.result()
                    if err:
                        print(f"  ERROR {factor_id}: {err}")
                    else:
                        print(f"  {factor_id}: wrote {rows:,} rows")
        else:
            for factor_id, factor_start in tqdm(backfill_tasks, desc="backfill"):
                try:
                    rows = backfill_factor(
                        factor_id,
                        factor_start,
                        end_date,
                        market_storage=market_storage,
                    )
                    print(f"  {factor_id}: wrote {rows:,} rows")
                except Exception as exc:
                    print(f"  ERROR {factor_id}: {exc}")
                    continue

        print("\nBackfill complete.")


if __name__ == "__main__":
    main()
