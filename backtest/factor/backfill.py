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
    python -m backtest.factor.backfill --pending               # all pending factors
    python -m backtest.factor.backfill f_001 --test-days 60    # last 60 trade days only
"""

from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.admission import get_pending_factor_ids
from backtest.factor.compute import apply_variant_pipeline, compute_factor
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
            market_storage = MarketStorage()
        if factor_storage is None:
            factor_storage = FactorStorage()

        df = apply_variant_pipeline(
            raw_df, factor_id, market_storage=market_storage,
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


def main():
    parser = argparse.ArgumentParser(description="Backfill factor values into the work DB")
    parser.add_argument("factor_id", nargs="?", help="Factor ID to backfill (e.g. f_001)")
    parser.add_argument("--pending", action="store_true",
                        help="Backfill all pending (unadmitted, unrejected) factors")
    parser.add_argument("--test-days", type=int, default=None,
                        help="Only backfill the last N trade days (debugging)")
    args = parser.parse_args()

    if not args.pending and not args.factor_id:
        parser.error("Specify a factor_id or --pending")

    stock_list = fetch_stock_list()
    earliest_date = _get_earliest_start_date(stock_list)

    with MarketStorage() as market_storage:
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

        with FactorStorage() as factor_storage:
            for factor_id in tqdm(factor_ids, desc="backfill"):
                try:
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

                    rows = backfill_factor(
                        factor_id,
                        factor_start,
                        end_date,
                        market_storage=market_storage,
                        factor_storage=factor_storage,
                    )
                    print(f"  {factor_id}: wrote {rows:,} rows")
                except Exception as exc:
                    print(f"  ERROR {factor_id}: {exc}")
                    continue

        print("\nBackfill complete.")


if __name__ == "__main__":
    main()
