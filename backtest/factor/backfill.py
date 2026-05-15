#!/usr/bin/env python3
"""
Unified backfill entry for all registered factors.

Usage:
    python -m backtest.factor.backfill --all
    python -m backtest.factor.backfill f_001
    python -m backtest.factor.backfill --all --test-days 10
"""

from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from backtest.data.storage import MarketStorage
from backtest.data.stock_list import fetch_stock_list
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.compute import compute_factor
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
    """Backfill a single factor from start_date to end_date.

    Returns the number of rows written.
    """
    df = compute_factor(
        factor_id,
        start_date,
        end_date,
        market_storage=market_storage,
        factor_storage=factor_storage,
    )
    if df.empty:
        return 0

    own_factor = factor_storage is None
    try:
        if factor_storage is None:
            factor_storage = FactorStorage()
        factor_storage.insert_factors(df)
        return len(df)
    finally:
        if own_factor and factor_storage is not None:
            factor_storage.close()


def _get_earliest_start_date(stock_list: pd.DataFrame) -> str:
    """Return the earliest list_date among all stocks."""
    earliest = stock_list["list_date"].min()
    return str(earliest)


def main():
    parser = argparse.ArgumentParser(description="Backfill factor values")
    parser.add_argument("factor_id", nargs="?", help="Factor ID to backfill (e.g. f_001)")
    parser.add_argument("--all", action="store_true", help="Backfill all registered factors")
    parser.add_argument("--test-days", type=int, default=None, help="Only backfill last N trade days")
    args = parser.parse_args()

    if not args.all and not args.factor_id:
        parser.error("Specify either --all or a factor_id")

    stock_list = fetch_stock_list()
    earliest_date = _get_earliest_start_date(stock_list)

    with MarketStorage() as market_storage:
        latest_date = market_storage.get_max_date()
        if latest_date is None:
            print("market_daily is empty. Run cold_start first.")
            return

        if args.test_days:
            all_dates = get_trade_dates(earliest_date, latest_date)
            if len(all_dates) >= args.test_days:
                start_date = all_dates[-args.test_days]
            else:
                start_date = earliest_date
            end_date = latest_date
        else:
            start_date = earliest_date
            end_date = latest_date

        print(f"Backfill range: {start_date} ~ {end_date}")

        if args.all:
            registry = get_registry()
            factor_ids = list(registry.keys())
        else:
            factor_ids = [args.factor_id]

        if not factor_ids:
            print("No factors registered.")
            return

        print(f"Factors to backfill: {factor_ids}")

        with FactorStorage() as factor_storage:
            for factor_id in tqdm(factor_ids, desc="backfill"):
                try:
                    existing_max = factor_storage.get_max_date(factor_id)
                    if existing_max and existing_max >= end_date:
                        print(f"  {factor_id}: already up to date ({existing_max})")
                        continue

                    if existing_max:
                        # Get the next trade date after existing_max
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
