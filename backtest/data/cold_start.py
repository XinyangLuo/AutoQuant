#!/usr/bin/env python3
"""
One-button cold start: backfill market_daily, fina_indicator_quarterly,
and dividends from scratch.

Phases run sequentially under one DB connection:
  1. market_daily   — loop trade dates, per-date OHLCV + adj + ST + limit + basic
  2. fina_indicator — loop symbols, per-symbol full quarterly history
  3. dividends      — loop symbols, per-symbol full dividend history

All phases are resumable: rows already in DB are skipped.

Usage:
    python -m backtest.data.cold_start
    python -m backtest.data.cold_start --recent-days 10   # quick test of market_daily only
"""

import argparse
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from backtest.data._pipeline import print_stats
from backtest.data.backfill_dividends import backfill_dividends
from backtest.data.backfill_fina_indicator import backfill_fina
from backtest.data.daily_fetcher import build_list_date_map, process_trade_date
from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates


def cold_start_market_daily(
    storage: MarketStorage,
    *,
    stock_list: pd.DataFrame,
    recent_days: int | None = None,
) -> None:
    """Backfill market_daily by trade date, resuming from existing dates."""
    list_date_map = build_list_date_map(stock_list)
    earliest_list_date = stock_list["list_date"].min()
    print(f"Stock list: {len(stock_list)} stocks, earliest list_date: {earliest_list_date}")

    today = datetime.now().strftime("%Y%m%d")

    if recent_days:
        all_dates = get_trade_dates("20200101", today)
        trade_dates = all_dates[-recent_days:] if len(all_dates) >= recent_days else all_dates
        start_date = trade_dates[0] if trade_dates else today
    else:
        trade_dates = get_trade_dates(earliest_list_date, today)
        start_date = earliest_list_date

    print(f"Trade dates to fetch: {len(trade_dates)} ({start_date} ~ {today})")

    existing = storage.get_existing_dates()
    trade_dates = [d for d in trade_dates if d not in existing]
    print(f"After skipping existing: {len(trade_dates)} dates to fetch")

    failed_dates = []
    for trade_date in tqdm(trade_dates, desc="market_daily"):
        try:
            daily_df = process_trade_date(trade_date, list_date_map)
            if not daily_df.empty:
                storage.insert_daily(daily_df)
        except Exception as exc:
            failed_dates.append((trade_date, str(exc)))
            print(f"\n  WARN: failed {trade_date}: {exc}")
            continue

    if failed_dates:
        print(f"\n  Failed dates ({len(failed_dates)}): {[d for d, _ in failed_dates]}")

    print_stats("market_daily", storage.get_stats(), date_col="date")


def main():
    parser = argparse.ArgumentParser(
        description="Cold start: backfill market_daily, fina_indicator, dividends"
    )
    parser.add_argument("--recent-days", type=int, default=None,
                        help="Only fetch last N trade days for market_daily (test mode; "
                             "skips fina + dividends)")
    args = parser.parse_args()

    test_mode = args.recent_days is not None

    stock_list = fetch_stock_list()

    with MarketStorage() as storage:
        print("\n=== Phase 1: market_daily ===")
        cold_start_market_daily(storage, stock_list=stock_list, recent_days=args.recent_days)

        if test_mode:
            print("\n(test mode: skipping fina + dividends)")
            return

        print("\n=== Phase 2: fina_indicator_quarterly ===")
        backfill_fina(storage, stock_list=stock_list)
        print_stats("fina_indicator", storage.get_fina_stats(), prefix="ann ")

        print("\n=== Phase 3: dividends ===")
        backfill_dividends(storage, stock_list=stock_list)
        print_stats("dividends", storage.get_dividend_stats(), prefix="ann ")

        print("\n" + "=" * 50)
        print("Cold start complete.")
        print(f"  DB path: {storage.db_path}")
        print("=" * 50)


if __name__ == "__main__":
    main()
