#!/usr/bin/env python3
"""
Incremental update for market_daily, income/balancesheet/cashflow, dividends,
index_daily, and index_members.

Phases run sequentially under one DB connection:
  0. trade_calendar — append from MAX(cal_date)+1 to today
  1. market_daily   — from MAX(date)+1 to today, per trade date
  2. fundamentals   — from MAX(f_ann_date) to today, per table (income / balancesheet / cashflow)
  3. dividends      — from MAX(ann_date) to today, per trade date by ann_date
  4. index_daily    — per-index MAX(date)+1 → today (benchmark OHLCV)
  5. index_members  — per-index MAX(trade_date)+1 → today (densified monthly snapshots)

``sw_industry`` is a slow-moving table (industry assignments change rarely)
and ``backfill/sw_industry.py`` does a full rebuild, so it is excluded from
the daily loop. Pass ``--include-sw-industry`` to rebuild it as well.

Usage:
    python -m backtest.data.update_daily
    python -m backtest.data.update_daily --include-sw-industry
"""

import argparse
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from backtest.data._pipeline import update_by_ann_date
from backtest.data.backfill.index_members import (
    DEFAULT_INDICES as INDEX_MEMBERS_DEFAULT,
    backfill_index_members,
)
from backtest.data.backfill.indices import (
    DEFAULT_INDICES as INDICES_DEFAULT,
    backfill_indices,
)
from backtest.data.backfill.cyq_chips import backfill_cyq_chips
from backtest.data.backfill.sw_industry import backfill_sw_industry
from backtest.data.fetcher.daily_fetcher import build_list_date_map, process_trade_date
from backtest.data.fetcher.dividends_fetcher import fetch_dividend_by_ann_date
from backtest.data.fetcher.fundamentals_fetcher import (
    fetch_balancesheet_by_f_ann_date,
    fetch_cashflow_by_f_ann_date,
    fetch_income_by_f_ann_date,
)
from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.data.backfill_trade_calendar import backfill_trade_calendar


def _next_day(yyyymmdd: str) -> str:
    return (datetime.strptime(yyyymmdd, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


def update_market_daily(storage: MarketStorage, *, stock_list: pd.DataFrame) -> bool:
    """Update market_daily up to today. Returns True if updates ran."""
    list_date_map = build_list_date_map(stock_list)
    today = datetime.now().strftime("%Y%m%d")

    max_date = storage.get_max_date()
    if max_date is None:
        print("market_daily: DB is empty. Run cold_start first.")
        return False

    start = _next_day(max_date)
    if start > today:
        print(f"market_daily: already up to date (last date {max_date}).")
        return False

    trade_dates = get_trade_dates(start, today)
    print(f"market_daily: {len(trade_dates)} trade dates to update "
          f"({start} ~ {today})")

    if not trade_dates:
        print("market_daily: no new trade dates.")
        return False

    failed = []
    for trade_date in tqdm(trade_dates, desc="market_daily"):
        try:
            daily_df = process_trade_date(trade_date, list_date_map)
            if not daily_df.empty:
                storage.insert_daily(daily_df)
        except Exception as exc:
            failed.append((trade_date, str(exc)))
            print(f"\n  WARN: failed {trade_date}: {exc}")
            continue

    if failed:
        print(f"\n  Failed dates ({len(failed)}): {[d for d, _ in failed]}")

    stats = storage.get_stats()
    print(f"market_daily: {stats['total_rows']:,} rows, "
          f"{stats['min_date']} ~ {stats['max_date']}")
    return True


def update_fundamentals(storage: MarketStorage) -> None:
    """Run incremental update for income, balancesheet, and cashflow."""
    configs = [
        ("income", "income_q", fetch_income_by_f_ann_date, storage.insert_income),
        ("balancesheet", "balancesheet_q", fetch_balancesheet_by_f_ann_date, storage.insert_balancesheet),
        ("cashflow", "cashflow_q", fetch_cashflow_by_f_ann_date, storage.insert_cashflow),
    ]
    for label, table, fetch_fn, insert_fn in configs:
        update_by_ann_date(
            label=label,
            get_max_ann_date=lambda t=table: storage.get_max_f_ann_date(t),
            fetch_by_ann_date=fetch_fn,
            insert=insert_fn,
        )


def update_dividends(storage: MarketStorage) -> None:
    update_by_ann_date(
        label="dividends",
        get_max_ann_date=storage.get_max_dividend_ann_date,
        fetch_by_ann_date=fetch_dividend_by_ann_date,
        insert=storage.insert_dividends,
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-sw-industry", action="store_true",
        help="Also rebuild sw_industry (slow-moving; off by default).",
    )
    args = parser.parse_args(argv)

    stock_list = fetch_stock_list()
    print(f"Stock list: {len(stock_list)} stocks")

    with MarketStorage() as storage:
        print("\n=== Phase 0: trade_calendar ===")
        max_cal = storage.get_max_cal_date()
        today = datetime.now().strftime("%Y%m%d")
        if max_cal:
            cal_start = _next_day(max_cal)
        else:
            cal_start = stock_list["list_date"].min()
        if cal_start <= today:
            n_cal = backfill_trade_calendar(start_date=cal_start, end_date=today)
            print(f"  trade_calendar: updated {n_cal:,} rows ({cal_start} ~ {today})")
        else:
            print("  trade_calendar: already up to date.")

        print("\n=== Phase 1: market_daily ===")
        update_market_daily(storage, stock_list=stock_list)

        print("\n=== Phase 2: income + balancesheet + cashflow ===")
        update_fundamentals(storage)

        print("\n=== Phase 3: dividends ===")
        update_dividends(storage)

        print("\n=== Phase 4: index_daily ===")
        backfill_indices(INDICES_DEFAULT)

        print("\n=== Phase 5: index_members ===")
        backfill_index_members(INDEX_MEMBERS_DEFAULT)

        print("\n=== Phase 6: cyq_chips ===")
        backfill_cyq_chips(
            symbols=stock_list["ts_code"].tolist(),
            sleep_sec=0.01,
        )

        if args.include_sw_industry:
            print("\n=== Phase 7: sw_industry (full rebuild) ===")
            backfill_sw_industry(["L1", "L2"])

    print("\n" + "=" * 50)
    print("Update complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
