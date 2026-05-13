#!/usr/bin/env python3
"""
Cold-start: backfill all historical daily data from the earliest trade date.
Fetches by trade date (all stocks per day).

Usage:
    python -m backtest.data.cold_start
    python -m backtest.data.cold_start --recent-days 10   # quick test
"""

import argparse
from datetime import datetime

from tqdm import tqdm

from backtest.data.daily_fetcher import build_list_date_map, process_trade_date
from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates


def main():
    parser = argparse.ArgumentParser(description="Cold-start market daily data backfill")
    parser.add_argument("--recent-days", type=int, default=None,
                        help="Only fetch last N trade days (for testing)")
    args = parser.parse_args()

    stock_list = fetch_stock_list()
    list_date_map = build_list_date_map(stock_list)
    earliest_list_date = stock_list["list_date"].min()
    print(f"Stock list: {len(stock_list)} stocks, earliest list_date: {earliest_list_date}")

    today = datetime.now().strftime("%Y%m%d")

    if args.recent_days:
        all_dates = get_trade_dates("20200101", today)
        trade_dates = all_dates[-args.recent_days:] if len(all_dates) >= args.recent_days else all_dates
        start_date = trade_dates[0]
    else:
        trade_dates = get_trade_dates(earliest_list_date, today)
        start_date = earliest_list_date

    print(f"Trade dates to fetch: {len(trade_dates)} ({start_date} ~ {today})")

    with MarketStorage() as storage:
        # Resume: skip dates already in DB
        existing = storage.conn.execute(
            "SELECT DISTINCT date FROM market_daily"
        ).fetchdf()["date"].tolist()
        existing = {d.strftime("%Y%m%d") for d in existing}
        trade_dates = [d for d in trade_dates if d not in existing]
        print(f"After skipping existing: {len(trade_dates)} dates to fetch")

        failed_dates = []
        for trade_date in tqdm(trade_dates, desc="Trade dates"):
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

        stats = storage.get_stats()

    print("\n" + "=" * 50)
    print("Cold start complete.")
    print(f"  Total rows    : {stats['total_rows']:,}")
    print(f"  Total symbols : {stats['total_symbols']:,}")
    print(f"  Date range    : {stats['min_date']} ~ {stats['max_date']}")
    print(f"  DB path       : {storage.db_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
