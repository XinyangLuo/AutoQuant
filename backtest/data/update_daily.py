#!/usr/bin/env python3
"""
Incremental update: fetch daily data from the last date in DB to today.
Fetches by trade date (all stocks per day).

Usage:
    python -m backtest.data.update_daily
"""

from datetime import datetime, timedelta

from tqdm import tqdm

from backtest.data.daily_fetcher import build_list_date_map, process_trade_date
from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates


def main():
    stock_list = fetch_stock_list()
    list_date_map = build_list_date_map(stock_list)
    print(f"Stock list: {len(stock_list)} stocks in market")

    today = datetime.now().strftime("%Y%m%d")

    with MarketStorage() as storage:
        max_date = storage.get_max_date()

        if max_date is None:
            print("DB is empty. Please run cold_start.py first.")
            return

        start = (datetime.strptime(max_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

        if start > today:
            print(f"Already up to date. Last date in DB: {max_date}")
            return

        trade_dates = get_trade_dates(start, today)
        print(f"Trade dates to update: {len(trade_dates)} ({start} ~ {today})")

        if not trade_dates:
            print("No new trade dates to update.")
            return

        for trade_date in tqdm(trade_dates, desc="Updating"):
            daily_df = process_trade_date(trade_date, list_date_map)
            if not daily_df.empty:
                storage.insert_daily(daily_df)

        stats = storage.get_stats()

    print("\n" + "=" * 50)
    print("Update complete.")
    print(f"  Total rows    : {stats['total_rows']:,}")
    print(f"  Total symbols : {stats['total_symbols']:,}")
    print(f"  Date range    : {stats['min_date']} ~ {stats['max_date']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
