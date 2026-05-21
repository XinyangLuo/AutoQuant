#!/usr/bin/env python3
"""
Backfill limit_up / limit_down for existing market_daily rows.
Uses SQL temporary-table + UPDATE FROM pattern (no pandas merge, no re-fetch of daily/adj/ST data).

Usage:
    python -m backtest.data.backfill_limit_prices
"""

from tqdm import tqdm

from backtest.data.daily_fetcher import fetch_limit_prices
from backtest.data.storage import MarketStorage
from backtest.data.tushare_client import api_call


def main():
    with MarketStorage() as storage:
        # Find dates that already have daily data but no limit prices yet
        dates = storage.conn.execute("""
            SELECT DISTINCT date
            FROM market_daily
            WHERE limit_up IS NULL
            ORDER BY date
        """).fetchdf()["date"].tolist()

        dates = [d.strftime("%Y%m%d") for d in dates]
        print(f"Dates to backfill: {len(dates)}")

        if not dates:
            print("All rows already have limit_up / limit_down.")
            return

        failed_dates = []
        updated_total = 0

        for trade_date in tqdm(dates, desc="Backfill limit prices"):
            try:
                limit_df = fetch_limit_prices(trade_date)
                if limit_df.empty:
                    continue

                # Register as temporary view and UPDATE only matching rows
                storage.conn.register("tmp_limit", limit_df)
                try:
                    result = storage.conn.execute("""
                        UPDATE market_daily m
                        SET limit_up = t.up_limit,
                            limit_down = t.down_limit
                        FROM tmp_limit t
                        WHERE m.date = strptime(t.trade_date, '%Y%m%d')::DATE
                          AND m.symbol = t.ts_code
                    """)
                    updated_total += result.fetchone()[0]
                finally:
                    storage.conn.unregister("tmp_limit")

            except Exception as exc:
                failed_dates.append((trade_date, str(exc)))
                print(f"\n  WARN: failed {trade_date}: {exc}")
                continue

        if failed_dates:
            print(f"\n  Failed dates ({len(failed_dates)}): {[d for d, _ in failed_dates]}")

        print(f"\nBackfill complete. Updated {updated_total:,} rows.")


if __name__ == "__main__":
    main()
