#!/usr/bin/env python3
"""
Backfill daily_basic indicators for existing market_daily rows.
Uses SQL temporary-table + UPDATE FROM pattern (no pandas merge, no re-fetch of daily/adj/ST data).

Usage:
    python -m backtest.data.backfill_daily_basic
"""

from tqdm import tqdm

from backtest.data.daily_fetcher import fetch_daily_basic
from backtest.data.storage import MarketStorage


_DAILY_BASIC_COLS = [
    "turnover_rate", "turnover_rate_f", "volume_ratio",
    "pe", "pe_ttm", "pb", "ps", "ps_ttm",
    "dv_ratio", "dv_ttm",
    "total_share", "float_share", "free_share",
    "total_mv", "circ_mv",
]


def main():
    with MarketStorage() as storage:
        # Find dates that already have daily data but no daily_basic yet
        # Use turnover_rate as sentinel (any daily_basic column would work)
        dates = storage.conn.execute("""
            SELECT DISTINCT date
            FROM market_daily
            WHERE turnover_rate IS NULL
            ORDER BY date
        """).fetchdf()["date"].tolist()

        dates = [d.strftime("%Y%m%d") for d in dates]
        print(f"Dates to backfill: {len(dates)}")

        if not dates:
            print("All rows already have daily_basic indicators.")
            return

        failed_dates = []
        updated_total = 0

        for trade_date in tqdm(dates, desc="Backfill daily_basic"):
            try:
                basic_df = fetch_daily_basic(trade_date)
                if basic_df.empty:
                    continue

                # Register as temporary view and UPDATE only matching rows
                storage.conn.register("tmp_basic", basic_df)
                try:
                    set_clause = ", ".join(
                        f'"{c}" = t."{c}"' for c in _DAILY_BASIC_COLS if c in basic_df.columns
                    )
                    result = storage.conn.execute(f"""
                        UPDATE market_daily m
                        SET {set_clause}
                        FROM tmp_basic t
                        WHERE m.date = strptime(t.trade_date, '%Y%m%d')::DATE
                          AND m.symbol = t.ts_code
                    """)
                    updated_total += result.fetchone()[0]
                finally:
                    storage.conn.unregister("tmp_basic")

            except Exception as exc:
                failed_dates.append((trade_date, str(exc)))
                print(f"\n  WARN: failed {trade_date}: {exc}")
                continue

        if failed_dates:
            print(f"\n  Failed dates ({len(failed_dates)}): {[d for d, _ in failed_dates]}")

        print(f"\nBackfill complete. Updated {updated_total:,} rows.")


if __name__ == "__main__":
    main()
