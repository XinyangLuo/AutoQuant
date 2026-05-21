#!/usr/bin/env python3
"""
Backfill capital flow (moneyflow) for existing market_daily rows.
Uses SQL temporary-table + UPDATE FROM pattern.

Usage:
    python -m backtest.data.backfill.moneyflow
"""

from tqdm import tqdm

from backtest.data.fetcher.daily_fetcher import (
    MONEYFLOW_COLS,
    MONEYFLOW_RENAME_MAP,
    convert_moneyflow_units,
    fetch_moneyflow,
)
from backtest.data.storage import MarketStorage


def main():
    with MarketStorage() as storage:
        dates = storage.conn.execute("""
            SELECT DISTINCT date
            FROM market_daily
            WHERE mf_net_mf_amount IS NULL
            ORDER BY date
        """).fetchdf()["date"].tolist()

        dates = [d.strftime("%Y%m%d") for d in dates]
        print(f"Dates to backfill: {len(dates)}")

        if not dates:
            print("All rows already have moneyflow data.")
            return

        failed_dates = []
        updated_total = 0

        for trade_date in tqdm(dates, desc="Backfill moneyflow"):
            try:
                mf_df = fetch_moneyflow(trade_date)
                if mf_df.empty:
                    continue

                mf_df = mf_df.rename(columns=MONEYFLOW_RENAME_MAP)
                convert_moneyflow_units(mf_df)

                storage.conn.register("tmp_mf", mf_df)
                try:
                    set_clause = ", ".join(
                        f'"{c}" = t."{c}"' for c in MONEYFLOW_COLS if c in mf_df.columns
                    )
                    result = storage.conn.execute(f"""
                        UPDATE market_daily m
                        SET {set_clause}
                        FROM tmp_mf t
                        WHERE m.date = strptime(t.trade_date, '%Y%m%d')::DATE
                          AND m.symbol = t.ts_code
                    """)
                    updated_total += result.fetchone()[0]
                finally:
                    storage.conn.unregister("tmp_mf")

            except Exception as exc:
                failed_dates.append((trade_date, str(exc)))
                print(f"\n  WARN: failed {trade_date}: {exc}")
                continue

        if failed_dates:
            print(f"\n  Failed dates ({len(failed_dates)}): {[d for d, _ in failed_dates]}")

        print(f"\nBackfill complete. Updated {updated_total:,} rows.")


if __name__ == "__main__":
    main()
