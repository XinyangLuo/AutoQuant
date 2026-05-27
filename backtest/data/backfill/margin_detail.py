#!/usr/bin/env python3
"""
Backfill margin trading detail (margin_detail) for existing market_daily rows.
Uses SQL temporary-table + UPDATE FROM pattern.

Usage:
    python -m backtest.data.backfill.margin_detail
"""

from tqdm import tqdm

from backtest.data.fetcher.daily_fetcher import MARGIN_COLS, MARGIN_RENAME_MAP, fetch_margin_detail
from backtest.data.storage import MarketStorage


def main():
    with MarketStorage() as storage:
        dates = storage.conn.execute("""
            SELECT DISTINCT date
            FROM market_daily
            WHERE margin_rzye IS NULL
               OR margin_rqye IS NULL
               OR margin_rzmre IS NULL
               OR margin_rqyl IS NULL
               OR margin_rzche IS NULL
               OR margin_rqchl IS NULL
               OR margin_rqmcl IS NULL
               OR margin_rzrqye IS NULL
            ORDER BY date
        """).fetchdf()["date"].tolist()

        dates = [d.strftime("%Y%m%d") for d in dates]
        print(f"Dates to backfill: {len(dates)}")

        if not dates:
            print("All rows already have margin_detail data.")
            return

        failed_dates = []
        updated_total = 0

        for trade_date in tqdm(dates, desc="Backfill margin_detail"):
            try:
                margin_df = fetch_margin_detail(trade_date)
                if margin_df.empty:
                    continue

                margin_df = margin_df.rename(columns=MARGIN_RENAME_MAP)

                storage.conn.register("tmp_margin", margin_df)
                try:
                    set_clause = ", ".join(
                        f'"{c}" = t."{c}"' for c in MARGIN_COLS if c in margin_df.columns
                    )
                    if not set_clause:
                        continue
                    result = storage.conn.execute(f"""
                        UPDATE market_daily m
                        SET {set_clause}
                        FROM tmp_margin t
                        WHERE m.date = strptime(t.date, '%Y%m%d')::DATE
                          AND m.symbol = t.symbol
                    """)
                    updated_total += result.fetchone()[0]
                finally:
                    storage.conn.unregister("tmp_margin")

            except Exception as exc:
                failed_dates.append((trade_date, str(exc)))
                print(f"\n  WARN: failed {trade_date}: {exc}")
                continue

        if failed_dates:
            print(f"\n  Failed dates ({len(failed_dates)}): {[d for d, _ in failed_dates]}")

        print(f"\nBackfill complete. Updated {updated_total:,} rows.")


if __name__ == "__main__":
    main()
