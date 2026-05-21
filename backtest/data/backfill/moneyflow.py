#!/usr/bin/env python3
"""
Backfill capital flow (moneyflow) for existing market_daily rows.
Uses SQL temporary-table + UPDATE FROM pattern.

Usage:
    python -m backtest.data.backfill_moneyflow
"""

from tqdm import tqdm

from backtest.data.daily_fetcher import fetch_moneyflow
from backtest.data.storage import MarketStorage

_MONEYFLOW_COLS = [
    "mf_buy_sm_vol", "mf_buy_sm_amount", "mf_sell_sm_vol", "mf_sell_sm_amount",
    "mf_buy_md_vol", "mf_buy_md_amount", "mf_sell_md_vol", "mf_sell_md_amount",
    "mf_buy_lg_vol", "mf_buy_lg_amount", "mf_sell_lg_vol", "mf_sell_lg_amount",
    "mf_buy_elg_vol", "mf_buy_elg_amount", "mf_sell_elg_vol", "mf_sell_elg_amount",
    "mf_net_mf_vol", "mf_net_mf_amount",
]

_MONEYFLOW_VOL_COLS = [
    "mf_buy_sm_vol", "mf_sell_sm_vol",
    "mf_buy_md_vol", "mf_sell_md_vol",
    "mf_buy_lg_vol", "mf_sell_lg_vol",
    "mf_buy_elg_vol", "mf_sell_elg_vol",
    "mf_net_mf_vol",
]

_MONEYFLOW_AMOUNT_COLS = [
    "mf_buy_sm_amount", "mf_sell_sm_amount",
    "mf_buy_md_amount", "mf_sell_md_amount",
    "mf_buy_lg_amount", "mf_sell_lg_amount",
    "mf_buy_elg_amount", "mf_sell_elg_amount",
    "mf_net_mf_amount",
]

_RENAME_MAP = {
    "buy_sm_vol": "mf_buy_sm_vol",
    "buy_sm_amount": "mf_buy_sm_amount",
    "sell_sm_vol": "mf_sell_sm_vol",
    "sell_sm_amount": "mf_sell_sm_amount",
    "buy_md_vol": "mf_buy_md_vol",
    "buy_md_amount": "mf_buy_md_amount",
    "sell_md_vol": "mf_sell_md_vol",
    "sell_md_amount": "mf_sell_md_amount",
    "buy_lg_vol": "mf_buy_lg_vol",
    "buy_lg_amount": "mf_buy_lg_amount",
    "sell_lg_vol": "mf_sell_lg_vol",
    "sell_lg_amount": "mf_sell_lg_amount",
    "buy_elg_vol": "mf_buy_elg_vol",
    "buy_elg_amount": "mf_buy_elg_amount",
    "sell_elg_vol": "mf_sell_elg_vol",
    "sell_elg_amount": "mf_sell_elg_amount",
    "net_mf_vol": "mf_net_mf_vol",
    "net_mf_amount": "mf_net_mf_amount",
}


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

                # Rename Tushare raw columns to market_daily schema
                mf_df = mf_df.rename(columns=_RENAME_MAP)

                # Unit conversion: vol 手→股, amount 万元→元
                for col in _MONEYFLOW_VOL_COLS:
                    if col in mf_df.columns:
                        mf_df[col] = (mf_df[col] * 100).round().astype("int64")
                for col in _MONEYFLOW_AMOUNT_COLS:
                    if col in mf_df.columns:
                        mf_df[col] = (mf_df[col] * 10000).round(3)

                storage.conn.register("tmp_mf", mf_df)
                try:
                    set_clause = ", ".join(
                        f'"{c}" = t."{c}"' for c in _MONEYFLOW_COLS if c in mf_df.columns
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
