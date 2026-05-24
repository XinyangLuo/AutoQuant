#!/usr/bin/env python3
"""Incremental update for admitted factors — writes to the **library** DB.

The library (``factor_library.duckdb``) is the long-term home for stabilised
factors. ``admit()`` seeds it once with the factor's full history; this
script's job is to keep the tail in sync with new market_daily rows.

It does *not* touch the work DB (``factors.duckdb``). New factors live in
work via ``backfill``; only after ``admit`` do they show up here.

Usage:
    python -m backtest.factor.update
"""

from __future__ import annotations

from tqdm import tqdm

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.admission import get_admitted_factor_ids
from backtest.factor.compute import compute_factor
from backtest.factor.storage import FactorLibrary


def main():
    with MarketStorage(read_only=True) as market_storage:
        latest_date = market_storage.get_max_date()
        if latest_date is None:
            print("market_daily is empty. Run cold_start first.")
            return

        factor_ids = get_admitted_factor_ids()
        if not factor_ids:
            print("No admitted factors. Admit one first with "
                  "`python -m backtest.factor.admission admit <factor_id>`.")
            return

        with FactorLibrary() as lib:
            for factor_id in tqdm(factor_ids, desc="update"):
                max_date = lib.get_max_date(factor_id)

                if max_date and max_date >= latest_date:
                    continue

                if max_date:
                    resume_dates = get_trade_dates(max_date, latest_date)
                    start = resume_dates[1] if len(resume_dates) > 1 else latest_date
                else:
                    # An admitted factor with no rows in the library is odd —
                    # likely a manual cleanup. Skip rather than mass-backfill.
                    print(f"  {factor_id}: no library rows yet, skip "
                          f"(re-backfill in work + re-admit if intended)")
                    continue

                if start > latest_date:
                    continue

                try:
                    df = compute_factor(
                        factor_id,
                        start,
                        latest_date,
                        market_storage=market_storage,
                        factor_storage=lib,
                    )
                    if not df.empty:
                        lib.insert_factors(df)
                        print(f"  {factor_id}: {len(df):,} rows ({start} ~ {latest_date})")
                except Exception as exc:
                    print(f"  ERROR {factor_id}: {exc}")
                    continue

        print("\nUpdate complete.")


if __name__ == "__main__":
    main()
