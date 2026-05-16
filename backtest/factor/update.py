#!/usr/bin/env python3
"""
Incremental update for all registered factors.

Resumes from each factor's MAX(date) and computes up to the latest market_daily date.

Usage:
    python -m backtest.factor.update
"""

from __future__ import annotations

from tqdm import tqdm

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.compute import compute_factor
from backtest.factor.admission import get_admitted_factor_ids
from backtest.factor.registry import get_registry
from backtest.factor.storage import FactorStorage


def main():
    with MarketStorage() as market_storage:
        latest_date = market_storage.get_max_date()
        if latest_date is None:
            print("market_daily is empty. Run cold_start first.")
            return

        factor_ids = get_admitted_factor_ids()
        if not factor_ids:
            print("No admitted factors found. Run admission gate first.")
            return

        with FactorStorage() as factor_storage:
            for factor_id in tqdm(factor_ids, desc="update"):
                max_date = factor_storage.get_max_date(factor_id)

                if max_date and max_date >= latest_date:
                    continue

                if max_date:
                    resume_dates = get_trade_dates(max_date, latest_date)
                    start = resume_dates[1] if len(resume_dates) > 1 else latest_date
                else:
                    start = latest_date

                if start > latest_date:
                    continue

                try:
                    df = compute_factor(
                        factor_id,
                        start,
                        latest_date,
                        market_storage=market_storage,
                        factor_storage=factor_storage,
                    )
                    if not df.empty:
                        factor_storage.insert_factors(df)
                        print(f"  {factor_id}: {len(df):,} rows ({start} ~ {latest_date})")
                except Exception as exc:
                    print(f"  ERROR {factor_id}: {exc}")
                    continue

        print("\nUpdate complete.")


if __name__ == "__main__":
    main()
