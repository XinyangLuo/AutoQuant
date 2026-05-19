#!/usr/bin/env python3
"""Backfill factor values into the **work** DB.

Use this while researching a new factor: it lands in ``factors.duckdb``
(work area). Once you decide to admit it, ``admit()`` moves the data to
``factor_library.duckdb`` and clears it from here. Until then it remains a
temporary research artefact.

Backfill 会按 registry 声明的所有变体(``parameters.neutralizations``)做 fan-out:
对每个声明的 ``(industry, cap)`` 组合应用中性化算子,各自连同 ``variant`` 列
入库。默认 2 变体(``raw`` + ``swl1_capq5``),可在 :func:`@register` 处覆盖。

Daily incremental refresh of *already-admitted* factors lives in
``update.py`` and writes to the library DB.

Usage:
    python -m backtest.factor.backfill f_001                   # single factor (all variants)
    python -m backtest.factor.backfill --pending               # all pending factors
    python -m backtest.factor.backfill f_001 --test-days 60    # last 60 trade days only
"""

from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.admission import get_pending_factor_ids
from backtest.factor.compute import apply_neutralizations, compute_factor
from backtest.factor.registry import get_factor_variants
from backtest.factor.storage import FactorStorage
from backtest.factor.variants import RAW_VARIANT


def backfill_factor(
    factor_id: str,
    start_date: str,
    end_date: str,
    *,
    market_storage: MarketStorage | None = None,
    factor_storage: FactorStorage | None = None,
) -> int:
    """Backfill a single factor (all declared variants) into the work DB.

    Returns total rows written across all variants.
    """
    raw_df = compute_factor(
        factor_id,
        start_date,
        end_date,
        market_storage=market_storage,
        factor_storage=factor_storage,
    )
    if raw_df.empty:
        return 0

    own_market = market_storage is None
    own_factor = factor_storage is None
    try:
        if market_storage is None:
            market_storage = MarketStorage()
        if factor_storage is None:
            factor_storage = FactorStorage()

        all_variants_df = apply_neutralizations(
            raw_df, factor_id, market_storage=market_storage,
        )
        if all_variants_df.empty:
            return 0
        factor_storage.insert_factors(all_variants_df)
        return len(all_variants_df)
    finally:
        if own_factor and factor_storage is not None:
            factor_storage.close()
        if own_market and market_storage is not None:
            market_storage.close()


def _get_earliest_start_date(stock_list: pd.DataFrame) -> str:
    return str(stock_list["list_date"].min())


def main():
    parser = argparse.ArgumentParser(description="Backfill factor values into the work DB")
    parser.add_argument("factor_id", nargs="?", help="Factor ID to backfill (e.g. f_001)")
    parser.add_argument("--pending", action="store_true",
                        help="Backfill all pending (unadmitted, unrejected) factors")
    parser.add_argument("--test-days", type=int, default=None,
                        help="Only backfill the last N trade days (debugging)")
    args = parser.parse_args()

    if not args.pending and not args.factor_id:
        parser.error("Specify a factor_id or --pending")

    stock_list = fetch_stock_list()
    earliest_date = _get_earliest_start_date(stock_list)

    with MarketStorage() as market_storage:
        latest_date = market_storage.get_max_date()
        if latest_date is None:
            print("market_daily is empty. Run cold_start first.")
            return

        if args.test_days:
            all_dates = get_trade_dates(earliest_date, latest_date)
            start_date = (all_dates[-args.test_days]
                          if len(all_dates) >= args.test_days else earliest_date)
            end_date = latest_date
        else:
            start_date = earliest_date
            end_date = latest_date

        print(f"Backfill range: {start_date} ~ {end_date} (work DB)")

        if args.factor_id:
            factor_ids = [args.factor_id]
        else:
            factor_ids = get_pending_factor_ids()

        if not factor_ids:
            print("No factors to backfill.")
            return

        print(f"Factors: {factor_ids}")

        with FactorStorage() as factor_storage:
            for factor_id in tqdm(factor_ids, desc="backfill"):
                try:
                    variants = get_factor_variants(factor_id)
                    # Resume cursor uses raw variant as the canonical "is this date covered?" signal.
                    existing_max = factor_storage.get_max_date(
                        factor_id, variant=RAW_VARIANT,
                    )
                    if existing_max and existing_max >= end_date:
                        print(f"  {factor_id}: already up to date ({existing_max})")
                        continue

                    if existing_max:
                        resume_dates = get_trade_dates(existing_max, end_date)
                        factor_start = resume_dates[1] if len(resume_dates) > 1 else end_date
                        print(f"  {factor_id}: resuming from {factor_start}")
                    else:
                        factor_start = start_date

                    rows = backfill_factor(
                        factor_id,
                        factor_start,
                        end_date,
                        market_storage=market_storage,
                        factor_storage=factor_storage,
                    )
                    print(
                        f"  {factor_id}: wrote {rows:,} rows total across "
                        f"{len(variants)} variants {variants}"
                    )
                except Exception as exc:
                    print(f"  ERROR {factor_id}: {exc}")
                    continue

        print("\nBackfill complete.")


if __name__ == "__main__":
    main()
