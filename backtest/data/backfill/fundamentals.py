#!/usr/bin/env python3
"""
Backfill income_q / balancesheet_q / cashflow_q: fetch full per-symbol history.
Skips symbols that already have data in *all three* tables unless --force is given.

The three tables are fetched sequentially for each symbol to respect Tushare
rate limits.  If any of the three fetches fails, the whole symbol is skipped
so that the three tables stay in sync.

Usage:
    python -m backtest.data.backfill_fundamentals
    python -m backtest.data.backfill_fundamentals --force
"""

import argparse

import pandas as pd
from tqdm import tqdm

from backtest.data.fetcher.fundamentals_fetcher import (
    fetch_balancesheet_by_symbol,
    fetch_cashflow_by_symbol,
    fetch_income_by_symbol,
)
from backtest.data._pipeline import print_stats
from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage


def _get_done_union(storage: MarketStorage) -> set[str]:
    """Symbols already present in all three tables."""
    s_inc = storage.get_symbols_in_fundamentals("income_q")
    s_bs = storage.get_symbols_in_fundamentals("balancesheet_q")
    s_cf = storage.get_symbols_in_fundamentals("cashflow_q")
    return s_inc & s_bs & s_cf


def backfill_fundamentals(
    storage: MarketStorage,
    *,
    force: bool = False,
    stock_list: pd.DataFrame | None = None,
) -> None:
    if stock_list is None:
        stock_list = fetch_stock_list()
    all_symbols = sorted(stock_list["ts_code"].tolist())

    if force:
        todo = all_symbols
    else:
        done = _get_done_union(storage)
        todo = [s for s in all_symbols if s not in done]
        print(f"fundamentals: {len(done)} symbols already have all three tables, "
              f"{len(todo)} to fetch")

    if not todo:
        print("fundamentals: nothing to do.")
        return

    failed: list[tuple[str, str]] = []
    for symbol in tqdm(todo, desc="fundamentals"):
        try:
            inc = fetch_income_by_symbol(symbol)
            bs = fetch_balancesheet_by_symbol(symbol)
            cf = fetch_cashflow_by_symbol(symbol)
        except Exception as exc:
            failed.append((symbol, str(exc)))
            print(f"\n  WARN: failed {symbol}: {exc}")
            continue

        if not inc.empty:
            storage.insert_income(inc)
        if not bs.empty:
            storage.insert_balancesheet(bs)
        if not cf.empty:
            storage.insert_cashflow(cf)

    if failed:
        head = [s for s, _ in failed][:20]
        tail = " ..." if len(failed) > 20 else ""
        print(f"\n  Failed symbols ({len(failed)}): {head}{tail}")


def main():
    parser = argparse.ArgumentParser(description="Backfill income / balancesheet / cashflow")
    parser.add_argument("--force", action="store_true",
                        help="Refetch even symbols already in all three tables")
    args = parser.parse_args()

    with MarketStorage() as storage:
        backfill_fundamentals(storage, force=args.force)

        for name in ("income_q", "balancesheet_q", "cashflow_q"):
            print_stats(name, storage.get_fundamentals_stats(name), date_col="f_ann_date", prefix="f_ann ")

    print("\n" + "=" * 50)
    print("Fundamentals backfill complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
