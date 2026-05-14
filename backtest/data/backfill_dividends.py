#!/usr/bin/env python3
"""
Backfill dividends: fetch full per-symbol history.
Skips symbols that already have data unless --force is given.

Only rows with div_proc='实施' are kept (filtered inside the fetcher).

Usage:
    python -m backtest.data.backfill_dividends
    python -m backtest.data.backfill_dividends --force
"""

import pandas as pd

from backtest.data._pipeline import backfill_by_symbol, run_symbol_backfill_cli
from backtest.data.dividends_fetcher import fetch_dividend_by_symbol
from backtest.data.storage import MarketStorage


def backfill_dividends(
    storage: MarketStorage,
    *,
    force: bool = False,
    stock_list: pd.DataFrame | None = None,
) -> None:
    backfill_by_symbol(
        label="dividends",
        fetch_one=fetch_dividend_by_symbol,
        insert=storage.insert_dividends,
        get_done=storage.get_symbols_in_dividends,
        force=force,
        stock_list=stock_list,
    )


def main():
    run_symbol_backfill_cli(
        "dividends",
        backfill_fn=backfill_dividends,
        get_stats=lambda s: s.get_dividend_stats(),
    )


if __name__ == "__main__":
    main()
