#!/usr/bin/env python3
"""
Backfill fina_indicator_quarterly: fetch full per-symbol history.
Skips symbols that already have data unless --force is given.

Usage:
    python -m backtest.data.backfill_fina_indicator
    python -m backtest.data.backfill_fina_indicator --force
"""

import pandas as pd

from backtest.data._pipeline import backfill_by_symbol, run_symbol_backfill_cli
from backtest.data.fina_fetcher import fetch_fina_by_symbol
from backtest.data.storage import MarketStorage


def backfill_fina(
    storage: MarketStorage,
    *,
    force: bool = False,
    stock_list: pd.DataFrame | None = None,
) -> None:
    backfill_by_symbol(
        label="fina_indicator",
        fetch_one=fetch_fina_by_symbol,
        insert=storage.insert_fina,
        get_done=storage.get_symbols_in_fina,
        force=force,
        stock_list=stock_list,
    )


def main():
    run_symbol_backfill_cli(
        "fina_indicator",
        backfill_fn=backfill_fina,
        get_stats=lambda s: s.get_fina_stats(),
    )


if __name__ == "__main__":
    main()
