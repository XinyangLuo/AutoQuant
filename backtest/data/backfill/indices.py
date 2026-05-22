#!/usr/bin/env python3
"""Backfill benchmark indices' daily OHLCV into market.duckdb / index_daily.

Usage:
    python -m backtest.data.backfill_indices                          # default indices
    python -m backtest.data.backfill_indices --symbols 000300.SH,000905.SH
    python -m backtest.data.backfill_indices --start 20100101         # force-refresh from a date
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from tqdm import tqdm

from backtest.data.fetcher.index_fetcher import fetch_index_daily
from backtest.data.storage import MarketStorage


DEFAULT_INDICES = [
    "000300.SH",  # CSI 300       (2005-04-08 -)
    "000905.SH",  # CSI 500       (2007-01-15 -)
    "000852.SH",  # CSI 1000      (2014-10-17 -)
    "932000.CSI", # CSI 2000      (2023-08-11 -)
    "000001.SH",  # SSE Composite
    "399006.SZ",  # ChiNext
]


def _next_day(yyyymmdd: str) -> str:
    return (datetime.strptime(yyyymmdd, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


def backfill_indices(symbols: list[str], start_override: str | None = None) -> None:
    today = datetime.today().strftime("%Y%m%d")
    with MarketStorage() as storage:
        for symbol in tqdm(symbols, desc="Backfill index_daily"):
            if start_override:
                start = start_override
            else:
                max_d = storage.get_max_index_date(symbol)
                start = _next_day(max_d) if max_d else None

            if start and start > today:
                tqdm.write(f"  {symbol}: already up to date")
                continue

            try:
                df = fetch_index_daily(symbol, start=start, end=today)
            except Exception as exc:
                tqdm.write(f"  {symbol}: fetch failed ({exc})")
                continue

            if df.empty:
                tqdm.write(f"  {symbol}: no new rows")
                continue

            storage.insert_index_daily(df)
            tqdm.write(f"  {symbol}: inserted {len(df):,} rows ({df['date'].min()} ~ {df['date'].max()})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.data.backfill_indices",
        description="Backfill index_daily table from Tushare pro.index_daily.",
    )
    parser.add_argument(
        "--symbols", "-s",
        default=",".join(DEFAULT_INDICES),
        help=f"Comma-separated ts_codes. Default: {','.join(DEFAULT_INDICES)}",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Force start date YYYYMMDD (overrides per-symbol max date).",
    )
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("at least one symbol required")
    backfill_indices(symbols, start_override=args.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
