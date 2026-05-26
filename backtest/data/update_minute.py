#!/usr/bin/env python3
"""Incremental update for minute-level bars.

Scans existing parquet partitions and fetches missing trade dates per symbol.

Usage:
    python -m backtest.data.update_minute --symbol 000001.SZ
    python -m backtest.data.update_minute --freq 5min
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from backtest.data.backfill.minute import write_minute_partition
from backtest.data.fetcher.minute_fetcher import fetch_minute_bars
from backtest.data.stock_list import fetch_stock_list
from backtest.data.trade_calendar import get_trade_dates
from backtest.data.tushare_client import _find_project_root


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _get_max_minute_date(symbol: str, freq: str, output_dir: Path) -> str | None:
    """Return the max date in existing parquet for a symbol, as YYYYMMDD, or None."""
    symbol_dir = output_dir / freq / symbol
    if not symbol_dir.exists():
        return None

    max_date: str | None = None
    for parquet_path in symbol_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet_path, columns=["date"])
        if df.empty:
            continue
        raw_max = df["date"].max()
        if pd.isna(raw_max):
            continue
        if hasattr(raw_max, "strftime"):
            candidate = raw_max.strftime("%Y%m%d")
        else:
            candidate = str(raw_max).replace("-", "")[:8]
        if max_date is None or candidate > max_date:
            max_date = candidate

    return max_date


# ---------------------------------------------------------------------------
# Update loop
# ---------------------------------------------------------------------------

def update_minute_bars(
    symbols: list[str],
    freq: str = "1min",
    output_dir: Path | None = None,
) -> None:
    """Fetch missing minute bars for each symbol up to today."""
    if output_dir is None:
        output_dir = _find_project_root() / "data" / "minute"

    for symbol in tqdm(symbols, desc=f"Update {freq}"):
        today = datetime.now().strftime("%Y%m%d")
        max_date = _get_max_minute_date(symbol, freq, output_dir)
        if max_date is None:
            tqdm.write(f"  {symbol}: no existing data, run backfill first")
            continue

        # Use datetime comparison to avoid string-comparison fragility
        if datetime.strptime(max_date, "%Y%m%d") >= datetime.strptime(today, "%Y%m%d"):
            tqdm.write(f"  {symbol}: already up to date (last {max_date})")
            continue

        # Re-fetch from boundary day so incomplete boundary rows are backfilled.
        # write_minute_partition dedupes by (date, time) keeping last → idempotent.
        start = max_date

        trade_dates = get_trade_dates(start, today)
        if not trade_dates:
            tqdm.write(f"  {symbol}: no new trade dates")
            continue

        tqdm.write(f"  {symbol}: {len(trade_dates)} dates to update ({start} ~ {today})")

        try:
            df = fetch_minute_bars(symbol, start, today, freq=freq)
        except Exception as exc:
            tqdm.write(f"  {symbol}: fetch failed ({exc})")
            continue

        if df.empty:
            tqdm.write(f"  {symbol}: no data returned")
            continue

        tqdm.write(f"  {symbol}: fetched {len(df):,} rows ({df['date'].min()} ~ {df['date'].max()})")
        write_minute_partition(df, freq, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Incremental update for minute bars")
    parser.add_argument("--symbol", "-s", help="Comma-separated ts_codes (default: all stocks)")
    parser.add_argument(
        "--freq",
        default="1min",
        choices=["1min", "5min", "15min", "30min", "60min"],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.symbol:
        symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
    else:
        stock_list = fetch_stock_list()
        symbols = sorted(stock_list["ts_code"].tolist())
        print(f"Stock list: {len(symbols)} stocks")

    update_minute_bars(
        symbols=symbols,
        freq=args.freq,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
