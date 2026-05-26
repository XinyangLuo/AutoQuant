#!/usr/bin/env python3
"""Backfill minute-level bars from Tushare into parquet partitions.

Storage layout: ``data/minute/{freq}/{symbol}/{year}.parquet``

Usage:
    python -m backtest.data.backfill.minute --symbol 000001.SZ --start 20250523 --end 20250523
    python -m backtest.data.backfill.minute --symbol 000001.SZ --start 20240101 --end 20250523 --freq 1min
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from backtest.data.fetcher.minute_fetcher import fetch_minute_bars
from backtest.data.tushare_client import _find_project_root


# ---------------------------------------------------------------------------
# Partition write helpers
# ---------------------------------------------------------------------------

def write_minute_partition(df: pd.DataFrame, freq: str, output_dir: Path) -> None:
    """Write DataFrame to symbol/year parquet partitions.

    If the target file already exists, merge and dedupe by (date, time),
    keeping the last occurrence.
    """
    if df.empty:
        return

    df = df.copy()
    df["_year"] = pd.to_datetime(df["date"]).dt.year

    for (symbol, year), group in df.groupby(["symbol", "_year"]):
        symbol_dir = output_dir / freq / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = symbol_dir / f"{year}.parquet"

        group = group.drop(columns=["_year"])
        group = group.sort_values(["date", "time"]).reset_index(drop=True)

        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            merged = pd.concat([existing, group], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date", "time"], keep="last")
            merged = merged.sort_values(["date", "time"]).reset_index(drop=True)
        else:
            merged = group

        merged.to_parquet(parquet_path, index=False)
        tqdm.write(f"  {symbol} {year}: {len(merged):,} rows → {parquet_path}")


# ---------------------------------------------------------------------------
# Backfill loop
# ---------------------------------------------------------------------------

def backfill_minute_bars(
    symbols: list[str],
    start_date: str,
    end_date: str,
    freq: str = "1min",
    output_dir: Path | None = None,
) -> None:
    """Fetch and write minute bars for the given symbols and date range."""
    if output_dir is None:
        output_dir = _find_project_root() / "data" / "minute"

    for symbol in tqdm(symbols, desc=f"Backfill {freq}"):
        try:
            df = fetch_minute_bars(symbol, start_date, end_date, freq=freq)
        except Exception as exc:
            tqdm.write(f"  {symbol}: fetch failed ({exc})")
            continue

        if df.empty:
            tqdm.write(f"  {symbol}: no data")
            continue

        tqdm.write(f"  {symbol}: fetched {len(df):,} rows ({df['date'].min()} ~ {df['date'].max()})")
        write_minute_partition(df, freq, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill minute bars from Tushare")
    parser.add_argument("--symbol", "-s", required=True, help="Comma-separated ts_codes")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument(
        "--freq",
        default="1min",
        choices=["1min", "5min", "15min", "30min", "60min"],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
    backfill_minute_bars(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
