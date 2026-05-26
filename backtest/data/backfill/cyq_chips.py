#!/usr/bin/env python3
"""Backfill cyq_chips (筹码分布) from Tushare into DuckDB LIST format.

Tushare ``pro.cyq_chips`` requires ``ts_code`` and supports ``start_date`` /
``end_date``.  Empirically the API caps at ~6 000 rows (~35 trade days for
high-bin stocks), so the backfill walks each symbol in trade-day chunks.

Usage::

    python -m backtest.data.backfill.cyq_chips --start 20240101 --end 20250526
    python -m backtest.data.backfill.cyq_chips --symbol 600519.SH --start 20240101 --end 20250526
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from backtest.data.cyq_storage import CyqStorage
from backtest.data.fetcher.cyq_fetcher import fetch_cyq_for_symbol_range
from backtest.data.stock_list import fetch_stock_list
from backtest.data.trade_calendar import get_trade_dates


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

_CHUNK_DAYS = 30  # trade days per API call — under the ~6 000 row cap


def _trade_date_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into inclusive (chunk_start, chunk_end) pairs."""
    dates = get_trade_dates(start, end)
    if not dates:
        return []
    chunks: list[tuple[str, str]] = []
    for i in range(0, len(dates), _CHUNK_DAYS):
        chunk_dates = dates[i : i + _CHUNK_DAYS]
        chunks.append((chunk_dates[0], chunk_dates[-1]))
    return chunks


# ---------------------------------------------------------------------------
# Backfill core
# ---------------------------------------------------------------------------

def backfill_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    storage: CyqStorage,
    sleep_sec: float = 0.05,
) -> int:
    """Backfill one symbol across [start_date, end_date].

    Returns packed row count (one per (date, symbol) inserted).
    """
    chunks = _trade_date_chunks(start_date, end_date)
    if not chunks:
        return 0

    total_packed = 0
    for chunk_start, chunk_end in chunks:
        if sleep_sec > 0:
            import time
            time.sleep(sleep_sec)

        try:
            df_list = fetch_cyq_for_symbol_range(symbol, chunk_start, chunk_end)
        except Exception as exc:
            tqdm.write(f"  {symbol} {chunk_start}~{chunk_end}: fetch failed ({exc})")
            continue

        for df in df_list:
            if df.empty:
                continue
            storage.insert_cyq(df)
            total_packed += df["date"].nunique()

    return total_packed


def backfill_cyq_chips(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    storage: CyqStorage | None = None,
    sleep_sec: float = 0.05,
) -> int:
    """Backfill cyq_chips for all symbols in the given range.

    Parameters
    ----------
    symbols : list[str] | None
        If None, fetches the full stock list.
    start_date : str | None
        YYYYMMDD.  If None, backfills from ``storage.get_max_date() + 1``
        (or ``20200101`` if the DB is empty).
    end_date : str | None
        YYYYMMDD.  If None, uses today.
    storage : CyqStorage | None
        Reuse an existing storage instance.
    sleep_sec : float
        Pause between API calls to respect Tushare rate limits.

    Returns
    -------
    int
        Total long-format rows inserted.
    """
    # Resolve date bounds
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if symbols is None:
        stock_list = fetch_stock_list()
        symbols = stock_list["ts_code"].tolist()

    # Resolve start_date using DB watermark
    close_storage = False
    if storage is None:
        storage = CyqStorage()
        close_storage = True

    try:
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        if start_date is None:
            max_date = storage.get_max_date()
            if max_date:
                start_date = (
                    datetime.strptime(max_date, "%Y%m%d") + timedelta(days=1)
                ).strftime("%Y%m%d")
            else:
                start_date = "20100101"  # earliest A-share data reasonable default

        start_dt = datetime.strptime(start_date, "%Y%m%d")
        if start_dt > end_dt:
            print(f"cyq_chips: already up to date (last {storage.get_max_date()}).")
            return 0

        chunks = _trade_date_chunks(start_date, end_date)
        if not chunks:
            print(f"cyq_chips: no trade dates in {start_date} ~ {end_date}.")
            return 0

        n_trade_dates = sum(
            len(get_trade_dates(s, e)) for s, e in chunks
        )
        print(
            f"cyq_chips: {len(symbols)} symbols, "
            f"{n_trade_dates} trade days ({start_date} ~ {end_date}), "
            f"{len(chunks)} chunks/symbol"
        )

        total_packed = 0
        for symbol in tqdm(symbols, desc="cyq_chips backfill"):
            n_packed = backfill_symbol(
                symbol, start_date, end_date, storage, sleep_sec=sleep_sec
            )
            if n_packed:
                total_packed += n_packed

        stats = storage.get_stats()
        print(
            f"cyq_chips: {total_packed} packed rows inserted "
            f"({stats['total_rows']:,} total in DB), "
            f"{stats['total_symbols']} symbols, "
            f"{stats['min_date']} ~ {stats['max_date']}"
        )
        return total_packed
    finally:
        if close_storage:
            storage.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbol", "-s", help="Comma-separated ts_codes (default: all stocks)"
    )
    parser.add_argument(
        "--start",
        help="Start date YYYYMMDD (default: MAX(date)+1 from DB, or 20200101)",
    )
    parser.add_argument(
        "--end", help="End date YYYYMMDD (default: today)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between API calls (default: 0.05; high-tier 5000+ pts)",

    )
    args = parser.parse_args(argv)

    symbols = None
    if args.symbol:
        symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]

    backfill_cyq_chips(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        sleep_sec=args.sleep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
