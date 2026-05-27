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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_CHUNK_DAYS = 25  # trade days per API call — max observed 195 bins/day; 25*240=6000 safe


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
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe rate limiter — caps API calls per minute across all threads."""

    def __init__(self, max_per_minute: int) -> None:
        self._min_interval = 60.0 / max_per_minute
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Backfill core
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("rate", "limit", "throttle", "频次", "频率", "too many", "429"))


# Exception types that indicate programmer error, not transient failures.
# These should never be retried.
_NON_RETRYABLE = frozenset({
    TypeError, AttributeError, NameError, SyntaxError,
    ImportError, KeyError, IndexError, ValueError,
})


def _retry_fetch(
    symbol: str,
    chunk_start: str,
    chunk_end: str,
    max_retries: int = 3,
) -> list[pd.DataFrame]:
    """Fetch with retry on transient errors (rate-limit, connection)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fetch_cyq_for_symbol_range(symbol, chunk_start, chunk_end)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries or type(exc) in _NON_RETRYABLE:
                raise
            wait = (5 * (2 ** attempt)) if _is_rate_limit_error(exc) else (2 * (attempt + 1))
            tqdm.write(
                f"  {symbol} {chunk_start}~{chunk_end}: "
                f"attempt {attempt + 1} failed ({exc}), retrying in {wait}s..."
            )
            time.sleep(wait)

    tqdm.write(f"  {symbol} {chunk_start}~{chunk_end}: all retries exhausted ({last_exc})")
    return [pd.DataFrame(columns=["date", "symbol", "price", "percent"])]


def backfill_cyq_chips(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    storage: CyqStorage | None = None,
    sleep_sec: float = 0.01,
    max_workers: int = 4,
) -> int:
    """Backfill cyq_chips for all symbols in the given range.

    Uses multi-threaded parallel fetching with a global rate limiter.

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
        Ignored in parallel mode (rate limiter handles pacing).
    max_workers : int
        Number of parallel threads.  Default 6.

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
                start_date = "20100101"

        start_dt = datetime.strptime(start_date, "%Y%m%d")
        if start_dt > end_dt:
            print(f"cyq_chips: already up to date (last {storage.get_max_date()}).")
            return 0

        chunks = _trade_date_chunks(start_date, end_date)
        if not chunks:
            print(f"cyq_chips: no trade dates in {start_date} ~ {end_date}.")
            return 0

        n_trade_dates = sum(len(get_trade_dates(s, e)) for s, e in chunks)
        print(
            f"cyq_chips: {len(symbols)} symbols, "
            f"{n_trade_dates} trade days ({start_date} ~ {end_date}), "
            f"{len(chunks)} chunks/symbol, "
            f"{max_workers} workers"
        )

        # --- Parallel fetch with rate limiter & serialized writes ---
        rate_limiter = RateLimiter(max_per_minute=900)  # 90% of 1000 limit
        write_lock = threading.Lock()
        pbar_lock = threading.Lock()
        pbar = tqdm(total=len(symbols), desc="cyq_chips backfill")
        total_packed = 0

        def _process_symbol(sym: str) -> int:
            packed = 0
            try:
                for cs, ce in chunks:
                    rate_limiter.acquire()
                    df_list = _retry_fetch(sym, cs, ce)
                    with write_lock:
                        for df in df_list:
                            if df.empty:
                                continue
                            storage.insert_cyq(df)
                            packed += df["date"].nunique()
            finally:
                with pbar_lock:
                    pbar.update(1)
            return packed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_symbol, s): s for s in symbols}
            for future in as_completed(futures):
                try:
                    n = future.result()
                    if n:
                        total_packed += n
                except Exception as exc:
                    sym = futures[future]
                    tqdm.write(f"  {sym}: worker crashed ({exc})")

        pbar.close()

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
        default=0.01,
        help="Seconds to sleep between API calls (default: 0.01; natural rate ~700/min, under 1000 limit)",

    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel fetch threads (default: 4)",
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
        max_workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
