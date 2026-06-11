#!/usr/bin/env python3
"""Backfill stock opening/closing call-auction data.

Usage:
    python -m backtest.data.backfill.stock_auction --start 20240101 --end 20240131
    python -m backtest.data.backfill.stock_auction --sessions open
"""

from __future__ import annotations

import argparse
from datetime import datetime

from tqdm import tqdm

from backtest.data.fetcher.auction_fetcher import (
    fetch_stock_auction_close,
    fetch_stock_auction_open,
)
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates

AUCTION_SESSIONS = ("open", "close")


def _parse_sessions(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        sessions = [s.strip() for s in value.split(",") if s.strip()]
    else:
        sessions = list(value)
    invalid = [s for s in sessions if s not in AUCTION_SESSIONS]
    if invalid:
        raise ValueError(f"Unknown auction sessions: {invalid}")
    return sessions


def backfill_stock_auction(
    *,
    storage: MarketStorage | None = None,
    start: str,
    end: str | None = None,
    sessions: str | list[str] | tuple[str, ...] = AUCTION_SESSIONS,
) -> dict[str, int]:
    """Fetch and UPSERT auction rows for every trade date in ``[start, end]``."""
    end = end or datetime.today().strftime("%Y%m%d")
    session_list = _parse_sessions(sessions)
    owns_storage = storage is None
    store = storage or MarketStorage()
    counts = {session: 0 for session in session_list}

    try:
        trade_dates = get_trade_dates(start, end)
        for trade_date in tqdm(trade_dates, desc="stock_auction"):
            if "open" in session_list:
                df = fetch_stock_auction_open(trade_date)
                if not df.empty:
                    store.insert_stock_auction_open(df)
                    counts["open"] += len(df)
            if "close" in session_list:
                df = fetch_stock_auction_close(trade_date)
                if not df.empty:
                    store.insert_stock_auction_close(df)
                    counts["close"] += len(df)
    finally:
        if owns_storage:
            store.close()

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.data.backfill.stock_auction",
        description="Backfill stock_auction_open/close tables from Tushare.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date YYYYMMDD.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYYMMDD. Defaults to today.",
    )
    parser.add_argument(
        "--sessions",
        default="open,close",
        help="Comma-separated sessions: open,close. Default: open,close.",
    )
    args = parser.parse_args(argv)

    counts = backfill_stock_auction(
        start=args.start,
        end=args.end,
        sessions=args.sessions,
    )
    print(
        "stock_auction: "
        + ", ".join(f"{session}={rows:,}" for session, rows in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
