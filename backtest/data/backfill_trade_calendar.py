#!/usr/bin/env python3
"""Backfill trade calendar into DuckDB with pre-computed week/month boundary flags.

Usage:
    python -m backtest.data.backfill_trade_calendar
    python -m backtest.data.backfill_trade_calendar --start 20000101 --end 20241231
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.tushare_client import api_call, pro


def _compute_boundary_flags(trade_dates: list[str]) -> dict[str, set[str]]:
    """Compute week/month first/last flags from a sorted trade date list.

    Boundaries are based on the *trade date sequence*, not natural calendar.
    """
    week_first: set[str] = set()
    week_last: set[str] = set()
    month_first: set[str] = set()
    month_last: set[str] = set()

    for i, d in enumerate(trade_dates):
        dt = pd.Timestamp(d)
        if i == 0:
            week_first.add(d)
            month_first.add(d)
        else:
            prev_dt = pd.Timestamp(trade_dates[i - 1])
            if dt.isocalendar()[1] != prev_dt.isocalendar()[1]:
                week_first.add(d)
                week_last.add(trade_dates[i - 1])
            if dt.month != prev_dt.month:
                month_first.add(d)
                month_last.add(trade_dates[i - 1])

        if i == len(trade_dates) - 1:
            week_last.add(d)
            month_last.add(d)

    return {
        "week_first": week_first,
        "week_last": week_last,
        "month_first": month_first,
        "month_last": month_last,
    }


def backfill_trade_calendar(
    start_date: str = "20000101",
    end_date: str | None = None,
) -> int:
    """Fetch full calendar from Tushare, compute flags, UPSERT into trade_calendar.

    Parameters
    ----------
    start_date : str
        YYYYMMDD start date.
    end_date : str | None
        YYYYMMDD end date.  Defaults to today.

    Returns
    -------
    int
        Number of rows written.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # 1. Fetch all calendar dates (both open and closed) from Tushare
    df = api_call(pro.trade_cal, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return 0

    df = df.sort_values("cal_date").reset_index(drop=True)

    # 2. Compute boundary flags from the trade date sequence
    trade_dates = df[df["is_open"] == "1"]["cal_date"].tolist()
    flags = _compute_boundary_flags(trade_dates)

    # 3. Build the output DataFrame
    df["is_open"] = df["is_open"].astype(str) == "1"
    df["is_week_first"] = df["cal_date"].isin(flags["week_first"])
    df["is_week_last"] = df["cal_date"].isin(flags["week_last"])
    df["is_month_first"] = df["cal_date"].isin(flags["month_first"])
    df["is_month_last"] = df["cal_date"].isin(flags["month_last"])

    # cal_date → DATE type
    df["cal_date"] = pd.to_datetime(df["cal_date"]).dt.date

    out_df = df[[
        "cal_date", "is_open", "is_week_first",
        "is_week_last", "is_month_first", "is_month_last",
    ]]

    # 4. UPSERT
    with MarketStorage() as storage:
        storage.insert_trade_calendar(out_df)

    return len(out_df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.data.backfill_trade_calendar",
        description="Backfill trade_calendar table from Tushare pro.trade_cal.",
    )
    parser.add_argument(
        "--start", default="20000101",
        help="Start date YYYYMMDD (default: 20000101).",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYYMMDD (default: today).",
    )
    args = parser.parse_args(argv)

    end = args.end or datetime.now().strftime("%Y%m%d")
    print(f"Backfilling trade_calendar ({args.start} ~ {end}) ...")
    n = backfill_trade_calendar(start_date=args.start, end_date=end)
    print(f"  Inserted/updated {n:,} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
