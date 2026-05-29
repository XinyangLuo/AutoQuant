"""Trade calendar operations — DuckDB-first with Tushare fallback."""

from __future__ import annotations

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.tushare_client import api_call, pro


def fetch_trade_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch trade calendar (open days only) for a date range from Tushare."""
    df = api_call(
        pro.trade_cal,
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_values("cal_date").reset_index(drop=True)


def _fetch_and_write(start: str, end: str) -> list[str]:
    """Fallback: fetch from Tushare, write to DB, return trade dates."""
    df = fetch_trade_calendar(start, end)
    if df.empty:
        return []

    # Build full calendar (open + closed) for the range so flags are correct
    full_df = api_call(pro.trade_cal, start_date=start, end_date=end)
    if full_df is not None and not full_df.empty:
        full_df = full_df.sort_values("cal_date").reset_index(drop=True)
        trade_dates = full_df[full_df["is_open"] == "1"]["cal_date"].tolist()

        from backtest.data.backfill_trade_calendar import _compute_boundary_flags
        flags = _compute_boundary_flags(trade_dates)

        full_df["is_open"] = full_df["is_open"].astype(str) == "1"
        full_df["is_week_first"] = full_df["cal_date"].isin(flags["week_first"])
        full_df["is_week_last"] = full_df["cal_date"].isin(flags["week_last"])
        full_df["is_month_first"] = full_df["cal_date"].isin(flags["month_first"])
        full_df["is_month_last"] = full_df["cal_date"].isin(flags["month_last"])
        full_df["cal_date"] = pd.to_datetime(full_df["cal_date"]).dt.date

        with MarketStorage() as storage:
            storage.insert_trade_calendar(full_df[[
                "cal_date", "is_open", "is_week_first",
                "is_week_last", "is_month_first", "is_month_last",
            ]])

    return df["cal_date"].tolist()


def get_trade_dates(start: str, end: str) -> list[str]:
    """Return sorted list of trade dates as YYYYMMDD strings.

    DuckDB-first: reads ``trade_calendar`` table.  If the range is not fully
    covered, falls back to ``pro.trade_cal`` and writes the fetched rows back
    so subsequent calls are DB-only.
    """
    with MarketStorage(read_only=True) as storage:
        dates = storage.get_trade_dates_from_db(start, end)

    # If DB returned nothing or partial, fallback to Tushare
    if not dates:
        dates = _fetch_and_write(start, end)

    return dates


def get_rebalance_dates(start: str, end: str, freq: str) -> list[str]:
    """Generate rebalancing trade dates from a frequency code.

    Parameters
    ----------
    start, end : str
        YYYYMMDD bounds.
    freq : str
        ``"1D"`` — all trade dates.
        ``"5D"`` — every 5th trade date.
        ``"1W"`` — first trade day of each ISO week.
        ``"2W"`` — first trade day of every other ISO week (even week numbers).
        ``"1M"`` — first trade day of each month.
        ``"EOM"`` — last trade day of each month.

    Returns
    -------
    list[str]
        Sorted list of YYYYMMDD rebalancing dates.
    """
    dates: list[str] = []
    with MarketStorage(read_only=True) as storage:
        dates = storage.get_rebalance_dates_from_db(start, end, freq)

    if not dates:
        _fetch_and_write(start, end)
        with MarketStorage(read_only=True) as storage:
            dates = storage.get_rebalance_dates_from_db(start, end, freq)

    return dates
