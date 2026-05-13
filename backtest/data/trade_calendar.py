"""Trade calendar operations."""

import pandas as pd

from backtest.data.tushare_client import api_call, pro


def fetch_trade_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch trade calendar (open days only) for a date range."""
    df = api_call(
        pro.trade_cal,
        start_date=start_date,
        end_date=end_date,
        is_open="1"
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_values("cal_date").reset_index(drop=True)


def get_trade_dates(start_date: str, end_date: str) -> list[str]:
    """Return sorted list of trade dates as YYYYMMDD strings."""
    df = fetch_trade_calendar(start_date, end_date)
    if df.empty:
        return []
    return df["cal_date"].tolist()
