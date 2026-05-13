"""Stock list operations."""

import pandas as pd

from backtest.data.tushare_client import api_call, pro


def fetch_stock_list() -> pd.DataFrame:
    """Fetch all stocks (listed, delisted, suspended) from Tushare."""
    df = api_call(
        pro.stock_basic,
        exchange="",
        list_status="",
        fields="ts_code,symbol,name,list_date,delist_date,exchange,market"
    )
    if df is None or df.empty:
        raise RuntimeError("Failed to fetch stock list")

    df = df[df["list_date"].notna() & (df["list_date"] != "")].copy()
    df["list_date"] = df["list_date"].astype(str)
    return df
