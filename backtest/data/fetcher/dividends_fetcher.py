"""Tushare pro.dividend fetcher.

Only `div_proc = '实施'` rows are kept (the implemented dividend, not the draft
/ board / shareholder-meeting stages). In the rare case a stock has multiple
implemented rounds for the same end_date, the latest announcement wins.
"""

import pandas as pd

from backtest.data.storage import DIVIDEND_COLUMNS
from backtest.data.tushare_client import fetch_and_transform, pro


_IMPLEMENTED = "实施"


def _clean_dividend(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["div_proc"] == _IMPLEMENTED]
    if df.empty:
        return df

    df = df.rename(columns={"ts_code": "symbol"}).dropna(subset=["symbol", "end_date"])
    if df.empty:
        return df

    df = df[[c for c in DIVIDEND_COLUMNS if c in df.columns]].copy()

    # Fill missing dates for PK stability.  Tushare occasionally returns NULL
    # ex_date / pay_date for older records; falling back preserves the event
    # rather than silently dropping it.
    df["ann_date"] = df["ann_date"].fillna(df["end_date"])
    df["ex_date"] = df["ex_date"].fillna(df["pay_date"]).fillna(df["ann_date"])
    df["pay_date"] = df["pay_date"].fillna(df["ex_date"]).fillna(df["ann_date"])

    # De-duplicate on the 4-column PK.  Multiple dividends for the same
    # reporting period (e.g. regular + special dividend) are preserved as
    # long as ann_date or ex_date differs.
    return df.drop_duplicates(
        subset=["symbol", "end_date", "ann_date", "ex_date"], keep="last"
    )


def fetch_dividend_by_symbol(symbol: str) -> pd.DataFrame:
    """Fetch the full dividend history for one stock."""
    return fetch_and_transform(pro.dividend, _clean_dividend, ts_code=symbol)


def fetch_dividend_by_ann_date(ann_date: str) -> pd.DataFrame:
    """Fetch all dividend rows announced on a given date (YYYYMMDD)."""
    return fetch_and_transform(pro.dividend, _clean_dividend, ann_date=ann_date)
