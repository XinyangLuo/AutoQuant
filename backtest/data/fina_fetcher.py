"""Tushare pro.fina_indicator fetcher.

Dedup note: Tushare may return 2 rows for the same (ts_code, end_date) — one
full, one partially NaN. We keep the row with fewer NaNs.
"""

import pandas as pd

from backtest.data.tushare_client import fetch_and_transform, pro


def _clean_fina(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ts_code", "end_date"])
    if df.empty:
        return df
    df = df.assign(_nan=df.isna().sum(axis=1))
    return (
        df.sort_values("_nan", kind="stable")
        .drop_duplicates(subset=["ts_code", "end_date"], keep="first")
        .drop(columns="_nan")
        .rename(columns={"ts_code": "symbol"})
    )


def fetch_fina_by_symbol(symbol: str) -> pd.DataFrame:
    """Fetch the full fina_indicator history for one stock."""
    return fetch_and_transform(pro.fina_indicator, _clean_fina, ts_code=symbol)


def fetch_fina_by_ann_date(ann_date: str) -> pd.DataFrame:
    """Fetch all fina_indicator rows announced on a given date (YYYYMMDD)."""
    return fetch_and_transform(pro.fina_indicator, _clean_fina, ann_date=ann_date)
