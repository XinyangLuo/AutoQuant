"""Tushare stock call-auction fetchers."""

from __future__ import annotations

import pandas as pd

from backtest.data.tushare_client import api_call, pro


AUCTION_COLUMNS = [
    "date", "symbol", "open", "high", "low", "close",
    "volume", "amount", "vwap",
]


def _fetch_auction(api_func, trade_date: str) -> pd.DataFrame:
    df = api_call(api_func, trade_date=trade_date)
    if df is None or df.empty:
        return pd.DataFrame(columns=AUCTION_COLUMNS)
    return transform_stock_auction(df)


def fetch_stock_auction_open(trade_date: str) -> pd.DataFrame:
    """Fetch opening call-auction rows for one trade date."""
    return _fetch_auction(pro.stk_auction_o, trade_date)


def fetch_stock_auction_close(trade_date: str) -> pd.DataFrame:
    """Fetch closing call-auction rows for one trade date."""
    return _fetch_auction(pro.stk_auction_c, trade_date)


def transform_stock_auction(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Tushare stk_auction_o/c output to internal schema.

    Tushare returns ``vol`` in shares and ``amount`` in yuan for auction data.
    """
    if df.empty:
        return pd.DataFrame(columns=AUCTION_COLUMNS)

    out = df.rename(
        columns={
            "trade_date": "date",
            "ts_code": "symbol",
            "vol": "volume",
        }
    ).copy()
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d").dt.date

    for col in AUCTION_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[AUCTION_COLUMNS]
