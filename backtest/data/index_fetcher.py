"""Tushare ``pro.index_daily`` wrapper.

Fetches index OHLCV bars (e.g. 000300.SH, 000905.SH) for use as backtest
benchmarks.  Output is normalised to the ``index_daily`` schema in
``backtest/data/storage.py``.
"""

from __future__ import annotations

import pandas as pd

from backtest.data.tushare_client import api_call, pro


_RAW_COLS = [
    "ts_code", "trade_date",
    "close", "open", "high", "low",
    "pre_close", "change", "pct_chg",
    "vol", "amount",
]


def fetch_index_daily(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch one index's daily OHLCV from Tushare.

    Parameters
    ----------
    symbol : str
        Tushare ts_code, e.g. ``"000300.SH"``.
    start, end : str | None
        YYYYMMDD inclusive bounds. Tushare interprets None as "all history".

    Returns
    -------
    pd.DataFrame
        Columns matching ``INDEX_DAILY_COLUMNS`` in ``storage.py``:
        ``date, symbol, open, high, low, close, pre_close, change, pct_chg, volume, amount``.
        Empty DF if no data.
    """
    kwargs = {"ts_code": symbol}
    if start:
        kwargs["start_date"] = start
    if end:
        kwargs["end_date"] = end

    df = api_call(pro.index_daily, **kwargs)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={
        "ts_code": "symbol",
        "trade_date": "date",
        "vol": "volume",
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date

    cols = ["date", "symbol", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "volume", "amount"]
    return df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)
