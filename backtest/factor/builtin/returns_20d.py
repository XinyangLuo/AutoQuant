"""20-day cumulative return factor."""

from __future__ import annotations

import pandas as pd

from backtest.factor.registry import register


@register(
    "f_002",
    name="returns_20d",
    category="momentum",
    data_sources=["market_daily"],
    description="20日累计收益率因子，基于复权价格计算",
    parameters={"window": 20},
)
def returns_20d(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute N-day cumulative return from adjusted close prices.

    Uses close * adj_factor as the adjusted price to account for
    splits, dividends, and other corporate actions.

    Parameters
    ----------
    panel : pd.DataFrame
        Wide DataFrame with columns [date, symbol, close, adj_factor].
    window : int
        Lookback window in trading days.

    Returns
    -------
    pd.Series
        MultiIndex (date, symbol) with cumulative return values.
    """
    if "close" not in panel.columns:
        raise ValueError("panel must contain 'close' column")

    df = panel[["date", "symbol", "close"]].copy()
    if "adj_factor" in panel.columns:
        df["adj_close"] = df["close"] * panel["adj_factor"]
    else:
        df["adj_close"] = df["close"]

    df = df.sort_values(["symbol", "date"])
    df["returns"] = df.groupby("symbol")["adj_close"].shift(window)
    df["returns"] = df["adj_close"] / df["returns"] - 1
    return df.set_index(["date", "symbol"])["returns"]
