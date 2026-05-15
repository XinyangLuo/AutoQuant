"""Built-in momentum factor: N-day return momentum."""

from __future__ import annotations

import pandas as pd

from backtest.factor.registry import register


@register(
    "f_001",
    name="momentum_20d",
    category="momentum",
    data_sources=["market_daily"],
    description="N日收益率动量因子，close / close.shift(window) - 1",
    parameters={"window": 20},
)
def momentum_20d(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute N-day momentum for each (date, symbol).

    Parameters
    ----------
    panel : pd.DataFrame
        Wide DataFrame from get_bars() with columns [date, symbol, close, ...].
    window : int
        Lookback window in trading days.

    Returns
    -------
    pd.Series
        MultiIndex (date, symbol) with momentum values.
    """
    if "close" not in panel.columns:
        raise ValueError("panel must contain 'close' column")

    df = panel[["date", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["momentum"] = df.groupby("symbol")["close"].shift(window)
    df["momentum"] = df["close"] / df["momentum"] - 1
    return df.set_index(["date", "symbol"])["momentum"]
