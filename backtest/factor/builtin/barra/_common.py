"""Shared helpers for Barra factor implementations.

Kept private to ``backtest.factor.builtin.barra`` — these are not part of the
public factor API. If something here is needed by user factors, promote it
to ``backtest.factor.transforms``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.transforms import (
    cs_mad_winsorize,
    cs_zscore,
    industry_median_fill,
)


def apply_l3_pipeline(
    raw_series: pd.Series,
    market_storage: MarketStorage,
    *,
    start: str,
    end: str,
) -> pd.Series:
    """The CNE6 L3 style-exposure pipeline.

    ``MAD winsorize (k=3) → SW-L1 industry median fill → cs_zscore``.
    Both ``apply_variant_pipeline`` (for any factor with ``variant='barra_l3'``)
    and the Barra L1 composites call this on each L3 sub-component before
    averaging — keeping the math in one place.
    """
    industry_panel = market_storage.get_industry_panel_range(
        start=start, end=end, level="L1",
    )
    series = cs_mad_winsorize(raw_series, k=3.0)
    series = industry_median_fill(series, industry_panel)
    series = cs_zscore(series)
    return series


def to_panel_series(df: pd.DataFrame, values, name: str) -> pd.Series:
    """Build a ``(date, symbol)``-indexed Series from a frame plus a value column.

    ``df`` must have ``date`` and ``symbol`` columns. ``values`` can be a
    column name or any array-like aligned with ``df``.
    """
    if isinstance(values, str):
        values = df[values].values
    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]], names=["date", "symbol"])
    return pd.Series(values, index=idx, name=name)


def latest_quarter_per_day(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent ``end_date`` row per ``(date, symbol)``.

    ``compute_factor`` concatenates one PIT snapshot per trade date, so every
    ``(date, symbol)`` carries the full quarter history visible on that day.
    This collapses to one row by picking the max ``end_date``.
    """
    df = panel.dropna(subset=["end_date"]).copy()
    df["end_date"] = df["end_date"].astype(str)
    keep = df.groupby(["date", "symbol"])["end_date"].idxmax()
    return df.loc[keep]


def regress_slope_over_mean(values: np.ndarray) -> float:
    """Slope of ``values`` vs integer time index, scaled by ``|mean(values)|``.

    NaN if fewer than 4 valid points or if time variation is degenerate.
    Used by AGRO/EGRO.
    """
    mask = ~np.isnan(values)
    if mask.sum() < 4:
        return np.nan
    y = values[mask]
    x = np.arange(values.size, dtype=float)[mask]
    if np.std(x) == 0:
        return np.nan
    cov = np.cov(x, y, bias=True)[0, 1]
    var = np.var(x)
    if var <= 0:
        return np.nan
    slope = cov / var
    mean = np.mean(y)
    if mean == 0 or np.isnan(slope):
        return np.nan
    return slope / abs(mean)


def pit_quarterly_slope(
    panel: pd.DataFrame,
    value_col: str,
    *,
    n: int = 20,
    sign: float = 1.0,
) -> pd.DataFrame:
    """PIT-safe per-trade-date slope-over-mean regression on quarterly history.

    For each ``(symbol, trade_date)`` row in ``panel`` (which carries that day's
    visible quarter history), regress the trailing ``n`` quarters of
    ``value_col`` on integer time, scale by ``|mean|``, multiply by ``sign``.

    Rows within each ``(symbol, date)`` group are assumed already sorted by
    ``end_date`` ascending — ``compute_factor`` produces them in that order
    and ``get_fina_snapshot`` guarantees uniqueness of ``end_date`` per group.

    Returns a frame with columns ``[date, symbol, value]`` carrying one row
    per ``(date, symbol)``.
    """
    df = panel.dropna(subset=[value_col, "end_date"]).copy()
    df["end_date"] = df["end_date"].astype(str)
    df = df.sort_values(["symbol", "date", "end_date"])

    def _score(arr: np.ndarray) -> float:
        return regress_slope_over_mean(arr[-n:]) * sign

    grouped = df.groupby(["symbol", "date"], sort=False)[value_col].apply(
        lambda s: _score(s.to_numpy())
    )
    return grouped.rename("value").reset_index()[["date", "symbol", "value"]]


def log_return(df: pd.DataFrame, price_col: str = "adj_close") -> pd.Series:
    """Per-symbol log return on a sorted ``(symbol, date, price_col)`` frame."""
    return df.groupby("symbol")[price_col].transform(lambda s: np.log(s).diff())


def halflife_weights(window: int, halflife: int) -> np.ndarray:
    """``0.5^((window-1-t)/halflife)`` for ``t=0..window-1`` — newest obs gets w=1."""
    lag = np.arange(window - 1, -1, -1, dtype=float)
    return np.power(0.5, lag / halflife)
