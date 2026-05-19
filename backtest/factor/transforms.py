"""Common transforms for factor compute functions.

All operators accept a MultiIndex ``(date, symbol)`` Series — the canonical
output type of registered factors — and return a Series of the same shape.

Provides two families:

- 截面归一化 / 时序变换: :func:`rank`, :func:`z_score`
- 中性化算子(因子层): :func:`industry_neutralize`, :func:`cap_neutralize`

中性化算子被 backfill 在变体 fan-out 时调用,把"原始因子值"加工成"行业/市值中性化后的纯净因子值",再连同 ``variant`` 列一起写入 ``factors_daily``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_panel_series(values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"values must be a pandas Series, got {type(values).__name__}")
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels < 2:
        raise ValueError(
            "values must have a MultiIndex with (date, symbol) levels; "
            "call .set_index(['date', 'symbol']) on a long-form panel first"
        )


def rank(values: pd.Series, ascending: bool = True) -> pd.Series:
    """Cross-sectional rank normalized to ``[0, 1]`` per date.

    For each date, non-NaN values are ranked by ``method='average'`` (ties get
    the average rank) and rescaled with ``(rank - 1) / (N - 1)``, so the
    smallest value maps to 0 and the largest to 1. NaNs are preserved.

    A date with a single non-NaN value yields 0.5 for that observation,
    avoiding the degenerate division by zero.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    ascending : bool, default True
        If True, the largest value maps to 1. If False, to 0.

    Returns
    -------
    pd.Series
        Same index as ``values``, with values in ``[0, 1]``.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples(
    ...     [("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")],
    ...     names=["date", "symbol"],
    ... )
    >>> s = pd.Series([10.0, 20.0, 30.0], index=idx)
    >>> rank(s).round(2).tolist()
    [0.0, 0.5, 1.0]
    """
    _check_panel_series(values)

    def _one(s: pd.Series) -> pd.Series:
        n = int(s.notna().sum())
        if n == 0:
            return s
        if n == 1:
            return s.where(s.isna(), 0.5)
        r = s.rank(method="average", ascending=ascending)
        return (r - 1.0) / (n - 1.0)

    return values.groupby(level=0, group_keys=False).apply(_one)


def _ts_roll(
    values: pd.Series,
    window: int,
    min_periods: int | None,
    *,
    window_min: int,
    min_periods_min: int = 1,
) -> pd.DataFrame:
    """Sort by (symbol, date), groupby symbol, rolling, restore (date, symbol) order.

    Returns the unmodified rolling result (caller does .mean(), .std(), etc.).
    """
    if min_periods is None:
        min_periods = window
    if window < window_min:
        raise ValueError(f"window must be >= {window_min}, got {window}")
    if min_periods < min_periods_min:
        raise ValueError(f"min_periods must be >= {min_periods_min}, got {min_periods}")

    sorted_vals = values.sort_index(level=[1, 0])
    result = sorted_vals.groupby(level=1).rolling(window, min_periods=min_periods)
    return result


def z_score(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Time-series z-score per symbol over a trailing rolling window.

    For each symbol, returns ``(x - rolling_mean) / rolling_std`` where the
    rolling statistics use a trailing window of ``window`` observations.
    When the rolling std is zero (constant window), the result is NaN.

    The first ``min_periods - 1`` observations per symbol are NaN.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations (typically trading days).
    min_periods : int, optional
        Minimum number of observations to produce a value. Defaults to
        ``window`` (strict — no z-score until the window is fully covered).

    Returns
    -------
    pd.Series
        Same index as ``values``, with time-series z-scored values.
    """
    _check_panel_series(values)
    roller = _ts_roll(values, window, min_periods, window_min=2, min_periods_min=2)
    stats = roller.agg(["mean", "std"])
    stats.index = stats.index.droplevel(0)

    sorted_vals = values.sort_index(level=[1, 0])
    z = (sorted_vals - stats["mean"]) / stats["std"].where(stats["std"] > 0, np.nan)
    return z.reindex(values.index)


def _group_zscore(s: pd.Series) -> pd.Series:
    """组内 zscore;单元素组 → 0;std=0 → 0。"""
    mean = s.mean()
    std = s.std()
    if std == 0 or pd.isna(std) or len(s) <= 1:
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def industry_neutralize(
    values: pd.Series,
    industry_panel: pd.DataFrame,
) -> pd.Series:
    """按行业组内 zscore,剥离行业暴露。

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` 的原始因子值。
    industry_panel : pd.DataFrame
        列 ``[date, symbol, industry_code]``,提供每个 ``(date, symbol)`` 在
        所选 level (SW-L1 / SW-L2) 下的行业归属。

    Returns
    -------
    pd.Series
        MultiIndex ``(date, symbol)``,与输入同 shape。每日按 industry_code
        分组后对组内值做 ``zscore``。
        - 缺失行业的 symbol 该日产出 ``NaN``
        - 单成员组该日产出 ``0``
        - 组内 std=0(常数组)产出 ``0``
    """
    _check_panel_series(values)
    if not {"date", "symbol", "industry_code"}.issubset(industry_panel.columns):
        raise ValueError(
            "industry_panel must have columns [date, symbol, industry_code]"
        )

    panel = industry_panel[["date", "symbol", "industry_code"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])

    df = values.rename("value").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(panel, on=["date", "symbol"], how="left")

    has_ind = df["industry_code"].notna()
    out = pd.Series(np.nan, index=df.index, dtype=float)

    if has_ind.any():
        sub = df.loc[has_ind].copy()
        sub["neutral"] = sub.groupby(
            ["date", "industry_code"], group_keys=False
        )["value"].transform(_group_zscore)
        out.loc[has_ind] = sub["neutral"].values

    out.index = pd.MultiIndex.from_arrays(
        [df["date"], df["symbol"]], names=["date", "symbol"]
    )
    return out.reindex(values.index)


def cap_neutralize(
    values: pd.Series,
    cap_panel: pd.DataFrame,
    *,
    cap_field: str = "circ_mv",
    quantiles: int = 5,
) -> pd.Series:
    """按市值分位组内 zscore,剥离市值暴露。

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` 的因子值(可以是原始,也可以是已经过
        行业中性化的)。
    cap_panel : pd.DataFrame
        列至少含 ``[date, symbol, <cap_field>]``。
    cap_field : str
        市值字段名。常用 ``"circ_mv"``(流通市值)或 ``"total_mv"``(总市值)。
    quantiles : int
        分组数。常用 5 或 10。

    Returns
    -------
    pd.Series
        MultiIndex ``(date, symbol)``,与输入同 shape。每日按 cap 分位分组
        后对组内值做 ``zscore``。
        - 缺失 cap 的 symbol 该日产出 ``NaN``
        - 单成员组、std=0 → 0
        - 当某日有效样本数 < ``quantiles`` 时,所有有效样本归到一个组,fallback 为整体 zscore
    """
    _check_panel_series(values)
    if quantiles < 2:
        raise ValueError(f"quantiles must be >= 2, got {quantiles}")
    if cap_field not in cap_panel.columns:
        raise ValueError(f"cap_field '{cap_field}' not in cap_panel columns")
    if not {"date", "symbol"}.issubset(cap_panel.columns):
        raise ValueError(
            "cap_panel must have columns [date, symbol, <cap_field>]"
        )

    panel = cap_panel[["date", "symbol", cap_field]].copy()
    panel = panel.rename(columns={cap_field: "_cap"})
    panel["date"] = pd.to_datetime(panel["date"])

    df = values.rename("value").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(panel, on=["date", "symbol"], how="left")

    out = pd.Series(np.nan, index=df.index, dtype=float)
    has_cap = df["_cap"].notna() & (df["_cap"] > 0)

    if has_cap.any():
        for date_val, idx in df.loc[has_cap].groupby("date").groups.items():
            sub = df.loc[idx]
            n = len(sub)
            if n < quantiles:
                out.loc[idx] = _group_zscore(sub["value"]).values
                continue
            try:
                bins = pd.qcut(sub["_cap"], quantiles, labels=False, duplicates="drop")
            except ValueError:
                out.loc[idx] = _group_zscore(sub["value"]).values
                continue
            neutral = sub["value"].groupby(bins, group_keys=False).transform(_group_zscore)
            out.loc[idx] = neutral.values

    out.index = pd.MultiIndex.from_arrays(
        [df["date"], df["symbol"]], names=["date", "symbol"]
    )
    return out.reindex(values.index)


def ts_rank(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Time-series rank per symbol over a trailing rolling window, scaled to ``[-1, 1]``.

    For each symbol, the current value is ranked within the trailing window
    of ``window`` observations. The rank is then linearly rescaled:
    ``(rank - 1) / (N - 1) * 2 - 1``, so the minimum value in the window
    maps to ``-1`` and the maximum to ``1``.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations (typically trading days).
    min_periods : int, optional
        Minimum number of observations to produce a value. Defaults to
        ``window``.

    Returns
    -------
    pd.Series
        Same index as ``values``, with values in ``[-1, 1]``.
    """
    _check_panel_series(values)

    def _rank_last(x: np.ndarray) -> float:
        valid = x[~np.isnan(x)]
        n = len(valid)
        if n <= 1:
            return 0.0
        last = valid[-1]
        le = int(np.sum(valid <= last))
        lt = int(np.sum(valid < last))
        r = (le + lt + 1) / 2.0
        return (r - 1.0) / (n - 1.0) * 2.0 - 1.0

    result = _ts_roll(values, window, min_periods, window_min=2).apply(
        _rank_last, raw=True
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_mean(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling mean per symbol.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations.
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    result = _ts_roll(values, window, min_periods, window_min=1).mean()
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_std(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling standard deviation per symbol (ddof=1).

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations.
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    result = _ts_roll(values, window, min_periods, window_min=2).std()
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


__all__ = [
    "rank",
    "z_score",
    "ts_rank",
    "ts_mean",
    "ts_std",
    "industry_neutralize",
    "cap_neutralize",
]
