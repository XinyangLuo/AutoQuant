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


def cs_ols_residualize(
    values: pd.Series,
    design_panel: pd.DataFrame,
    *,
    dummy_col: str | None = "industry_code",
    numeric_cols: tuple[str, ...] = (),
) -> pd.Series:
    """Cross-section OLS residual per date — regress ``values`` on the design.

    For each date, build a design matrix of:

    * an intercept column,
    * (optional) one-hot encoding of ``dummy_col`` with **the first category
      dropped** to avoid perfect collinearity with the intercept,
    * each numeric column listed in ``numeric_cols``.

    Then solve OLS via ``numpy.linalg.lstsq`` and return ``y - X @ beta``
    (residuals). Symbols missing any regressor on a given day are NaN that
    day; NaNs in ``values`` propagate. Days with fewer non-NaN rows than
    regressors are returned as NaN (rank-deficient).

    Used by the ``barra_ind_size`` variant pipeline (PLAN.md §2.2) to strip
    industry and Size exposure from a candidate alpha.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series — the factor to residualize.
    design_panel : pd.DataFrame
        Columns ``[date, symbol]`` plus ``dummy_col`` and every entry in
        ``numeric_cols``. One row per ``(date, symbol)`` for the universe
        used in the regression.
    dummy_col : str | None
        Categorical column name. Pass ``None`` to skip the dummy block
        entirely (regress on intercept + numeric only).
    numeric_cols : tuple[str, ...]
        Names of numeric regressor columns in ``design_panel``.

    Returns
    -------
    pd.Series
        MultiIndex ``(date, symbol)`` residual series, reindexed to match
        ``values.index``.
    """
    _check_panel_series(values)
    cols = ["date", "symbol"]
    if dummy_col is not None:
        cols.append(dummy_col)
    cols.extend(numeric_cols)
    missing = set(cols) - set(design_panel.columns)
    if missing:
        raise ValueError(f"design_panel missing columns: {sorted(missing)}")

    design = design_panel[cols].copy()
    design["date"] = pd.to_datetime(design["date"])

    df = values.rename("__y__").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(design, on=["date", "symbol"], how="left")

    out_index = pd.MultiIndex.from_arrays(
        [df["date"], df["symbol"]], names=["date", "symbol"]
    )
    out = pd.Series(np.nan, index=out_index, dtype=float)

    # Pre-encode dummies once for the whole panel; categorical codes let us
    # slice into a dense identity-style block per date without rebuilding
    # the dummy frame in the loop.
    if dummy_col is not None:
        cat = df[dummy_col].astype("category")
        codes_all = cat.cat.codes.to_numpy()
        n_levels = len(cat.cat.categories)
    else:
        codes_all = None
        n_levels = 0
    y_all = df["__y__"].to_numpy(dtype=float)
    numeric_block = (
        df[list(numeric_cols)].to_numpy(dtype=float) if numeric_cols else None
    )

    for _, idx in df.groupby("date", sort=False).groups.items():
        positions = np.asarray(idx, dtype=int)
        valid = ~np.isnan(y_all[positions])
        if numeric_block is not None:
            valid &= ~np.isnan(numeric_block[positions]).any(axis=1)
        if codes_all is not None:
            valid &= codes_all[positions] >= 0
        if valid.sum() < 2:
            continue
        sel = positions[valid]
        y = y_all[sel]

        x_parts: list[np.ndarray] = [np.ones((sel.size, 1))]
        if codes_all is not None:
            day_codes = codes_all[sel]
            present = np.unique(day_codes)
            if present.size > 1:
                # one-hot, drop first to avoid collinearity with intercept
                dummy_block = (day_codes[:, None] == present[None, 1:]).astype(float)
                x_parts.append(dummy_block)
        if numeric_block is not None:
            x_parts.append(numeric_block[sel])
        X = np.hstack(x_parts)

        if X.shape[0] <= X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.iloc[sel] = y - X @ beta

    return out.reindex(values.index)


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


def ts_sum(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling sum per symbol.

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
    result = _ts_roll(values, window, min_periods, window_min=1).sum()
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_min(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling minimum per symbol.

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
    result = _ts_roll(values, window, min_periods, window_min=1).min()
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_max(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling maximum per symbol.

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
    result = _ts_roll(values, window, min_periods, window_min=1).max()
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def _argmax_last(arr: np.ndarray) -> float:
    """Return distance from end to the maximum value (0-based).

    For array [1, 2, 3] → max=3 at position 2 → distance from end = 0.
    For array [3, 2, 1] → max=3 at position 0 → distance from end = 2.
    NaN values are ignored.
    """
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.nan
    v = arr[valid]
    idx = int(np.argmax(v))
    return float(len(v) - 1 - idx)


def _argmin_last(arr: np.ndarray) -> float:
    """Return distance from end to the minimum value (0-based).

    NaN values are ignored.
    """
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.nan
    v = arr[valid]
    idx = int(np.argmin(v))
    return float(len(v) - 1 - idx)


def ts_argmax(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Distance from window end to the maximum value, per symbol.

    For each symbol, over a trailing window of ``window`` observations,
    returns how many steps back from the current observation the maximum
    value occurs. A value of ``0`` means the current observation is the
    maximum; ``window - 1`` means the maximum is at the earliest point
    in the window.

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
        Same index as ``values``, values are integers in ``[0, window-1]``.
    """
    _check_panel_series(values)
    result = _ts_roll(values, window, min_periods, window_min=1).apply(
        _argmax_last, raw=True
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_argmin(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Distance from window end to the minimum value, per symbol.

    See :func:`ts_argmax` for semantics; this returns the distance to
    the minimum instead.

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
        Same index as ``values``, values are integers in ``[0, window-1]``.
    """
    _check_panel_series(values)
    result = _ts_roll(values, window, min_periods, window_min=1).apply(
        _argmin_last, raw=True
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_delta(
    values: pd.Series,
    d: int,
) -> pd.Series:
    """Difference between current value and the value ``d`` observations ago.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    d : int
        Look-back offset in observations (must be >= 1).

    Returns
    -------
    pd.Series
        Same index as ``values``. First ``d`` observations per symbol are NaN.
    """
    _check_panel_series(values)
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    sorted_vals = values.sort_index(level=[1, 0])
    result = sorted_vals.groupby(level=1).diff(d)
    return result.reindex(values.index)


def ts_delay(
    values: pd.Series,
    d: int,
) -> pd.Series:
    """Value from ``d`` observations ago.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    d : int
        Look-back offset in observations (must be >= 1).

    Returns
    -------
    pd.Series
        Same index as ``values``. First ``d`` observations per symbol are NaN.
    """
    _check_panel_series(values)
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    sorted_vals = values.sort_index(level=[1, 0])
    result = sorted_vals.groupby(level=1).shift(d)
    return result.reindex(values.index)


def ts_pct_change(
    values: pd.Series,
    d: int,
) -> pd.Series:
    """Percentage change over ``d`` observations: ``(current - prior) / prior``.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    d : int
        Look-back offset in observations (must be >= 1).

    Returns
    -------
    pd.Series
        Same index as ``values``. First ``d`` observations per symbol are NaN.
        Division by zero yields NaN.
    """
    _check_panel_series(values)
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    delayed = ts_delay(values, d)
    return (values - delayed) / delayed


def ts_product(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling product per symbol.

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
    result = _ts_roll(values, window, min_periods, window_min=1).apply(
        lambda x: np.nanprod(x) if len(x) > 0 else np.nan, raw=True
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_skewness(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling skewness (Fisher's definition) per symbol.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations (must be >= 3).
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    result = _ts_roll(values, window, min_periods, window_min=3).skew()
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_kurtosis(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling excess kurtosis (Fisher's definition) per symbol.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations (must be >= 4).
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    result = _ts_roll(values, window, min_periods, window_min=4).kurt()
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_ir(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing information ratio per symbol: ``mean / std``.

    When the rolling std is zero, the result is NaN.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations (must be >= 2).
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    roller = _ts_roll(values, window, min_periods, window_min=2)
    stats = roller.agg(["mean", "std"])
    stats.index = stats.index.droplevel(0)

    sorted_vals = values.sort_index(level=[1, 0])
    ir = stats["mean"] / stats["std"].where(stats["std"] > 0, np.nan)
    return ir.reindex(values.index)


def _linear_decay_weights(window: int) -> np.ndarray:
    """Weights [1, 2, ..., window] normalized to sum to 1."""
    w = np.arange(1, window + 1, dtype=float)
    return w / w.sum()


def ts_decay_linear(
    values: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Linearly decay-weighted rolling average per symbol.

    Weights increase linearly from the oldest to the newest observation
    within the window: ``[1, 2, ..., window] / sum(1..window)``.

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
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_periods is None:
        min_periods = window

    weights = _linear_decay_weights(window)

    def _wmean(arr: np.ndarray) -> float:
        valid = ~np.isnan(arr)
        n = int(valid.sum())
        if n < min_periods:
            return np.nan
        v = arr[valid]
        w = weights[-n:]
        return float(np.dot(v, w / w.sum()))

    result = _ts_roll(values, window, min_periods, window_min=1).apply(
        _wmean, raw=True
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_decay_exp(
    values: pd.Series,
    window: int,
    *,
    halflife: float = 10.0,
    min_periods: int | None = None,
) -> pd.Series:
    """Exponentially decay-weighted rolling average per symbol.

    Weights follow ``w_i = 0.5 ^ ((window - 1 - i) / halflife)`` where
    ``i = 0`` is the oldest observation in the window.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    window : int
        Rolling window length in observations.
    halflife : float, default 10.0
        Number of observations for weight to decay by half.
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")
    if min_periods is None:
        min_periods = window

    # weights[i] for i=0..window-1 where i=0 is oldest
    ages = np.arange(window - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / halflife)

    def _wmean(arr: np.ndarray) -> float:
        valid = ~np.isnan(arr)
        n = int(valid.sum())
        if n < min_periods:
            return np.nan
        v = arr[valid]
        w = weights[-n:]
        return float(np.dot(v, w / w.sum()))

    result = _ts_roll(values, window, min_periods, window_min=1).apply(
        _wmean, raw=True
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def ts_corr(
    x: pd.Series,
    y: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling Pearson correlation between two series per symbol.

    Parameters
    ----------
    x, y : pd.Series
        MultiIndex ``(date, symbol)`` Series with matching index.
    window : int
        Rolling window length in observations (must be >= 2).
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``x`` (and ``y``).
    """
    _check_panel_series(x)
    _check_panel_series(y)
    if not x.index.equals(y.index):
        raise ValueError("x and y must have identical MultiIndex")
    if min_periods is None:
        min_periods = window
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")

    df = pd.DataFrame({"x": x, "y": y})
    sorted_df = df.sort_index(level=[1, 0])

    # Per-symbol rolling corr to avoid pandas MultiIndex corr bug.
    out_vals: list[np.ndarray] = []
    out_idx: list[tuple] = []
    for sym, sub in sorted_df.groupby(level=1):
        sub = sub.droplevel(1)
        corr = sub["x"].rolling(window, min_periods=min_periods).corr(sub["y"])
        out_vals.append(corr.values)
        out_idx.extend([(d, sym) for d in corr.index])

    if not out_vals:
        return pd.Series(np.nan, index=x.index)

    result = pd.Series(
        np.concatenate(out_vals),
        index=pd.MultiIndex.from_tuples(out_idx, names=["date", "symbol"]),
    )
    return result.reindex(x.index)


def ts_covariance(
    x: pd.Series,
    y: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Trailing rolling covariance between two series per symbol (ddof=1).

    Parameters
    ----------
    x, y : pd.Series
        MultiIndex ``(date, symbol)`` Series with matching index.
    window : int
        Rolling window length in observations (must be >= 2).
    min_periods : int, optional
        Minimum observations. Defaults to ``window``.

    Returns
    -------
    pd.Series
        Same index as ``x`` (and ``y``).
    """
    _check_panel_series(x)
    _check_panel_series(y)
    if not x.index.equals(y.index):
        raise ValueError("x and y must have identical MultiIndex")
    if min_periods is None:
        min_periods = window
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")

    df = pd.DataFrame({"x": x, "y": y})
    sorted_df = df.sort_index(level=[1, 0])

    out_vals: list[np.ndarray] = []
    out_idx: list[tuple] = []
    for sym, sub in sorted_df.groupby(level=1):
        sub = sub.droplevel(1)
        cov = sub["x"].rolling(window, min_periods=min_periods).cov(sub["y"])
        out_vals.append(cov.values)
        out_idx.extend([(d, sym) for d in cov.index])

    if not out_vals:
        return pd.Series(np.nan, index=x.index)

    result = pd.Series(
        np.concatenate(out_vals),
        index=pd.MultiIndex.from_tuples(out_idx, names=["date", "symbol"]),
    )
    return result.reindex(x.index)


# ---------------------------------------------------------------------------
# Cross-sectional operators (cs_*)
# ---------------------------------------------------------------------------


def cs_zscore(values: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date: ``(x - mean) / std``.

    For each date, non-NaN values are centered and scaled. NaNs are
    preserved. If a date has <= 1 non-NaN value or std == 0, all
    non-NaN values are mapped to 0.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)

    def _one(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std()
        if std == 0 or pd.isna(std) or s.notna().sum() <= 1:
            return s.where(s.isna(), 0.0)
        return (s - mean) / std

    return values.groupby(level=0, group_keys=False).apply(_one)


def cs_demean(values: pd.Series) -> pd.Series:
    """Cross-sectional demean per date: ``x - mean(x)``.

    For each date, subtract the cross-sectional mean from all non-NaN
    values. NaNs are preserved.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)

    def _one(s: pd.Series) -> pd.Series:
        mean = s.mean()
        if pd.isna(mean):
            return s
        return s - mean

    return values.groupby(level=0, group_keys=False).apply(_one)


def cs_winsorize(
    values: pd.Series,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Cross-sectional winsorization (percentile clip) per date.

    For each date, non-NaN values are clipped to the ``lower`` and
    ``upper`` percentiles of the cross-sectional distribution.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    lower : float, default 0.01
        Lower percentile (0.0–1.0).
    upper : float, default 0.99
        Upper percentile (0.0–1.0).

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"require 0 <= lower < upper <= 1, got lower={lower}, upper={upper}")

    def _one(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 2:
            return s
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        clipped = s.clip(lower=lo, upper=hi)
        return clipped

    return values.groupby(level=0, group_keys=False).apply(_one)


def cs_mad_winsorize(values: pd.Series, *, k: float = 3.0) -> pd.Series:
    """Cross-sectional MAD winsorization per date: clip to ``median ± k·1.4826·MAD``.

    The 1.4826 scale factor makes ``1.4826·MAD`` a consistent estimator of σ
    under the normal distribution, so ``k=3`` corresponds to a 3-σ clip.
    For each date, non-NaN values outside the band are clipped to its edge;
    NaNs are preserved. Used as the first step of the Barra L3 pipeline
    (see PLAN.md §2.1 计算规则 step 2).

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    k : float, default 3.0
        Number of scaled-MAD widths to keep on each side of the median.
    """
    _check_panel_series(values)
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    def _one(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 2:
            return s
        med = valid.median()
        mad = (valid - med).abs().median()
        if mad == 0 or pd.isna(mad):
            return s
        delta = k * 1.4826 * mad
        return s.clip(lower=med - delta, upper=med + delta)

    return values.groupby(level=0, group_keys=False).apply(_one)


def industry_median_fill(
    values: pd.Series,
    industry_panel: pd.DataFrame,
) -> pd.Series:
    """Fill NaNs with the same-day, same-industry median.

    For each ``(date, industry_code)`` group, missing values are replaced
    with the median of non-NaN peers in that group. Symbols with no
    industry mapping that day are left untouched (still NaN). Used as
    step 3 of the Barra L3 pipeline (PLAN.md §2.1).

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    industry_panel : pd.DataFrame
        Columns ``[date, symbol, industry_code]``.
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
    if has_ind.any():
        med = df.loc[has_ind].groupby(
            ["date", "industry_code"], group_keys=False,
        )["value"].transform("median")
        fill_mask = has_ind & df["value"].isna()
        df.loc[fill_mask, "value"] = med[fill_mask]

    out = pd.Series(
        df["value"].values,
        index=pd.MultiIndex.from_arrays(
            [df["date"], df["symbol"]], names=["date", "symbol"]
        ),
    )
    return out.reindex(values.index)


# ---------------------------------------------------------------------------
# Math / utility operators
# ---------------------------------------------------------------------------


def abs_(values: pd.Series) -> pd.Series:
    """Element-wise absolute value.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    return values.abs()


def sign(values: pd.Series) -> pd.Series:
    """Element-wise sign: -1, 0, or 1.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    return np.sign(values)


def log(values: pd.Series) -> pd.Series:
    """Element-wise natural logarithm.

    Non-positive values produce NaN.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    return np.log(values.where(values > 0, np.nan))


def sqrt(values: pd.Series) -> pd.Series:
    """Element-wise square root.

    Negative values produce NaN.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    return np.sqrt(values.where(values >= 0, np.nan))


def signed_power(values: pd.Series, power: float) -> pd.Series:
    """``sign(x) * |x| ^ power``.

    Preserves the sign of the original value while raising its
    magnitude to ``power``. NaN values are preserved.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    power : float
        Exponent applied to the absolute value.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    return np.sign(values) * np.power(np.abs(values), power)


def inverse(values: pd.Series) -> pd.Series:
    """Element-wise reciprocal: ``1 / x``.

    Division by zero produces NaN.

    Parameters
    ----------
    values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)
    return 1.0 / values.replace(0, np.nan)


def if_else(
    condition: pd.Series,
    true_values: pd.Series,
    false_values: pd.Series,
) -> pd.Series:
    """Element-wise conditional selection.

    For each observation, returns ``true_values`` where ``condition``
    is True, else ``false_values``. All three series must share the
    same MultiIndex.

    Parameters
    ----------
    condition : pd.Series
        Boolean MultiIndex ``(date, symbol)`` Series.
    true_values : pd.Series
        MultiIndex ``(date, symbol)`` Series.
    false_values : pd.Series
        MultiIndex ``(date, symbol)`` Series.

    Returns
    -------
    pd.Series
        Same index as ``condition``.
    """
    _check_panel_series(condition)
    _check_panel_series(true_values)
    _check_panel_series(false_values)
    if not condition.index.equals(true_values.index):
        raise ValueError("condition and true_values must have identical index")
    if not condition.index.equals(false_values.index):
        raise ValueError("condition and false_values must have identical index")

    return true_values.where(condition, false_values)


__all__ = [
    "rank",
    "z_score",
    "ts_rank",
    "ts_mean",
    "ts_std",
    "ts_sum",
    "ts_min",
    "ts_max",
    "ts_argmin",
    "ts_argmax",
    "ts_delta",
    "ts_delay",
    "ts_pct_change",
    "ts_product",
    "ts_skewness",
    "ts_kurtosis",
    "ts_ir",
    "ts_decay_linear",
    "ts_decay_exp",
    "ts_corr",
    "ts_covariance",
    "cs_zscore",
    "cs_demean",
    "cs_winsorize",
    "cs_mad_winsorize",
    "cs_ols_residualize",
    "industry_median_fill",
    "abs_",
    "sign",
    "log",
    "sqrt",
    "signed_power",
    "inverse",
    "if_else",
    "industry_neutralize",
    "cap_neutralize",
]
