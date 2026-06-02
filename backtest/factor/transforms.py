"""Common transforms for factor compute functions.

All operators accept a MultiIndex ``(date, symbol)`` Series — the canonical
output type of registered factors — and return a Series of the same shape.

Provides two families:

- 截面归一化 / 时序变换: :func:`rank`, :func:`z_score`
- 中性化算子(因子层): :func:`industry_neutralize`, :func:`cap_neutralize`

中性化算子被 backfill 在变体 fan-out 时调用,把"原始因子值"加工成"行业/市值中性化后的纯净因子值",再连同 ``variant`` 列一起写入 ``factors_daily``。

Time-series operator min_periods convention
-------------------------------------------
All ``ts_*`` rolling operators (and :func:`z_score`) default to
``min_periods = ceil(0.7 * window)`` — i.e. a window must contain at least
70% non-NaN observations to emit a value, otherwise NaN. Callers can pass an
explicit ``min_periods`` to override.

The rationale: for daily-frequency factors, a 20-day rolling window over a
suspended stock with only 5 trading days of data shouldn't pass as a valid
``ts_std`` — but with pandas' default ``min_periods=window`` the suspension
would silently turn the whole tail to NaN (too strict, factor goes blank
right after resumption). 70% strikes a balance: tolerates ~6 missing days
out of 20 (typical for short suspensions, dividends, halts) while rejecting
windows where the stock genuinely wasn't trading.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import pandas as pd


MIN_VALID_RATIO: float = 0.7

# Number of threads for parallel per-date cross-sectional ops.
# 0 or 1 = serial (default). Set via env var AQ_CS_NUM_THREADS.
_AQ_CS_NUM_THREADS = int(os.environ.get("AQ_CS_NUM_THREADS", "0"))


def _default_min_periods(window: int, *, lower_bound: int = 1) -> int:
    """``ceil(MIN_VALID_RATIO * window)`` clamped to ``[lower_bound, window]``.

    Used by all ts_* operators when the caller doesn't pass an explicit
    ``min_periods``.
    """
    return max(lower_bound, min(window, math.ceil(MIN_VALID_RATIO * window)))


def _check_panel_series(values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"values must be a pandas Series, got {type(values).__name__}")
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels < 2:
        raise ValueError(
            "values must have a MultiIndex with (date, symbol) levels; "
            "call .set_index(['date', 'symbol']) on a long-form panel first"
        )


def _parallel_cs_apply(
    series: pd.Series,
    per_date_fn,
    *,
    num_threads: int | None = None,
) -> pd.Series:
    """Apply *per_date_fn* to each date group, optionally in parallel.

    *per_date_fn* receives a ``pd.Series`` (single-date slice with
    plain ``symbol`` Index) and must return a ``pd.Series`` of the same
    shape.  When ``num_threads`` is ``None`` the env var
    ``AQ_CS_NUM_THREADS`` controls parallelism; set it to e.g. ``8``.
    """
    threads = num_threads if num_threads is not None else _AQ_CS_NUM_THREADS
    if threads <= 1:
        return series.groupby(level=0, group_keys=False).apply(per_date_fn)

    dates = series.index.get_level_values(0).unique()
    chunk_size = max(1, math.ceil(len(dates) / threads))
    date_chunks = [
        list(dates[i : i + chunk_size]) for i in range(0, len(dates), chunk_size)
    ]

    def _process_chunk(date_list):
        mask = series.index.get_level_values(0).isin(date_list)
        sub = series[mask]
        return sub.groupby(level=0, group_keys=False).apply(per_date_fn)

    results = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(_process_chunk, chunk): i for i, chunk in enumerate(date_chunks)}
        for fut in as_completed(futures):
            results.append((futures[fut], fut.result()))

    results.sort(key=lambda x: x[0])
    return pd.concat([r for _, r in results])


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

    return _parallel_cs_apply(values, _one)


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
    When ``min_periods`` is None, defaults to ``ceil(0.7 * window)`` clamped
    to ``[window_min, window]`` — see module docstring for the rationale.
    """
    if window < window_min:
        raise ValueError(f"window must be >= {window_min}, got {window}")
    if min_periods is None:
        min_periods = _default_min_periods(window, lower_bound=window_min)
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
        ``ceil(0.7 * window)`` — see module docstring.

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

    y_name = "_cs_ols_y_"
    df = values.rename(y_name).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    # validate="m:1" catches dup (date, symbol) rows in design that would
    # silently fan-out the merge and corrupt residuals.
    df = df.merge(design, on=["date", "symbol"], how="left", validate="m:1")

    # Pre-encode dummies once; per-date we slice categorical codes into a
    # dense identity-style block, avoiding pd.get_dummies in the loop.
    if dummy_col is not None:
        codes_all = df[dummy_col].astype("category").cat.codes.to_numpy()
    else:
        codes_all = None
    y_all = df[y_name].to_numpy(dtype=float)
    numeric_block = (
        df[list(numeric_cols)].to_numpy(dtype=float) if numeric_cols else None
    )
    out_arr = np.full(len(df), np.nan)

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
                # drop first level to avoid collinearity with the intercept
                dummy_block = (day_codes[:, None] == present[None, 1:]).astype(float)
                x_parts.append(dummy_block)
        if numeric_block is not None:
            x_parts.append(numeric_block[sel])
        X = np.hstack(x_parts)

        if X.shape[0] <= X.shape[1]:
            continue
        # Normal equations are ~5-10x faster than lstsq's SVD at p≈30, N≈5k.
        # Fall back to lstsq on rank-deficient days.
        XtX = X.T @ X
        Xty = X.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out_arr[sel] = y - X @ beta

    out = pd.Series(
        out_arr,
        index=pd.MultiIndex.from_arrays(
            [df["date"], df["symbol"]], names=["date", "symbol"]
        ),
    )
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

    if not has_cap.any():
        out.index = pd.MultiIndex.from_arrays(
            [df["date"], df["symbol"]], names=["date", "symbol"]
        )
        return out.reindex(values.index)

    sub = df.loc[has_cap].copy()

    # Daily sample count for fallback logic.
    sub["daily_count"] = sub.groupby("date")["value"].transform("count")

    # Cap percentile rank per date — vectorised, no per-day Python loop.
    sub["cap_rank"] = sub.groupby("date")["_cap"].rank(pct=True)
    bins = np.linspace(0, 1, quantiles + 1)
    sub["cap_group"] = pd.cut(
        sub["cap_rank"], bins=bins, labels=False, include_lowest=True
    )

    # For dates with too few samples, collapse into a single fallback group.
    insufficient = sub["daily_count"] < quantiles
    sub.loc[insufficient, "cap_group"] = -1

    # Group zscore via C-level transform (no Python per-group function calls).
    g = sub.groupby(["date", "cap_group"])["value"]
    mean_vals = g.transform("mean")
    std_vals = g.transform("std")
    count_vals = g.transform("count")
    z = (sub["value"] - mean_vals) / std_vals.where(std_vals > 0, np.nan)
    z = z.where((std_vals > 0) & (count_vals > 1), 0.0)

    out.loc[sub.index] = z.values

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
        ``ceil(0.7 * window)`` — see module docstring.

    Returns
    -------
    pd.Series
        Same index as ``values``, with values in ``[-1, 1]``.
    """
    _check_panel_series(values)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    sorted_vals = values.sort_index(level=[1, 0])
    grp = sorted_vals.groupby(level=1)
    # pandas rolling rank (C-level) avoids per-window Python function calls.
    ranks = grp.rolling(window, min_periods=min_periods).rank(method="average")
    counts = grp.rolling(window, min_periods=min_periods).count()
    ranks.index = ranks.index.droplevel(0)
    counts.index = counts.index.droplevel(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(
            counts.values <= 1, 0.0, (ranks.values - 1.0) / (counts.values - 1.0) * 2.0 - 1.0
        )
    result = pd.Series(scaled, index=ranks.index)
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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

    Returns
    -------
    pd.Series
        Same index as ``values``.
    """
    _check_panel_series(values)

    # Fast path: when all non-NaN values are strictly positive, use
    # ``exp(rolling_sum(log(x)))`` — ~10-15× faster than rolling.apply.
    # Note: ``values > 0`` returns *False* (not NaN) for NaN entries, so we
    # must explicitly OR with the isna mask.
    pos_mask = values > 0
    na_mask = values.isna()
    if (pos_mask | na_mask).all():
        log_vals = np.log(values)
        log_sum = _ts_roll(log_vals, window, min_periods, window_min=1).sum()
        log_sum.index = log_sum.index.droplevel(0)
        return np.exp(log_sum).reindex(values.index)

    # Fallback: rolling.apply for mixed/negative values.
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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

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


def _wide_rolling_pairwise(
    x: pd.Series,
    y: pd.Series,
    window: int,
    min_periods: int,
    method: Literal["corr", "cov"],
) -> pd.Series:
    """Vectorised rolling corr/cov via wide-format DataFrame.rolling().

    Unstacks the two series to ``date × symbol`` matrices and lets
    pandas compute the rolling statistic column-wise in C code,
    avoiding the per-symbol Python ``for`` loop.
    """
    wide_x = x.unstack(level=1)
    wide_y = y.unstack(level=1)

    if method == "corr":
        wide_result = wide_x.rolling(window, min_periods=min_periods).corr(wide_y)
    else:
        wide_result = wide_x.rolling(window, min_periods=min_periods).cov(wide_y)

    result = wide_result.stack()
    return result.reindex(x.index)


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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

    Returns
    -------
    pd.Series
        Same index as ``x`` (and ``y``).
    """
    _check_panel_series(x)
    _check_panel_series(y)
    if not x.index.equals(y.index):
        raise ValueError("x and y must have identical MultiIndex")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods is None:
        min_periods = _default_min_periods(window, lower_bound=2)
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")

    return _wide_rolling_pairwise(x, y, window, min_periods, "corr")


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
        Minimum observations. Defaults to ``ceil(0.7 * window)``.

    Returns
    -------
    pd.Series
        Same index as ``x`` (and ``y``).
    """
    _check_panel_series(x)
    _check_panel_series(y)
    if not x.index.equals(y.index):
        raise ValueError("x and y must have identical MultiIndex")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods is None:
        min_periods = _default_min_periods(window, lower_bound=2)
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")

    return _wide_rolling_pairwise(x, y, window, min_periods, "cov")


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

    return _parallel_cs_apply(values, _one)


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

    return _parallel_cs_apply(values, _one)


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

    return _parallel_cs_apply(values, _one)


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

    return _parallel_cs_apply(values, _one)


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


# ---------------------------------------------------------------------------
# Fundamentals helpers — single-quarter derivation, TTM, YoY
# ---------------------------------------------------------------------------
# Unlike the (date, symbol)-indexed operators above, these consume the raw
# PIT-concat panel produced by ``backtest/factor/compute.py`` for factors with
# ``data_sources=['fundamentals']``. The panel has columns
# ``[date, symbol, end_date, inc_*/bs_*/cf_*]`` with multiple rows per
# ``(date, symbol)`` — one per visible quarter ``end_date``. Each helper
# returns a Series aligned with ``panel.index``.

FundamentalKind = Literal["flow", "stock"]

_PRIOR_MMDD_BY_MONTH = {"06": "0331", "09": "0630", "12": "0930"}
# Quarter-month → scale factor that annualises a YTD cumulative number when
# LY_FY / LY_same aren't yet visible. Public — Barra event_ttm reuses it.
TTM_ANNUALIZE_BY_MONTH = {"03": 4.0, "06": 2.0, "09": 4.0 / 3.0, "12": 1.0}


def _check_fundamental_panel(panel: pd.DataFrame, value_col: str) -> None:
    required = {"date", "symbol", "end_date", value_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"panel missing required columns: {sorted(missing)}; "
            f"got columns: {sorted(panel.columns)}"
        )


def _lookup_at_end_date(
    panel: pd.DataFrame, value_col: str, target_end_date: pd.Series
) -> np.ndarray:
    """For each row, look up ``value_col`` at same ``(date, symbol)`` but a
    different ``end_date``.

    Uses ``set_index + reindex`` rather than ``merge`` to avoid hash-join
    overhead at scale. Returns ndarray aligned with ``panel`` (NaN where the
    key isn't present, including rows where ``target_end_date`` is empty).
    """
    indexed = panel.set_index(["date", "symbol", "end_date"])[value_col]
    # Drop duplicate keys (defensive — same (date, symbol, end_date) should be unique
    # post snapshot, but PIT-concat can collide if upstream changes)
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    target_idx = pd.MultiIndex.from_arrays(
        [panel["date"].values, panel["symbol"].values, target_end_date.values],
        names=["date", "symbol", "end_date"],
    )
    return indexed.reindex(target_idx).to_numpy()


def _prev_year_str(end_date: pd.Series) -> pd.Series:
    """``end_date`` YYYY part minus one, as a 4-char string; NaN if non-numeric."""
    year_int = pd.to_numeric(end_date.str[:4], errors="coerce")
    prev = (year_int - 1).astype("Int64").astype(str)
    return prev.where(year_int.notna(), other=pd.NA)


def single_quarter(
    panel: pd.DataFrame, value_col: str, *, kind: FundamentalKind = "flow"
) -> pd.Series:
    """单季度数据 — 累计报告期值相减得到。

    *flow* (利润表 / 现金流量表)::

        Q1 = report (原值)
        Q2 = H1 − Q1
        Q3 = 9M − H1
        Q4 = FY − 9M

    缺环比报告期 → NaN。

    *stock* (资产负债表) 是时点数据，单季度 = 报告期值，直接返回 ``panel[value_col]``。

    Parameters
    ----------
    panel : pd.DataFrame
        必含列 ``date``, ``symbol``, ``end_date``, ``value_col``。``end_date``
        为 ``YYYYMMDD`` 字符串。
    value_col : str
        要取单季度的列名。
    kind : {"flow", "stock"}
        ``flow`` 走相减公式，``stock`` 走 identity。

    Returns
    -------
    pd.Series
        与 ``panel.index`` 对齐。
    """
    _check_fundamental_panel(panel, value_col)
    if kind == "stock":
        return panel[value_col].copy()
    if kind != "flow":
        raise ValueError(f"kind must be 'flow' or 'stock', got {kind!r}")

    end_date = panel["end_date"].astype(str)
    month_str = end_date.str[4:6]
    year_str = end_date.str[:4]

    prior_mmdd = month_str.map(_PRIOR_MMDD_BY_MONTH)
    prior_end_date = (year_str + prior_mmdd).where(prior_mmdd.notna(), other="")

    current = panel[value_col].to_numpy(dtype=float, na_value=np.nan)
    prior_val = _lookup_at_end_date(panel, value_col, prior_end_date)

    result = np.where(month_str.values == "03", current, current - prior_val)
    return pd.Series(result, index=panel.index, name=f"{value_col}_q")


def ttm(
    panel: pd.DataFrame, value_col: str, *, kind: FundamentalKind = "flow"
) -> pd.Series:
    """滚动 12 个月 (TTM)。

    *flow* (利润 / 现金流)::

        FY                : TTM = report
        非 FY 且 LY_* 可见: TTM = current + LY_FY − LY_same
        否则              : TTM = current × {Q1: 4, H1: 2, 9M: 4/3}  (年化兜底)

    *stock* (资产负债表): 时点数据，TTM 默认 = 最新报告期值，直接返回 ``panel[value_col]``。

    Parameters
    ----------
    panel : pd.DataFrame
        必含列 ``date``, ``symbol``, ``end_date``, ``value_col``。
    value_col : str
        要取 TTM 的列名。
    kind : {"flow", "stock"}

    Returns
    -------
    pd.Series
        与 ``panel.index`` 对齐。
    """
    _check_fundamental_panel(panel, value_col)
    if kind == "stock":
        return panel[value_col].copy()
    if kind != "flow":
        raise ValueError(f"kind must be 'flow' or 'stock', got {kind!r}")

    end_date = panel["end_date"].astype(str)
    month_str = end_date.str[4:6]
    prev_year = _prev_year_str(end_date)
    mmdd = end_date.str[4:]

    ly_fy_ed = (prev_year + "1231").where(prev_year.notna(), other="")
    ly_same_ed = (prev_year + mmdd).where(prev_year.notna(), other="")

    current = panel[value_col].to_numpy(dtype=float, na_value=np.nan)
    ly_fy_val = _lookup_at_end_date(panel, value_col, ly_fy_ed)
    ly_same_val = _lookup_at_end_date(panel, value_col, ly_same_ed)

    is_fy = month_str.values == "12"
    formula = current + ly_fy_val - ly_same_val

    annualize_scale = month_str.map(TTM_ANNUALIZE_BY_MONTH).to_numpy(dtype=float, na_value=np.nan)
    fallback = current * annualize_scale

    result = np.where(is_fy, current, formula)
    result = np.where(np.isnan(result), fallback, result)
    return pd.Series(result, index=panel.index, name=f"{value_col}_ttm")


def yoy(
    panel: pd.DataFrame, value_col: str, *, relative: bool = True
) -> pd.Series:
    """同比 — 当前 vs 上年同期。

    ``relative=True``  → ``(current − LY_same) / |LY_same|`` (增长率，分母 0 → NaN)
    ``relative=False`` → ``current − LY_same`` (绝对差)

    Parameters
    ----------
    panel : pd.DataFrame
        必含列 ``date``, ``symbol``, ``end_date``, ``value_col``。
    value_col : str
    relative : bool, default True

    Returns
    -------
    pd.Series
        与 ``panel.index`` 对齐;缺上年同期 → NaN。
    """
    _check_fundamental_panel(panel, value_col)

    end_date = panel["end_date"].astype(str)
    prev_year = _prev_year_str(end_date)
    ly_same_ed = (prev_year + end_date.str[4:]).where(prev_year.notna(), other="")

    current = panel[value_col].to_numpy(dtype=float, na_value=np.nan)
    ly_same_val = _lookup_at_end_date(panel, value_col, ly_same_ed)

    if relative:
        denom = np.abs(ly_same_val)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (current - ly_same_val) / denom
        result = np.where(denom > 0, ratio, np.nan)
        name = f"{value_col}_yoy"
    else:
        result = current - ly_same_val
        name = f"{value_col}_yoy_abs"
    return pd.Series(result, index=panel.index, name=name)


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
    "single_quarter",
    "ttm",
    "yoy",
]
