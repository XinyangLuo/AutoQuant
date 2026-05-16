"""Common transforms for factor compute functions.

All operators accept a MultiIndex ``(date, symbol)`` Series — the canonical
output type of registered factors — and return a Series of the same shape.
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
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods is None:
        min_periods = window
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")

    sorted_vals = values.sort_index(level=[1, 0])
    stats = (
        sorted_vals.groupby(level=1)
        .rolling(window, min_periods=min_periods)
        .agg(["mean", "std"])
    )
    stats.index = stats.index.droplevel(0)

    z = (sorted_vals - stats["mean"]) / stats["std"].where(stats["std"] > 0, np.nan)
    return z.reindex(values.index)


__all__ = ["rank", "z_score"]
