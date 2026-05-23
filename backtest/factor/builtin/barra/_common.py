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
    TTM_ANNUALIZE_BY_MONTH,
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

    On an empty ``raw_series`` (e.g. an early-history chunk where the L3
    helper produced no rows), return the empty Series unchanged. Without the
    short-circuit, pandas' ``groupby(...).apply`` inside ``cs_mad_winsorize``
    collapses the empty (date, symbol) MultiIndex to a single-level Index,
    which the next step's panel-series check then rejects.
    """
    if raw_series.empty:
        return raw_series
    industry_panel = market_storage.get_industry_panel_range(
        start=start, end=end, level="L1",
    )
    series = cs_mad_winsorize(raw_series, k=3.0)
    series = industry_median_fill(series, industry_panel)
    series = cs_zscore(series)
    return series


def event_ttm(events: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-event, per-history-quarter flow TTM.

    ``events`` is an event panel from
    :meth:`MarketStorage.get_fina_event_panel` — long-format with one
    row per ``(symbol, announce_end_date, end_date)``. For **every**
    history end_date in each event, compute the TTM as visible at that
    event's announcement (LY_FY / LY_same looked up within the same
    event's history rows).

    Returns the same long-format frame plus a ``<value_col>_ttm``
    column. Downstream slope / averaging consumes the TTM history as a
    time series indexed by end_date ASC.

    TTM formula (mirrors :func:`transforms.ttm` with ``kind="flow"``):

    - end_date is a fiscal year-end (month "12"): TTM = current
    - else if LY_FY and LY_same both visible: TTM = current + LY_FY − LY_same
    - else: annualize current by month scale
    """
    if events.empty:
        out = events.copy()
        out[f"{value_col}_ttm"] = pd.Series(dtype=float)
        return out

    df = events.copy()
    df["end_date"] = df["end_date"].astype(str)
    df["announce_end_date"] = df["announce_end_date"].astype(str)

    month_str = df["end_date"].str[4:6].to_numpy()
    year = df["end_date"].str[:4].astype(int).to_numpy()
    mmdd = df["end_date"].str[4:].to_numpy()
    prev_year = (year - 1).astype(str)
    ly_fy_key = np.char.add(prev_year, "1231")
    ly_same_key = np.char.add(prev_year, mmdd)

    # Two left-merges look up LY_FY / LY_same within the same event. Using
    # pandas merge on indexed columns is ~10× faster at 22k+ events than
    # the prior dict-of-tuples + list-comprehension lookup.
    lookup_cols = ["symbol", "announce_end_date", "end_date", value_col]
    lookup = df[lookup_cols].drop_duplicates(
        ["symbol", "announce_end_date", "end_date"], keep="last",
    ).rename(columns={value_col: "_v"})

    keys = df[["symbol", "announce_end_date"]].copy()
    keys["end_date"] = ly_fy_key
    ly_fy_val = keys.merge(lookup, on=lookup_cols[:3], how="left")["_v"].to_numpy()
    keys["end_date"] = ly_same_key
    ly_same_val = keys.merge(lookup, on=lookup_cols[:3], how="left")["_v"].to_numpy()

    current = df[value_col].to_numpy(dtype=float)
    is_fy = month_str == "12"
    formula = current + ly_fy_val - ly_same_val
    annualize_scale = np.array(
        [TTM_ANNUALIZE_BY_MONTH.get(m, np.nan) for m in month_str], dtype=float,
    )
    fallback = current * annualize_scale

    result = np.where(is_fy, current, formula)
    result = np.where(np.isnan(result), fallback, result)

    out = df.copy()
    out[f"{value_col}_ttm"] = result
    return out


def event_latest_value(events: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-event ``value_col`` taken at ``end_date == announce_end_date``.

    For balance-sheet stocks like ``bs_total_assets`` the PIT value is
    just the latest reported figure at the announcement.
    """
    if events.empty:
        return pd.DataFrame(
            columns=["symbol", "announce_end_date", "f_ann_date",
                     "next_f_ann_date", "value"],
        )
    df = events.copy()
    df["end_date"] = df["end_date"].astype(str)
    df["announce_end_date"] = df["announce_end_date"].astype(str)
    mask = df["end_date"].to_numpy() == df["announce_end_date"].to_numpy()
    out = df.loc[mask, [
        "symbol", "announce_end_date", "f_ann_date", "next_f_ann_date",
        value_col,
    ]].rename(columns={value_col: "value"}).reset_index(drop=True)
    return out


def event_slope_over_mean(
    events: pd.DataFrame,
    value_col: str,
    *,
    n: int = 20,
    sign: float = 1.0,
) -> pd.DataFrame:
    """Per-event ``slope / |mean|`` regression on a long-format event panel.

    ``events`` is the output of
    :meth:`MarketStorage.get_fina_event_panel` — one row per
    ``(symbol, announce_end_date, end_date)``, sorted by ``end_date``
    ASC within each event. For each ``(symbol, announce_end_date)``
    group, regress the last n quarters of ``value_col`` on integer
    time, scale by ``|mean|``, multiply by ``sign``.

    Vectorised: reshape via cumcount + scatter into a
    ``(N_events, n)`` numpy array, then the per-row OLS closed-form
    is a handful of broadcast ops.

    Returns a frame ``[symbol, announce_end_date, f_ann_date,
    next_f_ann_date, value]`` — one row per event, ready for
    :func:`expand_events_to_dates`.
    """
    df = events.dropna(subset=[value_col]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=["symbol", "announce_end_date", "f_ann_date",
                     "next_f_ann_date", "value"],
        )

    df = df.sort_values(
        ["symbol", "announce_end_date", "end_date"]
    ).reset_index(drop=True)

    grp = df.groupby(["symbol", "announce_end_date"], sort=False)
    seq = grp.cumcount().to_numpy()
    group_id = (seq == 0).cumsum() - 1
    n_groups = int(group_id.max()) + 1 if len(group_id) else 0

    values = df[value_col].to_numpy(dtype=float)
    mat = np.full((n_groups, n), np.nan)
    mat[group_id, seq] = values

    mask = ~np.isnan(mat)
    n_valid = mask.sum(axis=1).astype(float)
    x_full = np.broadcast_to(np.arange(n, dtype=float), mat.shape)
    x = np.where(mask, x_full, 0.0)
    y = np.where(mask, mat, 0.0)

    sum_x = x.sum(axis=1)
    sum_y = y.sum(axis=1)
    sum_xx = (x * x).sum(axis=1)
    sum_xy = (x * y).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = sum_x / n_valid
        mean_y = sum_y / n_valid
        cov_xy = sum_xy / n_valid - mean_x * mean_y
        var_x = sum_xx / n_valid - mean_x * mean_x
        slope = cov_xy / var_x
        score = sign * slope / np.abs(mean_y)

    bad = (n_valid < 4) | (var_x <= 0) | (mean_y == 0) | ~np.isfinite(score)
    score = np.where(bad, np.nan, score)

    # One row per event: pick the first row of each group, carry over
    # announcement metadata (f_ann_date / next_f_ann_date).
    first_mask = np.r_[True, np.diff(group_id) != 0]
    out = df.loc[
        first_mask,
        ["symbol", "announce_end_date", "f_ann_date", "next_f_ann_date"],
    ].reset_index(drop=True)
    out["value"] = score
    return out


def to_panel_series(df: pd.DataFrame, values, name: str) -> pd.Series:
    """Build a ``(date, symbol)``-indexed Series from a frame plus a value column.

    ``df`` must have ``date`` and ``symbol`` columns. ``values`` can be a
    column name or any array-like aligned with ``df``.
    """
    if isinstance(values, str):
        values = df[values].values
    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]], names=["date", "symbol"])
    return pd.Series(values, index=idx, name=name)


def expand_events_to_dates(
    events: pd.DataFrame,
    trade_dates: pd.Series,
    *,
    market_storage: MarketStorage,
    value_col: str = "value",
    name: str = "value",
) -> pd.Series:
    """Range-join per-event scalars onto every trade date in their validity interval.

    ``events`` must have columns ``[symbol, f_ann_date, next_f_ann_date,
    value_col]``. Each row is one announcement worth of computed factor
    value. ``next_f_ann_date`` is the publication date of the symbol's
    next announcement (NULL if still current). The validity interval is
    ``[f_ann_date, next_f_ann_date)`` half-open in trade-date space.

    Returns a ``(date, symbol)``-indexed Series. DuckDB's range-join is
    5–10× faster than the equivalent ``merge_asof + filter`` in pandas
    at our row counts; we run it on the live ``market_storage.conn`` so
    we inherit its memory-limit / spill PRAGMAs and avoid the per-call
    connection setup.
    """
    if events.empty:
        return pd.Series(dtype=float, name=name).rename_axis(["date", "symbol"])

    events = events[["symbol", "f_ann_date", "next_f_ann_date", value_col]].copy()
    events["f_ann_date"] = events["f_ann_date"].astype(str)
    events["next_f_ann_date"] = events["next_f_ann_date"].astype("string")

    td = pd.DataFrame({"date": pd.to_datetime(trade_dates)})

    con = market_storage.conn
    con.register("__events", events)
    con.register("__td", td)
    try:
        result = con.execute(f"""
            SELECT td.date, e.symbol, e."{value_col}" AS value
            FROM __td td
            JOIN __events e
              ON td.date >= strptime(e.f_ann_date, '%Y%m%d')
             AND (e.next_f_ann_date IS NULL
                  OR td.date < strptime(e.next_f_ann_date, '%Y%m%d'))
            ORDER BY td.date, e.symbol
        """).fetchdf()
    finally:
        con.unregister("__events")
        con.unregister("__td")

    if result.empty:
        return pd.Series(dtype=float, name=name).rename_axis(["date", "symbol"])

    return pd.Series(
        result["value"].to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [result["date"], result["symbol"]], names=["date", "symbol"],
        ),
        name=name,
    )


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
    Used by AGRO/EGRO when applied to a single time series; the matrix
    form for many series at once is :func:`event_slope_over_mean`.
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


def log_return(df: pd.DataFrame, price_col: str = "adj_close") -> pd.Series:
    """Per-symbol log return on a sorted ``(symbol, date, price_col)`` frame."""
    return df.groupby("symbol")[price_col].transform(lambda s: np.log(s).diff())


def halflife_weights(window: int, halflife: int) -> np.ndarray:
    """``0.5^((window-1-t)/halflife)`` for ``t=0..window-1`` — newest obs gets w=1."""
    lag = np.arange(window - 1, -1, -1, dtype=float)
    return np.power(0.5, lag / halflife)
