"""Barra Beta factor — internal helper for ``f_barra_beta``.

WLS regression of daily log-returns on CSI 300 daily log-returns.
Window 252d, half-life 63d (so weight ``w_t = 0.5^{(T-t)/63}``). Returns
the slope coefficient β; intercept and residual are discarded.

Implementation: per-symbol ``sliding_window_view`` materializes the
``(n_obs, window)`` view, then the 2-param WLS closed-form is vectorized
across the leading axis. Avoids the per-window Python loop / allocation
that ``rolling.apply`` would impose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from backtest.data.storage import MarketStorage
from backtest.factor.builtin.barra._common import (
    halflife_weights,
    to_panel_series,
)

BETA_WINDOW = 252
BETA_HALFLIFE = 63
_CSI300 = "000300.SH"


def _vectorized_wls_beta(
    y: np.ndarray, x: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """β at every position ``t >= window-1`` via vectorized closed-form WLS.

    ``y`` and ``x`` are 1-D arrays of equal length. Returns a same-length
    array with the first ``window-1`` entries NaN. NaNs in either input are
    masked per-window; if fewer than half the obs are valid the window
    returns NaN.
    """
    window = weights.size
    n = y.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    # Shape: (n - window + 1, window)
    y_win = sliding_window_view(y, window)
    x_win = sliding_window_view(x, window)

    mask = ~(np.isnan(y_win) | np.isnan(x_win))
    wm = weights * mask  # broadcasts along leading axis
    sw_m = wm.sum(axis=1)

    valid = (mask.sum(axis=1) >= window // 2) & (sw_m > 0)
    if not valid.any():
        return out

    y_clean = np.where(mask, y_win, 0.0)
    x_clean = np.where(mask, x_win, 0.0)

    sw_safe = np.where(sw_m > 0, sw_m, 1.0)
    x_mean = (wm * x_clean).sum(axis=1) / sw_safe
    y_mean = (wm * y_clean).sum(axis=1) / sw_safe

    xd = x_clean - x_mean[:, None]
    yd = y_clean - y_mean[:, None]
    cov = (wm * xd * yd).sum(axis=1)
    var = (wm * xd * xd).sum(axis=1)

    good = valid & (var > 0)
    betas = np.where(good, cov / np.where(var > 0, var, 1.0), np.nan)
    out[window - 1:] = betas
    return out


def barra_beta_beta(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
) -> pd.Series:
    df = panel[["date", "symbol", "close", "adj_factor"]].copy()
    df["adj_close"] = df["close"] * df["adj_factor"]
    df = df.sort_values(["symbol", "date"])

    start = df["date"].min().strftime("%Y%m%d")
    end = df["date"].max().strftime("%Y%m%d")
    bench = market_storage.get_index_bars(
        [_CSI300], start=start, end=end, columns=["close"],
    )
    if bench.empty:
        return pd.Series(dtype=float, name="beta")
    bench = bench.sort_values("date")
    bench["R"] = np.log(bench["close"]).diff()
    bench = bench[["date", "R"]].reset_index(drop=True)
    bench_dates = pd.DatetimeIndex(bench["date"])

    # Align each symbol to the bench trading calendar so suspensions become
    # explicit NaNs in r. Without this reindex, the per-symbol .diff() spans
    # multiple calendar days across a suspension (Jan 4 → Jan 6 with Jan 5
    # missing), and the resulting return is paired against the bench's
    # single-day return — numerically wrong.
    sym_pieces: list[pd.DataFrame] = []
    for sym, g in df.groupby("symbol", sort=False):
        sub = g.set_index("date")[["adj_close"]].reindex(bench_dates)
        sub.index.name = "date"
        sub["r"] = np.log(sub["adj_close"]).diff()
        sub["symbol"] = sym
        sym_pieces.append(sub.reset_index())
    if not sym_pieces:
        return pd.Series(dtype=float, name="beta")
    df = pd.concat(sym_pieces, ignore_index=True)
    df = df.merge(bench, on="date", how="left")
    df = df.sort_values(["symbol", "date"])

    weights = halflife_weights(BETA_WINDOW, BETA_HALFLIFE)

    def _one(g: pd.DataFrame) -> pd.Series:
        beta = _vectorized_wls_beta(
            g["r"].to_numpy(),
            g["R"].to_numpy(),
            weights,
        )
        return pd.Series(beta, index=g.index)

    df["beta"] = df.groupby("symbol", group_keys=False)[["r", "R"]].apply(_one)
    # Reindex-introduced rows have adj_close NaN — drop them so the returned
    # Series carries one row per actually-traded (date, symbol) only.
    df = df.dropna(subset=["adj_close"])
    return to_panel_series(df, "beta", name="beta")
