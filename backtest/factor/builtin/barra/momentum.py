"""Barra Momentum factor — RSTR.

CNE6 definition: every day compute ``ln(1 + r_t)``, take an EWMA with
half-life 126 days over the trailing 252 days, lag the result by 11
trading days, then equal-weight average the lagged value over 11 days
(``T-21 ... T-11``). The lag + smoothing kills the short-term reversal that
would otherwise contaminate medium-term momentum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.builtin.barra._common import (
    halflife_weights,
    log_return,
    to_panel_series,
)
from backtest.factor.registry import register
from backtest.factor.variants import BARRA_L3_VARIANT, CATEGORY_BARRA_L3

RSTR_WINDOW = 252
RSTR_HALFLIFE = 126
RSTR_LAG = 11
RSTR_SMOOTH = 11


def _ewm_log_return_sum(log_ret: pd.Series, window: int, halflife: int) -> pd.Series:
    """Exponentially-weighted *sum* over a rolling window.

    pandas ``ewm`` is mean-style and unbounded; CNE6 momentum wants a
    finite-window weighted sum. Equivalent to ``ewma(window) * sum(weights)``,
    with a rescale that keeps the scale stable across NaN-skipped windows.
    """
    weights = halflife_weights(window, halflife)
    sw = weights.sum()

    def _kernel(buf: np.ndarray) -> float:
        mask = ~np.isnan(buf)
        if mask.sum() < window // 2:
            return np.nan
        clean = np.where(mask, buf, 0.0)
        wm = weights * mask
        denom = wm.sum()
        if denom <= 0:
            return np.nan
        return (weights * clean).sum() * (sw / denom)

    return log_ret.rolling(window, min_periods=window).apply(_kernel, raw=True)


@register(
    "f_barra_momentum_rstr",
    name="Barra Momentum — RSTR",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily"],
    description=(
        f"EWMA(window={RSTR_WINDOW}, half-life={RSTR_HALFLIFE}) of ln(1+r_t), "
        f"lagged {RSTR_LAG}d then {RSTR_SMOOTH}d equal-weight smoothed."
    ),
    variant=BARRA_L3_VARIANT,
    frequency="D",
    parameters={"window": RSTR_WINDOW + RSTR_LAG + RSTR_SMOOTH},
)
def barra_momentum_rstr(panel: pd.DataFrame, window: int | None = None) -> pd.Series:
    del window
    df = panel[["date", "symbol", "close", "adj_factor"]].copy()
    df["adj_close"] = df["close"] * df["adj_factor"]
    df = df.sort_values(["symbol", "date"])
    df["log_ret"] = log_return(df, "adj_close")

    def _one(s: pd.Series) -> pd.Series:
        ewma_sum = _ewm_log_return_sum(s, RSTR_WINDOW, RSTR_HALFLIFE)
        lagged = ewma_sum.shift(RSTR_LAG)
        return lagged.rolling(RSTR_SMOOTH, min_periods=RSTR_SMOOTH // 2).mean()

    df["rstr"] = df.groupby("symbol", group_keys=False)["log_ret"].apply(_one)
    return to_panel_series(df, "rstr", name="rstr")
