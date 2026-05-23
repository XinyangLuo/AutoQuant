"""Barra Liquidity factor — internal helper for ``f_barra_liquidity``.

``STOM = ln(sum over last 21 trade days of amount_t / circ_mv_t)``.

Tushare daily ``amount`` is in 千元 and ``circ_mv`` in 万元, so rescale
before the ratio. We require the full 21-day window to avoid biasing
shorter-history names downward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.builtin.barra._common import to_panel_series

STOM_WINDOW = 21


def barra_liquidity_stom(panel: pd.DataFrame, window: int = STOM_WINDOW) -> pd.Series:
    df = panel[["date", "symbol", "amount", "circ_mv"]].copy()
    df = df.sort_values(["symbol", "date"])

    ratio = (df["amount"] * 1e3) / (df["circ_mv"] * 1e4).where(df["circ_mv"] > 0, np.nan)
    df["ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    df["roll_sum"] = (
        df.groupby("symbol")["ratio"]
          .transform(lambda s: s.rolling(window, min_periods=window).sum())
    )

    stom = np.log(df["roll_sum"].where(df["roll_sum"] > 0, np.nan))
    return to_panel_series(df, stom.values, name="stom")
