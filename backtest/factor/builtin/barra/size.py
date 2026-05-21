"""Barra Size factor — log of float-share market cap.

L3: ``f_barra_size_lncap = ln(circ_mv)`` (units: 万元 from Tushare).
L1: ``f_barra_size = f_barra_size_lncap`` (single-L3 wrapper, no averaging).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.builtin.barra._common import to_panel_series
from backtest.factor.registry import register
from backtest.factor.variants import BARRA_L3_VARIANT


@register(
    "f_barra_size_lncap",
    name="Barra Size — LNCAP",
    category="barra_l3",
    data_sources=["market_daily"],
    description="ln(circ_mv). Floating-share market cap in log space.",
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_size_lncap(panel: pd.DataFrame) -> pd.Series:
    df = panel[["date", "symbol", "circ_mv"]]
    cap = df["circ_mv"].where(df["circ_mv"] > 0, np.nan)
    return to_panel_series(df, np.log(cap.values), name="lncap")
