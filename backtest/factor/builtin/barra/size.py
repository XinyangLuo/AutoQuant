"""Barra Size factor — internal helper for the L1 ``f_barra_size`` composite.

``barra_size_lncap(panel) = ln(circ_mv)`` (units: 万元 from Tushare).
The L1 composite applies the standard L3 pipeline (MAD → industry median
fill → cs_zscore) to this raw series and returns it as-is — Size is a
single-input composite, so the average step is identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.builtin.barra._common import to_panel_series


def barra_size_lncap(panel: pd.DataFrame) -> pd.Series:
    df = panel[["date", "symbol", "circ_mv"]]
    cap = df["circ_mv"].where(df["circ_mv"] > 0, np.nan)
    return to_panel_series(df, np.log(cap.values), name="lncap")
