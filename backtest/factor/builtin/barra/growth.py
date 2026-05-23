"""Barra Growth factor — internal helper for ``f_barra_growth``.

``EGRO = slope(last 20 quarterly TTM EPS on time) / |mean(TTM EPS)|``.
Positive direction so faster earnings growth ⇒ higher quality (opposite
sign to AGRO). Input is TTM EPS rather than as-reported YTD cumulative EPS
— the latter is sawtooth-shaped (Q1, H1, 9M, FY, Q1, …) and biases the
OLS slope; TTM smooths the seasonality so the slope reflects multi-year
growth rather than within-year accumulation.
"""

from __future__ import annotations

import pandas as pd

from backtest.factor.builtin.barra._common import pit_quarterly_slope, to_panel_series
from backtest.factor.transforms import ttm


def barra_growth_egro(panel: pd.DataFrame) -> pd.Series:
    sub = panel[["date", "symbol", "inc_basic_eps", "end_date"]].copy()
    sub["inc_basic_eps_ttm"] = ttm(sub, "inc_basic_eps", kind="flow")
    # No latest_quarter_per_day here: pit_quarterly_slope reads the full
    # multi-quarter history per (date, symbol) and OLS-regresses it.
    scored = pit_quarterly_slope(
        sub[["date", "symbol", "inc_basic_eps_ttm", "end_date"]],
        value_col="inc_basic_eps_ttm",
        n=20,
        sign=1.0,
    )
    return to_panel_series(scored, "value", name="egro")
