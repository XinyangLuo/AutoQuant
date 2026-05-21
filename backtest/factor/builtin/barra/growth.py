"""Barra Growth factor — EGRO.

``EGRO = slope(last 20 quarterly EPS on time) / |mean(EPS)|``. Positive
direction so faster earnings growth ⇒ higher quality (opposite sign to AGRO).
EPS uses ``inc_basic_eps``; we keep YTD-as-reported and take the slope on
those values rather than reconstructing per-quarter EPS, mirroring the
asset-growth treatment in AGRO.
"""

from __future__ import annotations

import pandas as pd

from backtest.factor.builtin.barra._common import pit_quarterly_slope, to_panel_series
from backtest.factor.registry import register
from backtest.factor.variants import BARRA_L3_VARIANT, CATEGORY_BARRA_L3


@register(
    "f_barra_growth_egro",
    name="Barra Growth — EGRO",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily", "income_q"],
    description=(
        "Slope of last 20 quarterly basic_eps on time, divided by |mean(EPS)|. "
        "Positive direction: faster EPS growth ⇒ higher score."
    ),
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_growth_egro(panel: pd.DataFrame) -> pd.Series:
    scored = pit_quarterly_slope(
        panel[["date", "symbol", "inc_basic_eps", "end_date"]],
        value_col="inc_basic_eps",
        n=20,
        sign=1.0,
    )
    return to_panel_series(scored, "value", name="egro")
