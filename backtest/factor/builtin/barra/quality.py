"""Barra Quality factor — ROA, GP, AGRO.

* **ROA** ``= ttm_net_income / latest_total_assets``. Net income TTM via
  `transforms.ttm` (Shi Chuan formula: current + LY_FY − LY_same, fallback
  annualize). Total assets uses the latest reported balance-sheet value
  (point-in-time snapshot).
* **GP** ``= (ttm_revenue - ttm_oper_cost) / latest_total_assets``.
* **AGRO** ``= -k / |mean(TA)|`` where ``k`` is the OLS slope of last 20
  quarterly ``total_assets`` on time. Negative sign so faster expansion is
  *lower* quality (the Barra convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.builtin.barra._common import (
    latest_quarter_per_day,
    pit_quarterly_slope,
    to_panel_series,
)
from backtest.factor.registry import register
from backtest.factor.transforms import ttm
from backtest.factor.variants import BARRA_L3_VARIANT, CATEGORY_BARRA_L3


@register(
    "f_barra_quality_roa",
    name="Barra Quality — ROA",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily", "income_q", "balancesheet_q"],
    description="TTM net income (attr. to parent) / latest total assets.",
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_quality_roa(panel: pd.DataFrame) -> pd.Series:
    cols = ["date", "symbol", "inc_n_income_attr_p", "bs_total_assets", "end_date"]
    sub = panel[cols].copy()
    sub["inc_n_income_attr_p_ttm"] = ttm(sub, "inc_n_income_attr_p", kind="flow")
    df = latest_quarter_per_day(sub)
    assets = df["bs_total_assets"].where(df["bs_total_assets"] > 0, np.nan)
    roa = df["inc_n_income_attr_p_ttm"] / assets
    return to_panel_series(df, roa.values, name="roa")


@register(
    "f_barra_quality_gp",
    name="Barra Quality — GP",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily", "income_q", "balancesheet_q"],
    description="(TTM revenue - TTM oper_cost) / latest total assets.",
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_quality_gp(panel: pd.DataFrame) -> pd.Series:
    cols = ["date", "symbol", "inc_revenue", "inc_oper_cost", "bs_total_assets", "end_date"]
    sub = panel[cols].copy()
    sub["inc_revenue_ttm"] = ttm(sub, "inc_revenue", kind="flow")
    sub["inc_oper_cost_ttm"] = ttm(sub, "inc_oper_cost", kind="flow")
    df = latest_quarter_per_day(sub)
    assets = df["bs_total_assets"].where(df["bs_total_assets"] > 0, np.nan)
    gp = (df["inc_revenue_ttm"] - df["inc_oper_cost_ttm"]) / assets
    return to_panel_series(df, gp.values, name="gp")


@register(
    "f_barra_quality_agro",
    name="Barra Quality — AGRO",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily", "balancesheet_q"],
    description=(
        "-1 × (slope of last 20 quarterly total_assets on time) / |mean(total_assets)|. "
        "Negative so faster asset growth ⇒ lower quality."
    ),
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_quality_agro(panel: pd.DataFrame) -> pd.Series:
    scored = pit_quarterly_slope(
        panel[["date", "symbol", "bs_total_assets", "end_date"]],
        value_col="bs_total_assets",
        n=20,
        sign=-1.0,
    )
    return to_panel_series(scored, "value", name="agro")
