"""Barra Quality factor — internal helpers for ``f_barra_quality``.

* **ROA** ``= ttm_net_income / latest_total_assets``. Net income TTM via
  Shi Chuan formula: current + LY_FY − LY_same, fallback annualize.
* **GP** ``= (ttm_revenue - ttm_oper_cost) / latest_total_assets``.
* **AGRO** ``= -slope(last 20 quarterly total_assets) / |mean|``.

**Event-driven**: all three sub-factors only change on f_ann_date
announcements (~4 events/year/symbol). Compute one scalar per event
per sub-factor, then range-join to trade dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.builtin.barra._common import (
    event_latest_value,
    event_slope_over_mean,
    event_ttm,
    expand_events_to_dates,
)


def _merge_inc_bs_events(
    inc_events: pd.DataFrame, bs_per_event: pd.DataFrame,
) -> pd.DataFrame:
    """Join an inc event panel with one row-per-event of bs ``total_assets``.

    Matches on (symbol, announce_end_date). The two tables have separate
    PIT histories — Tushare can publish income and balancesheet on
    different f_ann_dates for the same end_date — but our event scaffold
    sources both from each table's own ``get_fina_event_panel`` and
    aligns on ``announce_end_date``. The result inherits inc's
    ``(f_ann_date, next_f_ann_date)`` interval since that's the timing
    that decides when ROA/GP are recomputed for the alpha pipeline.
    """
    bs_per_event = bs_per_event[[
        "symbol", "announce_end_date", "value",
    ]].rename(columns={"value": "bs_total_assets"})
    return inc_events.merge(
        bs_per_event, on=["symbol", "announce_end_date"], how="left",
    )


def barra_quality_roa(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Event-driven ROA = TTM net income / latest total assets."""
    inc_events = market_storage.get_fina_event_panel(
        start=start_date, end=end_date,
        columns=["inc_n_income_attr_p"], last_n_quarters=5,
    )
    bs_events = market_storage.get_fina_event_panel(
        start=start_date, end=end_date,
        columns=["bs_total_assets"], last_n_quarters=1,
    )
    if inc_events.empty or bs_events.empty:
        return pd.Series(dtype=float, name="roa").rename_axis(["date", "symbol"])

    ni_ttm = event_ttm(inc_events, "inc_n_income_attr_p")
    # Reduce to one row per event (announce_end_date == end_date row).
    cur_mask = ni_ttm["announce_end_date"].to_numpy() == ni_ttm["end_date"].to_numpy()
    ni_per_event = ni_ttm.loc[cur_mask, [
        "symbol", "announce_end_date", "f_ann_date", "next_f_ann_date",
        "inc_n_income_attr_p_ttm",
    ]].reset_index(drop=True)

    ta = event_latest_value(bs_events, "bs_total_assets")
    merged = _merge_inc_bs_events(ni_per_event, ta)

    assets = merged["bs_total_assets"].where(merged["bs_total_assets"] > 0, np.nan)
    merged["value"] = merged["inc_n_income_attr_p_ttm"] / assets

    trade_dates = get_trade_dates(start_date, end_date)
    return expand_events_to_dates(
        merged[["symbol", "f_ann_date", "next_f_ann_date", "value"]],
        pd.Series(trade_dates), name="roa",
    )


def barra_quality_gp(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Event-driven GP = (TTM revenue − TTM oper_cost) / latest total assets."""
    inc_events = market_storage.get_fina_event_panel(
        start=start_date, end=end_date,
        columns=["inc_revenue", "inc_oper_cost"], last_n_quarters=5,
    )
    bs_events = market_storage.get_fina_event_panel(
        start=start_date, end=end_date,
        columns=["bs_total_assets"], last_n_quarters=1,
    )
    if inc_events.empty or bs_events.empty:
        return pd.Series(dtype=float, name="gp").rename_axis(["date", "symbol"])

    rev_ttm = event_ttm(inc_events, "inc_revenue")
    cost_ttm = event_ttm(rev_ttm, "inc_oper_cost")
    cur_mask = cost_ttm["announce_end_date"].to_numpy() == cost_ttm["end_date"].to_numpy()
    inc_per_event = cost_ttm.loc[cur_mask, [
        "symbol", "announce_end_date", "f_ann_date", "next_f_ann_date",
        "inc_revenue_ttm", "inc_oper_cost_ttm",
    ]].reset_index(drop=True)

    ta = event_latest_value(bs_events, "bs_total_assets")
    merged = _merge_inc_bs_events(inc_per_event, ta)

    assets = merged["bs_total_assets"].where(merged["bs_total_assets"] > 0, np.nan)
    merged["value"] = (
        merged["inc_revenue_ttm"] - merged["inc_oper_cost_ttm"]
    ) / assets

    trade_dates = get_trade_dates(start_date, end_date)
    return expand_events_to_dates(
        merged[["symbol", "f_ann_date", "next_f_ann_date", "value"]],
        pd.Series(trade_dates), name="gp",
    )


def barra_quality_agro(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Event-driven AGRO = -slope(20 quarterly total_assets) / |mean|."""
    bs_events = market_storage.get_fina_event_panel(
        start=start_date, end=end_date,
        columns=["bs_total_assets"], last_n_quarters=20,
    )
    if bs_events.empty:
        return pd.Series(dtype=float, name="agro").rename_axis(["date", "symbol"])

    scored = event_slope_over_mean(
        bs_events, "bs_total_assets", n=20, sign=-1.0,
    )
    trade_dates = get_trade_dates(start_date, end_date)
    return expand_events_to_dates(scored, pd.Series(trade_dates), name="agro")
