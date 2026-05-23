"""Barra Quality factor — internal helpers for ``f_barra_quality``.

* **ROA** ``= ttm_net_income / latest_total_assets``. Net income TTM via
  Shi Chuan formula: current + LY_FY − LY_same, fallback annualize.
* **GP** ``= (ttm_revenue - ttm_oper_cost) / latest_total_assets``.
* **AGRO** ``= -slope(last 20 quarterly total_assets) / |mean|``.

**Event-driven**: all three sub-factors only change on f_ann_date
announcements (~4 events/year/symbol). Compute one scalar per event
per sub-factor, then range-join to trade dates. The composite
:func:`barra_quality` fetches the inc / bs event panels once and shares
them across ROA/GP/AGRO — without sharing each sub-factor re-runs the
same SQL.
"""

from __future__ import annotations

from typing import Callable

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


def _ttm_over_assets(
    inc_events: pd.DataFrame,
    bs_events: pd.DataFrame,
    ttm_cols: list[str],
    numerator: Callable[[pd.DataFrame], pd.Series],
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
    name: str,
) -> pd.Series:
    """Shared body of ROA / GP: TTM of one or more income cols, then divide
    by latest total_assets, then expand events to trade dates.

    ``numerator(df)`` is invoked on the per-event frame after every TTM
    column has been added (``f"{c}_ttm"``) and produces the numerator
    Series (e.g. ``df['inc_revenue_ttm'] - df['inc_oper_cost_ttm']``).
    """
    if inc_events.empty or bs_events.empty:
        return pd.Series(dtype=float, name=name).rename_axis(["date", "symbol"])

    panel = inc_events
    for col in ttm_cols:
        panel = event_ttm(panel, col)

    # Collapse to one row per (symbol, announce_end_date): the row whose
    # history end_date IS the announcement quarter carries the current
    # TTM values for that event.
    cur_mask = panel["announce_end_date"].to_numpy() == panel["end_date"].to_numpy()
    keep_cols = ["symbol", "announce_end_date", "f_ann_date", "next_f_ann_date"]
    keep_cols += [f"{c}_ttm" for c in ttm_cols]
    per_event = panel.loc[cur_mask, keep_cols].reset_index(drop=True)

    ta = event_latest_value(bs_events, "bs_total_assets").rename(
        columns={"value": "bs_total_assets"},
    )[["symbol", "announce_end_date", "bs_total_assets"]]
    merged = per_event.merge(ta, on=["symbol", "announce_end_date"], how="left")

    assets = merged["bs_total_assets"].where(merged["bs_total_assets"] > 0, np.nan)
    merged["value"] = numerator(merged) / assets

    trade_dates = get_trade_dates(start_date, end_date)
    return expand_events_to_dates(
        merged[["symbol", "f_ann_date", "next_f_ann_date", "value"]],
        pd.Series(trade_dates),
        market_storage=market_storage, name=name,
    )


def barra_quality_roa(
    *,
    inc_events: pd.DataFrame,
    bs_events: pd.DataFrame,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    return _ttm_over_assets(
        inc_events, bs_events,
        ttm_cols=["inc_n_income_attr_p"],
        numerator=lambda df: df["inc_n_income_attr_p_ttm"],
        market_storage=market_storage,
        start_date=start_date, end_date=end_date, name="roa",
    )


def barra_quality_gp(
    *,
    inc_events: pd.DataFrame,
    bs_events: pd.DataFrame,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    return _ttm_over_assets(
        inc_events, bs_events,
        ttm_cols=["inc_revenue", "inc_oper_cost"],
        numerator=lambda df: df["inc_revenue_ttm"] - df["inc_oper_cost_ttm"],
        market_storage=market_storage,
        start_date=start_date, end_date=end_date, name="gp",
    )


def barra_quality_agro(
    *,
    bs_events: pd.DataFrame,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    if bs_events.empty:
        return pd.Series(dtype=float, name="agro").rename_axis(["date", "symbol"])
    scored = event_slope_over_mean(bs_events, "bs_total_assets", n=20, sign=-1.0)
    trade_dates = get_trade_dates(start_date, end_date)
    return expand_events_to_dates(
        scored, pd.Series(trade_dates),
        market_storage=market_storage, name="agro",
    )
