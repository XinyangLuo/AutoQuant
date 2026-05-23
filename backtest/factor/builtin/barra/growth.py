"""Barra Growth factor — internal helper for ``f_barra_growth``.

``EGRO = slope(last 20 quarterly TTM EPS on time) / |mean(TTM EPS)|``.
Positive direction so faster earnings growth ⇒ higher quality (opposite
sign to AGRO). Input is TTM EPS rather than as-reported YTD cumulative EPS
— the latter is sawtooth-shaped (Q1, H1, 9M, FY, Q1, …) and biases the
OLS slope; TTM smooths the seasonality so the slope reflects multi-year
growth rather than within-year accumulation.

**Event-driven**: slope only changes when a new ``f_ann_date``
announcement supersedes the previous one (~4 events/year/symbol).
Compute one scalar per event then range-join to trade dates.
"""

from __future__ import annotations

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.builtin.barra._common import (
    event_slope_over_mean,
    event_ttm,
    expand_events_to_dates,
)


def barra_growth_egro(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Event-driven EGRO. ``panel`` is unused (kept for signature parity).

    TTM flow needs LY_FY and LY_same end_dates (≤ 1 year prior to each
    history quarter). Fetch 20 + 4 quarters so the earliest of the 20
    slope inputs still has its LY lookups inside the panel, then trim
    to 20 TTM values per event before regressing.
    """
    events = market_storage.get_fina_event_panel(
        start=start_date, end=end_date,
        columns=["inc_basic_eps"], last_n_quarters=24,
    )
    if events.empty:
        return pd.Series(dtype=float, name="egro").rename_axis(["date", "symbol"])

    history_with_ttm = event_ttm(events, "inc_basic_eps")
    # Keep the most recent 20 TTM quarters per event for the regression.
    history_with_ttm = history_with_ttm.sort_values(
        ["symbol", "announce_end_date", "end_date"]
    )
    history_with_ttm = history_with_ttm.groupby(
        ["symbol", "announce_end_date"], sort=False,
    ).tail(20)

    scored = event_slope_over_mean(
        history_with_ttm, "inc_basic_eps_ttm", n=20, sign=1.0,
    )
    trade_dates = get_trade_dates(start_date, end_date)
    return expand_events_to_dates(
        scored, pd.Series(trade_dates),
        market_storage=market_storage, name="egro",
    )
