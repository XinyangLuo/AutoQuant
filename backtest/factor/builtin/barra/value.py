"""Barra Value factor — BTOP, ETOP, DTOP.

* **BTOP** ``= book_equity / circ_mv``. Book equity = ``bs_total_hldr_eqy_inc_min_int``
  (公司全部所有者权益, includes minority interest, as Barra does).
* **ETOP** ``= ttm_net_income / circ_mv``. Net income trailing 12 months,
  approximated by annualizing the latest reported YTD cumulative net income
  (Q1 ×4, Q2 ×2, Q3 ×4/3, Q4 ×1).
* **DTOP** ``= ttm_cash_dividend_per_share / prev_close``. Sum of cash_div
  with ``ex_date`` in the trailing 365 days, divided by ``pre_close``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.builtin.barra._common import (
    annualize_ytd,
    latest_quarter_per_day,
    to_panel_series,
)
from backtest.factor.registry import register
from backtest.factor.variants import BARRA_L3_VARIANT, CATEGORY_BARRA_L3

DIVIDEND_LOOKBACK_DAYS = 400


@register(
    "f_barra_value_btop",
    name="Barra Value — BTOP",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily", "balancesheet_q"],
    description="Book equity (incl. minority interest) / floating market cap.",
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_value_btop(panel: pd.DataFrame) -> pd.Series:
    df = latest_quarter_per_day(
        panel[["date", "symbol", "circ_mv", "bs_total_hldr_eqy_inc_min_int", "end_date"]]
    )
    # circ_mv: 万元; balance-sheet figures: 元.
    book = df["bs_total_hldr_eqy_inc_min_int"]
    cap_yuan = df["circ_mv"] * 1e4
    btop = book / cap_yuan.where(cap_yuan > 0, np.nan)
    return to_panel_series(df, btop.values, name="btop")


@register(
    "f_barra_value_etop",
    name="Barra Value — ETOP",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily", "income_q"],
    description=(
        "Annualized-YTD net income attributable to parent / floating market cap. "
        "Approximates Barra trailing-12-month earnings yield."
    ),
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_value_etop(panel: pd.DataFrame) -> pd.Series:
    df = latest_quarter_per_day(
        panel[["date", "symbol", "circ_mv", "inc_n_income_attr_p", "end_date"]]
    )
    ttm_income = annualize_ytd(df["inc_n_income_attr_p"], df["end_date"])
    cap_yuan = df["circ_mv"] * 1e4
    etop = ttm_income / cap_yuan.where(cap_yuan > 0, np.nan)
    return to_panel_series(df, etop.values, name="etop")


@register(
    "f_barra_value_dtop",
    name="Barra Value — DTOP",
    category=CATEGORY_BARRA_L3,
    data_sources=["market_daily"],
    description=(
        "Trailing-12m cash dividend / pre_close. Reads dividends event table "
        "directly from MarketStorage rather than through panel."
    ),
    variant=BARRA_L3_VARIANT,
    frequency="D",
)
def barra_value_dtop(
    panel: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    df = panel[["date", "symbol", "pre_close"]].copy()
    df = df.sort_values(["symbol", "date"])

    # Extend the dividend fetch ~400 days before start_date so the cumsum
    # baseline is properly anchored: an early trade date in the range needs
    # ex-dividends from up to 365 days earlier to compute its TTM correctly.
    div_start = start_date
    if start_date:
        div_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=DIVIDEND_LOOKBACK_DAYS)
        ).strftime("%Y%m%d")

    with MarketStorage() as ms:
        div = ms.get_dividends(start=div_start, end=end_date)
    if div.empty:
        return to_panel_series(df, np.nan, name="dtop")

    div = div[div["div_proc"].astype(str).str.startswith("实施")].copy()
    div["ex_date"] = pd.to_datetime(div["ex_date"], format="%Y%m%d", errors="coerce").astype("datetime64[ns]")
    div = div.dropna(subset=["ex_date", "cash_div"])

    div = div.sort_values(["symbol", "ex_date"])
    div["cum_cash"] = div.groupby("symbol")["cash_div"].cumsum()
    div = div.sort_values("ex_date")

    df["date_dt"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df = df.sort_values("date_dt")

    merged = pd.merge_asof(
        df, div[["symbol", "ex_date", "cum_cash"]],
        left_on="date_dt", right_on="ex_date", by="symbol",
        direction="backward",
    ).rename(columns={"cum_cash": "cum_now"})

    df_lag = df.copy()
    df_lag["lag_date"] = df_lag["date_dt"] - pd.Timedelta(days=365)
    df_lag = df_lag.sort_values("lag_date")
    merged_lag = pd.merge_asof(
        df_lag, div[["symbol", "ex_date", "cum_cash"]],
        left_on="lag_date", right_on="ex_date", by="symbol",
        direction="backward",
    ).rename(columns={"cum_cash": "cum_lag"})

    out = merged.merge(
        merged_lag[["date_dt", "symbol", "cum_lag"]],
        on=["date_dt", "symbol"], how="left",
    )
    ttm_div = out["cum_now"].fillna(0.0) - out["cum_lag"].fillna(0.0)

    denom = out["pre_close"].where(out["pre_close"] > 0, np.nan)
    dtop = ttm_div / denom

    return to_panel_series(out, dtop.values, name="dtop")
