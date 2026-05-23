"""Barra Value factor — internal helpers for ``f_barra_value`` composite.

* **BTOP** ``= book_equity / circ_mv``. Book equity = ``bs_total_hldr_eqy_inc_min_int``
  (公司全部所有者权益, includes minority interest, as Barra does).
* **ETOP** ``= ttm_net_income / circ_mv``. Net income TTM via
  `transforms.ttm` (Shi Chuan formula: current + LY_FY − LY_same, fallback
  annualize for the first year of history).
* **DTOP** ``= ttm_cash_dividend_per_share / prev_close``. Sum of cash_div
  with ``ex_date`` in the trailing 365 days, divided by ``pre_close``.

The L1 composite ``f_barra_value`` z-scores each via the standard L3
pipeline (MAD → industry median fill → cs_zscore) and equal-weight averages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.builtin.barra._common import (
    latest_quarter_per_day,
    to_panel_series,
)
from backtest.factor.transforms import ttm

DIVIDEND_LOOKBACK_DAYS = 400


def barra_value_btop(panel: pd.DataFrame) -> pd.Series:
    df = latest_quarter_per_day(
        panel[["date", "symbol", "circ_mv", "bs_total_hldr_eqy_inc_min_int", "end_date"]]
    )
    # circ_mv: 万元; balance-sheet figures: 元.
    book = df["bs_total_hldr_eqy_inc_min_int"]
    cap_yuan = df["circ_mv"] * 1e4
    btop = book / cap_yuan.where(cap_yuan > 0, np.nan)
    return to_panel_series(df, btop.values, name="btop")


def barra_value_etop(panel: pd.DataFrame) -> pd.Series:
    cols = ["date", "symbol", "circ_mv", "inc_n_income_attr_p", "end_date"]
    sub = panel[cols].copy()
    sub["inc_n_income_attr_p_ttm"] = ttm(sub, "inc_n_income_attr_p", kind="flow")
    df = latest_quarter_per_day(sub)
    cap_yuan = df["circ_mv"] * 1e4
    etop = df["inc_n_income_attr_p_ttm"] / cap_yuan.where(cap_yuan > 0, np.nan)
    return to_panel_series(df, etop.values, name="etop")


def barra_value_dtop(
    panel: pd.DataFrame,
    *,
    market_storage: MarketStorage,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    # When the panel is a PIT-concat (multiple end_date rows per (date, symbol)),
    # we only need one row per (date, symbol) for the per-day pre_close lookup.
    df = panel[["date", "symbol", "pre_close"]].drop_duplicates(["date", "symbol"]).copy()
    df = df.sort_values(["symbol", "date"])

    # Extend the dividend fetch ~400 days before start_date so the cumsum
    # baseline is properly anchored: an early trade date in the range needs
    # ex-dividends from up to 365 days earlier to compute its TTM correctly.
    div_start = start_date
    if start_date:
        div_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=DIVIDEND_LOOKBACK_DAYS)
        ).strftime("%Y%m%d")

    div = market_storage.get_dividends(start=div_start, end=end_date)
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
