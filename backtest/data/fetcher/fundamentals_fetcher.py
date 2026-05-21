"""Tushare pro.income / pro.balancesheet / pro.cashflow fetchers.

Each table is fetched independently.  All rows with report_type='1'
(consolidated statements) are kept; other report types are dropped.
The shared key columns are renamed from ts_code → symbol.
"""

from functools import partial

import pandas as pd

from backtest.data.tushare_client import fetch_and_transform, pro


CONSOLIDATED = "1"


def _keep_consolidated(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only consolidated report rows (report_type='1') and rename ts_code."""
    if df.empty:
        return df
    df = df[df.get("report_type", CONSOLIDATED).astype(str) == CONSOLIDATED]
    if "ts_code" in df.columns:
        df = df.rename(columns={"ts_code": "symbol"})
    return df.dropna(subset=["symbol", "end_date"])


def _fetch_by_symbol(api_func, symbol: str) -> pd.DataFrame:
    """Fetch full history for one stock."""
    return fetch_and_transform(api_func, _keep_consolidated, ts_code=symbol)


def _fetch_by_f_ann_date(api_func, date: str) -> pd.DataFrame:
    """Fetch all rows whose f_ann_date equals *date* (YYYYMMDD)."""
    return fetch_and_transform(api_func, _keep_consolidated, f_ann_date=date)


# Convenience bindings for each table

fetch_income_by_symbol = partial(_fetch_by_symbol, pro.income)
fetch_balancesheet_by_symbol = partial(_fetch_by_symbol, pro.balancesheet)
fetch_cashflow_by_symbol = partial(_fetch_by_symbol, pro.cashflow)

fetch_income_by_f_ann_date = partial(_fetch_by_f_ann_date, pro.income)
fetch_balancesheet_by_f_ann_date = partial(_fetch_by_f_ann_date, pro.balancesheet)
fetch_cashflow_by_f_ann_date = partial(_fetch_by_f_ann_date, pro.cashflow)
