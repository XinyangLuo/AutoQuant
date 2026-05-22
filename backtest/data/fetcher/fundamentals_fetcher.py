"""Tushare pro.income / pro.balancesheet / pro.cashflow fetchers.

Each table is fetched independently.  All consolidated report types
(report_type in {'1','2','3','4','5'}) are kept; parent-company types
(6/11/12) are dropped.  The shared key columns are renamed from
ts_code → symbol.

See `backtest/data/DESIGN.md` §"财报数据使用指南" §1 for the report_type
taxonomy and the rationale for keeping the full consolidated family.
"""

from functools import partial

import pandas as pd

from backtest.data.tushare_client import fetch_and_transform, pro


CONSOLIDATED_REPORT_TYPES = frozenset({"1", "2", "3", "4", "5"})


def _keep_consolidated(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only consolidated-family rows and rename ts_code."""
    if df.empty:
        return df
    if "report_type" in df.columns:
        mask = df["report_type"].astype(str).isin(CONSOLIDATED_REPORT_TYPES)
        df = df.loc[mask]
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
