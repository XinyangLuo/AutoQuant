"""Tushare cyq_chips fetcher — 筹码分布."""

from __future__ import annotations

import time

import pandas as pd
import tushare as ts

from backtest.data.tushare_client import pro


def fetch_cyq_for_date(
    trade_date: str,
    ts_code: str,
) -> pd.DataFrame:
    """Fetch cyq_chips for a single date and one symbol.

    Tushare ``pro.cyq_chips`` requires ``ts_code`` and returns long-format rows:
    ``[ts_code, trade_date, price, percent]``.

    Parameters
    ----------
    trade_date : str
        YYYYMMDD.
    ts_code : str
        Required — Tushare does not support bulk fetch for this endpoint.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, symbol, price, percent]`` with ``symbol`` renamed
        from ``ts_code``.
    """
    df = pro.cyq_chips(ts_code=ts_code, trade_date=trade_date)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "symbol", "price", "percent"])

    df = df.rename(columns={"ts_code": "symbol", "trade_date": "date"})
    df["date"] = trade_date
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["percent"] = pd.to_numeric(df["percent"], errors="coerce")

    # Warn on NaN from coerced values so bad upstream data is visible
    nan_price = df["price"].isna().sum()
    nan_pct = df["percent"].isna().sum()
    if nan_price or nan_pct:
        print(
            f"[cyq_fetcher] WARN: {ts_code} on {trade_date}: "
            f"{nan_price} NaN price(s), {nan_pct} NaN percent(s) "
            f"after coercion — upstream data may be malformed"
        )

    return df[["date", "symbol", "price", "percent"]]


# Tushare empirically caps cyq_chips at ~6000 rows.  High-bin stocks (~175
# bins/day) therefore cap at ~34 trade days before truncation.  We flag it
# so backfill callers can shrink their chunks if needed.
_CYQ_ROW_CAP = 6_000

def fetch_cyq_for_symbol_range(
    ts_code: str,
    start_date: str,
    end_date: str,
) -> list[pd.DataFrame]:
    """Fetch cyq_chips for one symbol across a date range.

    Tushare cyq_chips supports ``start_date`` / ``end_date`` when
    ``ts_code`` is provided.  Returns one or more DataFrames — the fetcher
    auto-chunks on Tushare's ~6000-row cap so the caller always gets the
    full set.

    Each returned DataFrame has columns ``[date, symbol, price, percent]``.
    """
    dfs: list[pd.DataFrame] = []
    chunk_start = start_date
    while True:
        df = pro.cyq_chips(
            ts_code=ts_code,
            start_date=chunk_start,
            end_date=end_date,
        )
        if df is None or df.empty:
            break

        n = len(df)
        df = df.rename(columns={"ts_code": "symbol", "trade_date": "date"})
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["percent"] = pd.to_numeric(df["percent"], errors="coerce")
        df = df[["date", "symbol", "price", "percent"]]
        dfs.append(df)

        if n < _CYQ_ROW_CAP:
            break  # not truncated
        # Pick up from the day after the last date in this chunk
        max_d = df["date"].max()
        if max_d >= end_date:
            break
        chunk_start = max_d

    if not dfs:
        return [pd.DataFrame(columns=["date", "symbol", "price", "percent"])]
    return dfs


def fetch_cyq_all_symbols(
    trade_date: str,
    symbols: list[str],
    sleep_sec: float = 0.05,
) -> pd.DataFrame:
    """Batch-fetch cyq_chips for multiple symbols on one date — serial loop.

    Tushare rate limit varies by account tier.  Default ``sleep_sec=0.05``
    suits high-tier (5000+ pts) accounts; raise it for basic tiers.
    Sleep is applied **before** each request to avoid bursting the first call.
    """
    pieces: list[pd.DataFrame] = []
    failed: list[str] = []

    for sym in symbols:
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        try:
            df = fetch_cyq_for_date(trade_date, sym)
            if not df.empty:
                pieces.append(df)
        except Exception as exc:
            failed.append(sym)
            print(f"[cyq_fetcher] {sym} on {trade_date}: {exc}")

    if failed and not pieces:
        raise RuntimeError(
            f"All {len(failed)} symbol(s) failed for {trade_date} "
            f"(first: {failed[:3]}) — possible systemic API error"
        )

    if not pieces:
        return pd.DataFrame(columns=["date", "symbol", "price", "percent"])
    return pd.concat(pieces, ignore_index=True)
