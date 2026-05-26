"""Per-symbol minute bar fetcher with automatic range chunking."""

import pandas as pd
import tushare as ts

from backtest.data.tushare_client import api_call


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------

# Conservative chunk sizes for Tushare's ~8000-row limit per request.
# 1min ≈ 240 bars/day → 30 days/chunk (7200 rows)
_CHUNK_DAYS = {
    "1min": 30,
    "5min": 150,
    "15min": 500,
    "30min": 1000,
    "60min": 2000,
}


def _split_date_range(start_date: str, end_date: str, chunk_days: int):
    """Split [start, end] into inclusive sub-ranges of at most chunk_days."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    cur = start
    while cur <= end:
        sub_end = min(cur + pd.Timedelta(days=chunk_days - 1), end)
        chunks.append((cur.strftime("%Y%m%d"), sub_end.strftime("%Y%m%d")))
        cur = sub_end + pd.Timedelta(days=1)
    return chunks


def _fetch_minute_chunk(symbol: str, start_date: str, end_date: str, freq: str) -> pd.DataFrame:
    """Call ts.pro_bar for a single symbol and date range."""
    df = api_call(
        ts.pro_bar,
        ts_code=symbol,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        asset="E",
    )
    return df if df is not None and not df.empty else pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_minute_bars(symbol: str, start_date: str, end_date: str, freq: str = "1min") -> pd.DataFrame:
    """Fetch minute bars for a single symbol via ``ts.pro_bar``.

    Automatically chunks the request to respect Tushare's ~8000-row
    per-request limit.  Returns an empty DataFrame if no data.
    """
    chunk_days = _CHUNK_DAYS.get(freq, 30)
    chunks = _split_date_range(start_date, end_date, chunk_days)

    pieces = []
    for s, e in chunks:
        df = _fetch_minute_chunk(symbol, s, e, freq)
        if not df.empty:
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, ignore_index=True)
    return _transform_minute(result)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def _transform_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Tushare minute output to internal schema."""
    if df.empty:
        return df

    # Defensive: Tushare must return these columns
    required = {"ts_code", "trade_time", "vol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tushare response missing required columns: {missing}")

    df = df.rename(columns={
        "ts_code": "symbol",
        "trade_time": "time",
        "vol": "volume",
    })

    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df["date"] = df["time"].dt.date

    # vol is in 手 (hands) → 股 (shares); use nullable Int64 so NaN survives
    df["volume"] = (df["volume"] * 100).round().astype("Int64")

    cols = [
        "date", "time", "symbol", "open", "high", "low", "close",
        "volume", "amount", "pre_close", "change", "pct_chg",
    ]
    return df[[c for c in cols if c in df.columns]].copy()
