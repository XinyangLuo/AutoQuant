"""xtquant (国金 QMT / 迅投) realtime quote adapter.

Read-only intraday data only. Trading APIs (xttrader) are out of scope here —
those belong in a future ``trading/broker/`` module with explicit account
authorization (see CLAUDE.md §6.5).

xtquant ships only as a Windows DLL bundle. On macOS / Linux you typically run
it under Wine / CrossOver with Windows Python 3.9. This module assumes the
``xtquant`` package is importable in the current interpreter; deployment is
left to the user (see ``README.md`` in this directory).

Output schemas are intentionally aligned with ``market_daily`` column names so
downstream signal code can treat realtime and EOD frames interchangeably.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

import pandas as pd


def _require_xtdata():
    """Lazy import so the rest of the package works without xtquant installed."""
    try:
        from xtquant import xtdata  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "xtquant is not importable in this interpreter. "
            "On macOS/Linux run under Wine + Windows Python 3.9; "
            "see backtest/data/realtime/README.md for setup."
        ) from exc
    return xtdata


# ---------------------------------------------------------------------------
# Snapshot (cross-section)
# ---------------------------------------------------------------------------

def fetch_full_tick(symbols: Sequence[str]) -> pd.DataFrame:
    """Realtime L1 snapshot for a list of symbols.

    Returns a DataFrame with columns aligned to ``market_daily``:
    ``date, symbol, open, high, low, close, pre_close, volume, amount,
    change, pct_chg`` plus a ``ts`` column with the snapshot timestamp.

    ``close`` here is the *last traded price*, not the official close. Use only
    for intraday signal generation; never UPSERT into ``market_daily``.
    """
    xtdata = _require_xtdata()
    raw = xtdata.get_full_tick(list(symbols)) or {}

    today = datetime.now().date()
    rows = []
    for symbol in symbols:
        d = raw.get(symbol)
        if not d:
            continue
        last = float(d.get("lastPrice", 0.0))
        pre = float(d.get("lastClose", 0.0))
        change = last - pre if pre else 0.0
        pct_chg = (change / pre * 100.0) if pre else 0.0
        rows.append({
            "date": today,
            "symbol": symbol,
            "open": float(d.get("open", 0.0)),
            "high": float(d.get("high", 0.0)),
            "low": float(d.get("low", 0.0)),
            "close": last,
            "pre_close": pre,
            "volume": int(d.get("volume", 0)),
            "amount": float(d.get("amount", 0.0)),
            "change": round(change, 4),
            "pct_chg": round(pct_chg, 4),
            "ts": datetime.now(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bars (time series)
# ---------------------------------------------------------------------------

_ALLOWED_PERIODS = {"1m", "5m", "15m", "30m", "60m", "1d"}


def fetch_bars(
    symbols: Sequence[str],
    period: str = "1m",
    count: int = 240,
    download: bool = True,
) -> dict[str, pd.DataFrame]:
    """Intraday / daily bars per symbol.

    Parameters
    ----------
    symbols
        Sequence of Tushare-style codes (``600519.SH`` / ``000001.SZ``).
    period
        One of ``1m / 5m / 15m / 30m / 60m / 1d``.
    count
        Number of most recent bars to return.
    download
        If True, call ``download_history_data`` first to ensure local cache is
        populated. Set False if you already pre-warmed via a scheduled task.

    Returns
    -------
    Dict keyed by symbol, each value a DataFrame with columns
    ``open / high / low / close / volume`` indexed by timestamp.
    """
    if period not in _ALLOWED_PERIODS:
        raise ValueError(f"period must be one of {_ALLOWED_PERIODS}, got {period!r}")

    xtdata = _require_xtdata()
    syms = list(symbols)

    if download:
        for s in syms:
            xtdata.download_history_data(s, period=period, start_time="", end_time="")

    raw = xtdata.get_market_data_ex(
        ["open", "high", "low", "close", "volume"],
        syms,
        period=period,
        count=count,
    ) or {}

    out: dict[str, pd.DataFrame] = {}
    for s in syms:
        df = raw.get(s)
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        out[s] = df
    return out


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def fetch_instrument_details(symbols: Iterable[str]) -> pd.DataFrame:
    """Per-symbol static info: name, limit-up / limit-down, prev close.

    Useful for the simulation engine's limit-price filter and the signal
    layer's risk preview text.
    """
    xtdata = _require_xtdata()
    rows = []
    for s in symbols:
        d = xtdata.get_instrument_detail(s)
        if not d:
            continue
        rows.append({
            "symbol": s,
            "name": d.get("InstrumentName"),
            "limit_up": float(d.get("UpStopPrice", 0.0)),
            "limit_down": float(d.get("DownStopPrice", 0.0)),
            "pre_close": float(d.get("PreClose", 0.0)),
        })
    return pd.DataFrame(rows)
