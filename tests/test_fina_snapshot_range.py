"""Equivalence test: get_fina_snapshot_range == per-date get_fina_snapshot loop.

The new range function rewrites the per-trade-date PIT snapshot loop as a
single range-join SQL. Semantic must stay bit-identical to the old impl —
this test asserts that on a small slice of the real market.duckdb.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def storage():
    ms = MarketStorage()
    try:
        yield ms
    finally:
        ms.close()


@pytest.fixture(scope="module")
def test_window(storage):
    """Pick a recent ~30-day window where all three fina tables have data."""
    start, end = "20240101", "20240131"
    dates = get_trade_dates(start, end)
    assert dates, "no trade dates in test window"
    return start, end, dates


def _iterative_panel(
    storage: MarketStorage,
    dates: list[str],
    symbols: list[str] | None,
    columns: list[str] | None,
) -> pd.DataFrame:
    """Build the long panel using the old per-date snapshot loop."""
    pieces = []
    for d in dates:
        snap = storage.get_fina_snapshot(
            as_of_date=d, symbols=symbols, columns=columns,
        )
        if snap.empty:
            continue
        snap = snap.copy()
        snap["date"] = pd.Timestamp(d).date()
        pieces.append(snap)
    if not pieces:
        return pd.DataFrame(columns=["date", "symbol", "end_date"])
    return pd.concat(pieces, ignore_index=True)


def _normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Sort + reset to compare two frames invariant to row order."""
    if df.empty:
        return df
    keep = [c for c in cols if c in df.columns]
    return df[keep].sort_values(keep[:3]).reset_index(drop=True)


def test_range_matches_iterative_small_universe(storage, test_window):
    """Same (date, symbol, end_date) rows + identical numeric values."""
    start, end, dates = test_window
    symbols = ["000001.SZ", "600519.SH", "300750.SZ"]  # 银行 / 白酒 / 电池
    columns = ["inc_n_income", "bs_total_assets", "cf_n_cashflow_act"]

    iterative = _iterative_panel(storage, dates, symbols, columns)
    ranged = storage.get_fina_snapshot_range(
        start, end, symbols=symbols, columns=columns,
    )

    assert not iterative.empty, "iterative panel empty — test window has no fina data"
    assert not ranged.empty, "range panel empty"

    cols = ["date", "symbol", "end_date"] + columns
    iterative["date"] = pd.to_datetime(iterative["date"]).dt.date
    ranged["date"] = pd.to_datetime(ranged["date"]).dt.date

    iterative_n = _normalize(iterative, cols)
    ranged_n = _normalize(ranged, cols)

    pd.testing.assert_frame_equal(
        iterative_n, ranged_n, check_dtype=False,
    )


def test_last_n_quarters_trims_history(storage, test_window):
    """``last_n_quarters=4`` keeps only the 4 most recent end_dates per (date, symbol)."""
    start, end, _ = test_window
    symbols = ["000001.SZ"]
    columns = ["inc_n_income"]

    full = storage.get_fina_snapshot_range(start, end, symbols=symbols, columns=columns)
    trimmed = storage.get_fina_snapshot_range(
        start, end, symbols=symbols, columns=columns, last_n_quarters=4,
    )

    assert not full.empty and not trimmed.empty
    # For every (date, symbol), trimmed should have <= 4 rows.
    counts = trimmed.groupby(["date", "symbol"]).size()
    assert (counts <= 4).all(), counts.max()
    # And the kept end_dates should be the top-4 by descending order.
    for (d, sym), grp in trimmed.groupby(["date", "symbol"]):
        full_grp = full[(full["date"] == d) & (full["symbol"] == sym)]
        if len(full_grp) <= 4:
            assert len(grp) == len(full_grp)
            continue
        expected = sorted(full_grp["end_date"].tolist(), reverse=True)[:4]
        assert sorted(grp["end_date"].tolist(), reverse=True) == expected


def test_empty_window_returns_empty(storage):
    """A pre-listing-era window returns an empty frame, not an error."""
    df = storage.get_fina_snapshot_range(
        "19900101", "19900201", columns=["inc_n_income"],
    )
    # Empty is OK; non-empty would mean we leaked rows from future dates.
    if not df.empty:
        assert pd.to_datetime(df["date"]).max().strftime("%Y%m%d") <= "19900201"
