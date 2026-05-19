"""Unit tests for xtquant_quote adapter.

xtquant is Windows-only and rarely importable in CI. Tests use a fake xtdata
module injected via monkeypatch, so coverage runs on any platform.
"""

from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime

import pandas as pd
import pytest

from backtest.data.realtime import xtquant_quote


@pytest.fixture
def fake_xtdata(monkeypatch):
    """Inject a stub xtquant.xtdata into sys.modules and return it."""
    fake = types.SimpleNamespace()

    fake.full_tick_return = {
        "000001.SZ": {
            "lastPrice": 12.50, "open": 12.30, "high": 12.60,
            "low": 12.20, "lastClose": 12.40, "volume": 12345678,
            "amount": 1.5e8,
        },
        "600519.SH": {
            "lastPrice": 1680.00, "open": 1670.00, "high": 1685.00,
            "low": 1665.00, "lastClose": 1672.00, "volume": 234567,
            "amount": 3.9e8,
        },
    }
    fake.get_full_tick = lambda syms: {s: fake.full_tick_return.get(s) for s in syms}

    bars_idx = pd.date_range("2026-05-19 09:30", periods=3, freq="1min")
    fake.bars_return = {
        "600519.SH": pd.DataFrame(
            {"open": [1670, 1671, 1672], "high": [1672, 1673, 1674],
             "low": [1669, 1670, 1671], "close": [1671, 1672, 1673],
             "volume": [1000, 1100, 1200]},
            index=bars_idx,
        ),
    }
    fake.get_market_data_ex = lambda fields, syms, period, count: {
        s: fake.bars_return.get(s) for s in syms
    }
    fake.download_history_data = lambda *a, **kw: None

    fake.instrument_return = {
        "600519.SH": {
            "InstrumentName": "贵州茅台", "UpStopPrice": 1839.20,
            "DownStopPrice": 1504.80, "PreClose": 1672.00,
        },
    }
    fake.get_instrument_detail = lambda s: fake.instrument_return.get(s)

    pkg = types.ModuleType("xtquant")
    pkg.xtdata = fake
    monkeypatch.setitem(sys.modules, "xtquant", pkg)
    monkeypatch.setitem(sys.modules, "xtquant.xtdata", fake)
    importlib.reload(xtquant_quote)
    return fake


def test_fetch_full_tick_shape_and_columns(fake_xtdata):
    df = xtquant_quote.fetch_full_tick(["000001.SZ", "600519.SH"])
    expected_cols = {
        "date", "symbol", "open", "high", "low", "close", "pre_close",
        "volume", "amount", "change", "pct_chg", "ts",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 2
    assert df["date"].iloc[0] == datetime.now().date()


def test_fetch_full_tick_change_math(fake_xtdata):
    df = xtquant_quote.fetch_full_tick(["000001.SZ"])
    row = df.iloc[0]
    # 12.50 - 12.40 = 0.10; 0.10 / 12.40 * 100 ≈ 0.8065
    assert row["change"] == pytest.approx(0.10, abs=1e-4)
    assert row["pct_chg"] == pytest.approx(0.8065, abs=1e-3)


def test_fetch_full_tick_skips_missing(fake_xtdata):
    df = xtquant_quote.fetch_full_tick(["000001.SZ", "999999.XX"])
    assert list(df["symbol"]) == ["000001.SZ"]


def test_fetch_bars_returns_per_symbol_frames(fake_xtdata):
    out = xtquant_quote.fetch_bars(["600519.SH"], period="1m", count=3)
    assert "600519.SH" in out
    df = out["600519.SH"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3


def test_fetch_bars_rejects_unknown_period(fake_xtdata):
    with pytest.raises(ValueError, match="period must be one of"):
        xtquant_quote.fetch_bars(["600519.SH"], period="2h")


def test_fetch_instrument_details(fake_xtdata):
    df = xtquant_quote.fetch_instrument_details(["600519.SH"])
    assert df.iloc[0]["name"] == "贵州茅台"
    assert df.iloc[0]["limit_up"] == pytest.approx(1839.20)


def test_missing_xtquant_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "xtquant", None)
    importlib.reload(xtquant_quote)
    with pytest.raises(ImportError, match="Wine"):
        xtquant_quote.fetch_full_tick(["000001.SZ"])
