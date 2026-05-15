"""Tests for factor compute engine."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.factor.compute import compute_factor
from backtest.factor.registry import register


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry state before each test."""
    from backtest.factor import registry
    registry._REGISTRY_CACHE = {}
    registry._FACTOR_FUNCTIONS.clear()
    yield
    registry._REGISTRY_CACHE = {}
    registry._FACTOR_FUNCTIONS.clear()


class TestComputeFactor:
    def test_market_only_factor(self, tmp_path, monkeypatch):
        """Test computing a simple market-only factor with mock data."""

        @register("f_mock", name="mock", category="test", data_sources=["market_daily"])
        def mock_factor(panel, window=2):
            df = panel[["date", "symbol", "close"]].copy()
            df = df.sort_values(["symbol", "date"])
            df["ret"] = df.groupby("symbol")["close"].shift(window)
            df["ret"] = df["close"] / df["ret"] - 1
            return df.set_index(["date", "symbol"])["ret"]

        # Mock MarketStorage
        mock_storage = MockMarketStorage()
        result = compute_factor("f_mock", "20240103", "20240105", market_storage=mock_storage)

        assert len(result) == 6  # 3 dates × 2 symbols
        assert list(result.columns) == ["date", "symbol", "factor_id", "value"]
        assert result["factor_id"].unique() == ["f_mock"]

        # Check values: for symbol A, 2024-01-03, close=103, window=2 -> shift to 2024-01-01 close=101
        a_row = result[(result["symbol"] == "A") & (result["date"] == "2024-01-03")]
        assert len(a_row) == 1
        assert a_row["value"].iloc[0] == pytest.approx(103.0 / 101.0 - 1, abs=1e-6)

    def test_parameter_binding(self, tmp_path, monkeypatch):
        """Test that parameters are correctly bound to the compute function."""

        @register(
            "f_param",
            name="param_test",
            category="test",
            data_sources=["market_daily"],
            parameters={"multiplier": 2.0},
        )
        def param_factor(panel, multiplier=1.0):
            df = panel[["date", "symbol", "close"]].copy()
            df["value"] = df["close"] * multiplier
            return df.set_index(["date", "symbol"])["value"]

        mock_storage = MockMarketStorage()
        result = compute_factor("f_param", "20240103", "20240103", market_storage=mock_storage)

        # multiplier=2.0 should be bound from registry parameters
        a_row = result[(result["symbol"] == "A") & (result["date"] == "2024-01-03")]
        assert a_row["value"].iloc[0] == pytest.approx(103.0 * 2.0, abs=1e-6)

    def test_date_filtering(self, tmp_path, monkeypatch):
        """Test that results are filtered to the requested date range."""

        @register("f_range", name="range_test", category="test", data_sources=["market_daily"])
        def range_factor(panel):
            df = panel[["date", "symbol", "close"]].copy()
            df["value"] = df["close"]
            return df.set_index(["date", "symbol"])["value"]

        mock_storage = MockMarketStorage()
        result = compute_factor("f_range", "20240103", "20240104", market_storage=mock_storage)

        dates = result["date"].unique()
        assert len(dates) == 2
        assert pd.Timestamp("2024-01-03") in dates
        assert pd.Timestamp("2024-01-04") in dates
        assert pd.Timestamp("2024-01-05") not in dates

    def test_empty_result(self, tmp_path, monkeypatch):
        """Test that empty market data returns empty DataFrame."""

        @register("f_empty", name="empty", category="test", data_sources=["market_daily"])
        def empty_factor(panel):
            return panel.set_index(["date", "symbol"])["close"]

        mock_storage = MockMarketStorage(empty=True)
        result = compute_factor("f_empty", "20240101", "20240105", market_storage=mock_storage)

        assert result.empty
        assert list(result.columns) == ["date", "symbol", "factor_id", "value"]


class MockMarketStorage:
    """Mock MarketStorage for testing compute_factor without a real database."""

    def __init__(self, empty=False):
        self.empty = empty
        self._data = self._create_mock_data()

    def _create_mock_data(self):
        if self.empty:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

        dates = pd.date_range("2024-01-01", periods=5)
        return pd.DataFrame({
            "date": list(dates) * 2,
            "symbol": ["A"] * 5 + ["B"] * 5,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0] * 2,
            "high": [101.0, 102.0, 103.0, 104.0, 105.0] * 2,
            "low": [99.0, 100.0, 101.0, 102.0, 103.0] * 2,
            "close": [101.0, 102.0, 103.0, 104.0, 105.0] + [201.0, 202.0, 203.0, 204.0, 205.0],
            "volume": [1000] * 10,
        })

    def get_bars(self, symbols=None, start=None, end=None):
        df = self._data.copy()
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df

    def get_fina_snapshot(self, as_of_date, symbols=None, columns=None):
        return pd.DataFrame()

    def close(self):
        pass
