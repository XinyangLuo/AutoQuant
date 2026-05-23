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

    def get_fina_snapshot_range(
        self, start, end, symbols=None, columns=None, last_n_quarters=None,
    ):
        return pd.DataFrame()

    def close(self):
        pass


class TestApplyVariantPipeline:
    """Verify barra_ind_size pipeline strips industry and Size_z exposure."""

    def test_barra_ind_size_residualizes(self, tmp_path, monkeypatch):
        from backtest.factor.compute import apply_variant_pipeline
        from backtest.factor.registry import register
        from backtest.factor.storage import FactorLibrary
        from backtest.factor.variants import BARRA_IND_SIZE_VARIANT

        @register(
            "f_alpha_test",
            name="alpha test", category="test",
            data_sources=["market_daily"],
            variant=BARRA_IND_SIZE_VARIANT, frequency="D",
        )
        def _alpha(panel):
            return panel.set_index(["date", "symbol"])["close"]

        import numpy as np
        rng = np.random.default_rng(11)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        symbols = [f"S{i:03d}" for i in range(60)]
        industries = {sym: f"I{i % 5}" for i, sym in enumerate(symbols)}
        ind_effect = {f"I{i}": rng.standard_normal() * 2 for i in range(5)}

        # Build raw alpha = industry effect + size_z * coef + noise.
        raw_rows = []
        size_rows = []
        ind_rows = []
        size_z_per_day = {}
        for d in dates:
            sz = {sym: rng.standard_normal() for sym in symbols}
            sz_arr = np.array(list(sz.values()))
            sz_z = (sz_arr - sz_arr.mean()) / sz_arr.std()
            for sym, z in zip(symbols, sz_z):
                sz[sym] = z
            size_z_per_day[d] = sz
            for sym in symbols:
                y = ind_effect[industries[sym]] + 1.2 * sz[sym] + 0.4 * rng.standard_normal()
                raw_rows.append({"date": d, "symbol": sym, "value": y, "factor_id": "f_alpha_test"})
                size_rows.append({"date": d, "symbol": sym, "factor_id": "f_barra_size", "value": sz[sym]})
                ind_rows.append({"date": d, "symbol": sym, "industry_code": industries[sym]})

        raw_df = pd.DataFrame(raw_rows)
        size_df = pd.DataFrame(size_rows)
        ind_df = pd.DataFrame(ind_rows)

        class _MS:
            def get_industry_panel_range(self, start, end, level):
                return ind_df.copy()
            def close(self):
                pass

        # Seed Size_z into a tmp library DB and point FactorLibrary at it.
        lib_path = tmp_path / "factor_library.duckdb"
        monkeypatch.setattr(
            "backtest.factor.storage.FACTOR_LIBRARY_DB_PATH", lib_path,
        )
        with FactorLibrary() as lib:
            lib.insert_factors(size_df, allow_unadmitted=True)

        out = apply_variant_pipeline(
            raw_df, "f_alpha_test",
            market_storage=_MS(),
        )

        # Residualized values should be ~orthogonal to size_z and industry dummies.
        merged = out.merge(
            size_df.rename(columns={"value": "size_z"})[["date", "symbol", "size_z"]],
            on=["date", "symbol"],
        ).merge(ind_df, on=["date", "symbol"])
        corrs = []
        for _, sub in merged.groupby("date"):
            if len(sub) < 10:
                continue
            corrs.append(abs(sub["value"].corr(sub["size_z"])))
            for ind in sub["industry_code"].unique():
                dummy = (sub["industry_code"] == ind).astype(float)
                if dummy.std() > 0:
                    corrs.append(abs(sub["value"].corr(dummy)))
        max_corr = float(np.nanmax(corrs))
        assert max_corr < 1e-6, f"residual max |corr| with design = {max_corr}"

    def test_barra_l3_pipeline_zscores(self, tmp_path):
        from backtest.factor.compute import apply_variant_pipeline
        from backtest.factor.registry import register
        from backtest.factor.variants import BARRA_L3_VARIANT

        @register(
            "f_l3_test",
            name="l3 test", category="test",
            data_sources=["market_daily"],
            variant=BARRA_L3_VARIANT, frequency="D",
        )
        def _f(panel):
            return panel.set_index(["date", "symbol"])["close"]

        import numpy as np
        rng = np.random.default_rng(3)
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        symbols = [f"S{i}" for i in range(30)]
        raw = []
        ind_rows = []
        for d in dates:
            for i, s in enumerate(symbols):
                raw.append({"date": d, "symbol": s, "factor_id": "f_l3_test",
                            "value": rng.standard_normal() * 5})
                ind_rows.append({"date": d, "symbol": s, "industry_code": f"I{i % 3}"})

        class _MS:
            def get_industry_panel_range(self, start, end, level):
                return pd.DataFrame(ind_rows)
            def close(self):
                pass

        out = apply_variant_pipeline(
            pd.DataFrame(raw), "f_l3_test", market_storage=_MS(),
        )
        # After barra_l3 pipeline output should be z-scored cross-sectionally.
        for _, sub in out.groupby("date"):
            assert abs(sub["value"].mean()) < 1e-9
            assert abs(sub["value"].std(ddof=0) - 1.0) < 0.05
