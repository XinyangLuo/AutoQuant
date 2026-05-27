"""Tests for strategy module."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


class TestConfig:
    """Test strategy configuration loading and validation."""

    def test_from_dict_basic(self):
        """Test loading config from a nested dict."""
        from backtest.strategy.config import StrategyConfig

        d = {
            "name": "momentum_test",
            "strategy": {"type": "single_factor_topk", "rebalance_freq": "1M", "delay": 1},
            "universe": {"exclude_st": True, "include_kcb": False},
            "factors": [{"id": "f_001", "direction": "desc", "weight": 1.0}],
            "selection": {"method": "topk", "top_k": 20},
            "weighting": {"method": "equal"},
        }
        config = StrategyConfig.from_dict(d)
        config.validate()

        assert config.name == "momentum_test"
        assert config.strategy_type == "single_factor_topk"
        assert config.rebalance_freq == "1M"
        assert config.delay == 1
        assert config.universe.exclude_st is True
        assert config.universe.include_kcb is False
        assert len(config.factors) == 1
        assert config.factors[0].id == "f_001"
        assert config.selection.top_k == 20

    def test_from_json(self, tmp_path):
        """Test loading config from a JSON file."""
        from backtest.strategy.config import StrategyConfig

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "name": "json_test",
                    "factors": [{"id": "f_002", "direction": "asc"}],
                    "selection": {"method": "long_short", "top_k": 10, "bottom_k": 10},
                }
            )
        )
        config = StrategyConfig.from_json(config_path)
        config.validate()

        assert config.name == "json_test"
        assert config.selection.method == "long_short"
        assert config.selection.bottom_k == 10

    def test_validate_missing_factors(self):
        """Test that validation fails when no factors are configured."""
        from backtest.strategy.config import StrategyConfig

        config = StrategyConfig()
        config.factors = []
        with pytest.raises(ValueError, match="At least one factor"):
            config.validate()

    def test_validate_invalid_direction(self):
        """Test that invalid factor direction is rejected."""
        from backtest.strategy.config import StrategyConfig, FactorConfig, SelectionConfig

        config = StrategyConfig()
        config.factors = [FactorConfig(id="f_001", direction="invalid")]
        config.selection = SelectionConfig(method="topk", top_k=20)
        with pytest.raises(ValueError, match="'asc' or 'desc'"):
            config.validate()

    def test_validate_invalid_selection_method(self):
        """Test that invalid selection method is rejected."""
        from backtest.strategy.config import StrategyConfig, FactorConfig, SelectionConfig

        config = StrategyConfig()
        config.factors = [FactorConfig(id="f_001")]
        config.selection = SelectionConfig(method="invalid")
        with pytest.raises(ValueError, match="topk.*long_short.*decile"):
            config.validate()

    def test_validate_invalid_rebalance_freq(self):
        """Test that invalid rebalance frequency is rejected."""
        from backtest.strategy.config import StrategyConfig, FactorConfig, SelectionConfig

        config = StrategyConfig()
        config.factors = [FactorConfig(id="f_001")]
        config.selection = SelectionConfig(method="topk", top_k=20)
        config.rebalance_freq = "3W"
        with pytest.raises(ValueError, match="1D.*1W.*2W.*1M.*EOM"):
            config.validate()

    def test_validate_top_k_top_pct_xor(self):
        """topk 模式 top_k / top_pct 必须恰好一个非 None。"""
        from backtest.strategy.config import StrategyConfig, FactorConfig, SelectionConfig

        config = StrategyConfig()
        config.factors = [FactorConfig(id="f_001")]
        config.selection = SelectionConfig(method="topk")  # 两者都 None
        with pytest.raises(ValueError, match="top_k 或 top_pct"):
            config.validate()

        config.selection = SelectionConfig(method="topk", top_k=10, top_pct=0.1)
        with pytest.raises(ValueError, match="top_k 或 top_pct"):
            config.validate()

    def test_validate_top_pct_range(self):
        """top_pct 必须在 (0, 1] 区间。"""
        from backtest.strategy.config import StrategyConfig, FactorConfig, SelectionConfig

        config = StrategyConfig()
        config.factors = [FactorConfig(id="f_001")]
        config.selection = SelectionConfig(method="topk", top_pct=1.5)
        with pytest.raises(ValueError, match="top_pct"):
            config.validate()

        config.selection = SelectionConfig(method="topk", top_pct=0.0)
        with pytest.raises(ValueError, match="top_pct"):
            config.validate()

    def test_validate_top_pct_happy(self):
        """top_pct=0.1 单独使用应通过。"""
        from backtest.strategy.config import StrategyConfig, FactorConfig, SelectionConfig

        config = StrategyConfig()
        config.factors = [FactorConfig(id="f_001")]
        config.selection = SelectionConfig(method="topk", top_pct=0.1)
        config.validate()  # 不抛错


class TestUniverseFilter:
    """Test universe filtering logic."""

    def test_exclude_st(self):
        """Test filtering out ST stocks."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(exclude_st=True)
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ", "600000.SH"],
            "is_st": [0, 1, 0],
        })
        result = uf.filter("20240101", panel)
        assert list(result["symbol"]) == ["000001.SZ", "600000.SH"]

    def test_include_st(self):
        """Test including ST stocks when filter is off."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(exclude_st=False)
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ"],
            "is_st": [0, 1],
        })
        result = uf.filter("20240101", panel)
        assert list(result["symbol"]) == ["000001.SZ", "000002.SZ"]

    def test_missing_is_st_column(self):
        """Test graceful handling when is_st column is missing."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(exclude_st=True)
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ"],
        })
        result = uf.filter("20240101", panel)
        assert len(result) == 2

    def test_new_ipo_filter(self):
        """Test filtering out newly listed stocks."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(exclude_new_ipo_days=252)
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["OLD001.SZ", "NEW001.SZ"],
            "list_date": ["19900101", "20231201"],
        })
        result = uf.filter("20240101", panel)
        assert list(result["symbol"]) == ["OLD001.SZ"]

    def test_board_filter_cyb(self):
        """Test filtering out ChiNext (CYB) stocks."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(include_cyb=False, include_kcb=True)
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["000001.SZ", "300001.SZ", "688001.SH"],
        })
        result = uf.filter("20240101", panel)
        assert list(result["symbol"]) == ["000001.SZ", "688001.SH"]

    def test_board_filter_kcb(self):
        """Test filtering out STAR Market (KCB) stocks."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(include_cyb=True, include_kcb=False)
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["000001.SZ", "300001.SZ", "688001.SH"],
        })
        result = uf.filter("20240101", panel)
        assert list(result["symbol"]) == ["000001.SZ", "300001.SZ"]

    def test_market_cap_filter(self):
        """Test filtering by minimum market cap."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(min_market_cap=1e9)
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["SMALL.SZ", "BIG.SH"],
            "circ_mv": [5e4, 2e9],  # 5e4 万元 = 5亿元 < 10亿门槛
        })
        result = uf.filter("20240101", panel)
        assert list(result["symbol"]) == ["BIG.SH"]

    def test_combined_filters(self):
        """Test that multiple filters are applied in sequence."""
        from backtest.strategy.universe import UniverseFilter
        from backtest.strategy.config import UniverseConfig

        config = UniverseConfig(
            exclude_st=True,
            include_kcb=False,
            min_market_cap=1e9,
        )
        uf = UniverseFilter(config)

        panel = pd.DataFrame({
            "symbol": ["A.SZ", "B.SZ", "C.SH", "D.SH", "E.SH"],
            "is_st": [0, 1, 0, 0, 0],
            "list_date": ["20000101", "20000101", "20000101", "20000101", "20000101"],
            "circ_mv": [2e9, 2e9, 2e9, 5e4, 2e9],  # D.SH = 5亿元 < 10亿门槛
        })
        result = uf.filter("20240101", panel)
        # A: pass (not ST, not KCB, cap OK)
        # B: excluded (ST)
        # C: pass (not KCB - C.SH doesn't start with 68)
        # D: excluded (cap too small)
        # E: pass
        assert set(result["symbol"]) == {"A.SZ", "C.SH", "E.SH"}


class TestWeightAllocator:
    """Test portfolio weight allocation methods."""

    def test_equal_weight(self):
        """Test equal weight allocation."""
        from backtest.strategy.weight import WeightAllocator
        from backtest.strategy.config import WeightingConfig

        wa = WeightAllocator(WeightingConfig(method="equal"))
        df = pd.DataFrame({"symbol": ["A", "B", "C"]})
        weights = wa.allocate(df)

        assert len(weights) == 3
        assert weights.sum() == pytest.approx(1.0)
        assert weights["A"] == pytest.approx(1 / 3)
        assert weights["B"] == pytest.approx(1 / 3)

    def test_equal_weight_empty(self):
        """Test equal weight with empty input."""
        from backtest.strategy.weight import WeightAllocator
        from backtest.strategy.config import WeightingConfig

        wa = WeightAllocator(WeightingConfig(method="equal"))
        df = pd.DataFrame({"symbol": []})
        weights = wa.allocate(df)
        assert weights.empty

    def test_market_cap_weight(self):
        """Test market-cap weighted allocation."""
        from backtest.strategy.weight import WeightAllocator
        from backtest.strategy.config import WeightingConfig

        wa = WeightAllocator(WeightingConfig(method="market_cap"))
        df = pd.DataFrame({
            "symbol": ["A", "B"],
            "circ_mv": [1e9, 3e9],
        })
        weights = wa.allocate(df)

        assert weights.sum() == pytest.approx(1.0)
        assert weights["A"] == pytest.approx(0.25)
        assert weights["B"] == pytest.approx(0.75)

    def test_market_cap_missing_column(self):
        """Test market-cap weight fails without circ_mv column."""
        from backtest.strategy.weight import WeightAllocator
        from backtest.strategy.config import WeightingConfig

        wa = WeightAllocator(WeightingConfig(method="market_cap"))
        df = pd.DataFrame({"symbol": ["A", "B"]})
        with pytest.raises(ValueError, match="circ_mv"):
            wa.allocate(df)

    def test_factor_value_weight(self):
        """Test factor-value weighted allocation."""
        from backtest.strategy.weight import WeightAllocator
        from backtest.strategy.config import WeightingConfig

        wa = WeightAllocator(WeightingConfig(method="factor_value"))
        df = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "f_001": [1.0, 2.0, 3.0],
        })
        weights = wa.allocate(df, factor_col="f_001")

        assert weights.sum() == pytest.approx(1.0)
        assert weights["A"] == pytest.approx(1 / 6)
        assert weights["B"] == pytest.approx(2 / 6)
        assert weights["C"] == pytest.approx(3 / 6)

    def test_factor_value_with_negative_values(self):
        """Test factor-value weight uses absolute values."""
        from backtest.strategy.weight import WeightAllocator
        from backtest.strategy.config import WeightingConfig

        wa = WeightAllocator(WeightingConfig(method="factor_value"))
        df = pd.DataFrame({
            "symbol": ["A", "B"],
            "f_001": [-2.0, 1.0],
        })
        weights = wa.allocate(df, factor_col="f_001")

        assert weights.sum() == pytest.approx(1.0)
        assert weights["A"] == pytest.approx(2 / 3)
        assert weights["B"] == pytest.approx(1 / 3)


class TestDecay:
    """Test linear decay smoothing on factor panels."""

    def test_decay_basic(self):
        """Test decay smoothing with a simple series."""
        from backtest.strategy.base import StrategyBase

        # Single stock, 5 days, factor values [1, 2, 3, 4, 5]
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["A"] * 5,
            "f_001": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = StrategyBase._apply_decay(df, n=3)

        # Day 1 (min_periods=1): (1*1)/1 = 1
        # Day 2 (min_periods=2): (2*2 + 1*1)/(2+1) = 5/3
        # Day 3: (3*3 + 2*2 + 1*1)/(3+2+1) = 14/6 = 7/3
        # Day 4: (4*3 + 3*2 + 2*1)/6 = 20/6 = 10/3
        # Day 5: (5*3 + 4*2 + 3*1)/6 = 26/6 = 13/3
        vals = result.set_index("date")["f_001"].values
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(5.0 / 3.0)
        assert vals[2] == pytest.approx(14.0 / 6.0)
        assert vals[3] == pytest.approx(20.0 / 6.0)
        assert vals[4] == pytest.approx(26.0 / 6.0)

    def test_decay_multiple_stocks(self):
        """Test decay is applied per stock, not globally."""
        from backtest.strategy.base import StrategyBase

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3).tolist() * 2,
            "symbol": ["A"] * 3 + ["B"] * 3,
            "f_001": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        result = StrategyBase._apply_decay(df, n=2)
        result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Stock A: day1=1, day2=(2*2+1)/3=5/3, day3=(3*2+2)/3=8/3
        a_vals = result[result["symbol"] == "A"]["f_001"].values
        assert a_vals[0] == pytest.approx(1.0)
        assert a_vals[1] == pytest.approx(5.0 / 3.0)
        assert a_vals[2] == pytest.approx(8.0 / 3.0)

        # Stock B: day1=10, day2=(20*2+10)/3=50/3, day3=(30*2+20)/3=80/3
        b_vals = result[result["symbol"] == "B"]["f_001"].values
        assert b_vals[0] == pytest.approx(10.0)
        assert b_vals[1] == pytest.approx(50.0 / 3.0)
        assert b_vals[2] == pytest.approx(80.0 / 3.0)

    def test_decay_preserves_shape(self):
        """Test decay output has same rows and columns as input."""
        from backtest.strategy.base import StrategyBase

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["A"] * 5,
            "f_001": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f_002": [5.0, 4.0, 3.0, 2.0, 1.0],
        })

        result = StrategyBase._apply_decay(df, n=3)

        assert len(result) == len(df)
        assert set(result.columns) == {"date", "symbol", "f_001", "f_002"}


class TestSignals:
    """Test signal formatting and weight normalization."""

    def test_format_signals_basic(self):
        """Test basic signal formatting."""
        from backtest.strategy.signals import format_signals

        signals = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "symbol": ["A", "B", "A"],
            "target_weight": [0.5, 0.5, 1.0],
        })
        result = format_signals(signals)

        assert len(result) == 3
        assert list(result.columns) == ["date", "symbol", "target_weight"]
        assert np.issubdtype(result["date"].dtype, np.datetime64)

    def test_format_signals_missing_column(self):
        """Test formatting fails with missing columns."""
        from backtest.strategy.signals import format_signals

        signals = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["A"]})
        with pytest.raises(ValueError, match="missing columns"):
            format_signals(signals)

    def test_format_signals_drops_na(self):
        """Test that rows with NaN weights are dropped."""
        from backtest.strategy.signals import format_signals

        signals = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01"],
            "symbol": ["A", "B"],
            "target_weight": [0.5, None],
        })
        result = format_signals(signals)
        assert len(result) == 1
        assert list(result["symbol"]) == ["A"]

    def test_normalize_weights_long_only(self):
        """Test long-only weight normalization."""
        from backtest.strategy.signals import normalize_weights

        weights = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        result = normalize_weights(weights, long_sum=1.0)

        assert result.sum() == pytest.approx(1.0)
        assert result["A"] == pytest.approx(1 / 6)
        assert result["C"] == pytest.approx(3 / 6)

    def test_normalize_weights_long_short(self):
        """Test long-short weight normalization."""
        from backtest.strategy.signals import normalize_weights

        weights = pd.Series({"A": 1.0, "B": 2.0, "C": -1.0, "D": -3.0})
        result = normalize_weights(weights, long_sum=0.5, short_sum=0.5)

        assert result["A"] > 0
        assert result["B"] > 0
        assert result["C"] < 0
        assert result["D"] < 0
        assert result[result > 0].sum() == pytest.approx(0.5)
        assert result[result < 0].sum() == pytest.approx(-0.5)

    def test_normalize_weights_empty_positive(self):
        """Test normalization when no positive weights exist."""
        from backtest.strategy.signals import normalize_weights

        weights = pd.Series({"A": -1.0, "B": -2.0})
        result = normalize_weights(weights, long_sum=1.0, short_sum=0.5)

        # No positive weights, so positive side remains zero
        assert result[result > 0].sum() == pytest.approx(0.0)
        assert result[result < 0].sum() == pytest.approx(-0.5)


class TestSelection:
    """Test shared selection logic."""

    def test_build_signals_topk(self):
        """Test topk selection builds correct signals."""
        from backtest.strategy.selection import build_signals
        from backtest.strategy.config import SelectionConfig, WeightingConfig

        scores = pd.Series({"A": 1.0, "B": 3.0, "C": 2.0, "D": 0.5})
        df = pd.DataFrame({"symbol": ["A", "B", "C", "D"]})
        selection = SelectionConfig(method="topk", top_k=2)
        weighting = WeightingConfig(method="equal")

        rows = build_signals(
            pd.Timestamp("2024-01-01"), scores.sort_values(ascending=False),
            df, selection, weighting,
        )

        assert len(rows) == 2
        symbols = {r["symbol"] for r in rows}
        assert symbols == {"B", "C"}  # top 2 scores
        assert sum(r["target_weight"] for r in rows) == pytest.approx(1.0)

    def test_build_signals_long_short(self):
        """Test long-short selection builds correct signals."""
        from backtest.strategy.selection import build_signals
        from backtest.strategy.config import SelectionConfig, WeightingConfig

        scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
        df = pd.DataFrame({"symbol": ["A", "B", "C", "D"]})
        selection = SelectionConfig(method="long_short", top_k=2, bottom_k=2)
        weighting = WeightingConfig(method="equal")

        rows = build_signals(
            pd.Timestamp("2024-01-01"), scores.sort_values(ascending=False),
            df, selection, weighting,
        )

        symbols = {r["symbol"] for r in rows}
        assert symbols == {"D", "C", "A", "B"}

        long_weights = [r["target_weight"] for r in rows if r["target_weight"] > 0]
        short_weights = [r["target_weight"] for r in rows if r["target_weight"] < 0]
        assert sum(long_weights) == pytest.approx(0.5)
        assert sum(short_weights) == pytest.approx(-0.5)

    def test_build_signals_decile(self):
        """Test decile selection returns all 10 groups."""
        from backtest.strategy.selection import build_signals
        from backtest.strategy.config import SelectionConfig, WeightingConfig

        scores = pd.Series({f"S{i:02d}": float(i) for i in range(20)})
        df = pd.DataFrame({"symbol": list(scores.index)})
        selection = SelectionConfig(method="decile")
        weighting = WeightingConfig(method="equal")

        rows = build_signals(
            pd.Timestamp("2024-01-01"), scores.sort_values(ascending=False),
            df, selection, weighting,
        )

        # 20 stocks, 10 deciles -> 2 per group
        assert len(rows) == 20
        groups = {r["decile_group"] for r in rows}
        assert groups == set(range(10))

    def test_build_signals_decile_specific_group(self):
        """Test selecting a specific decile group."""
        from backtest.strategy.selection import build_signals
        from backtest.strategy.config import SelectionConfig, WeightingConfig

        scores = pd.Series({f"S{i:02d}": float(i) for i in range(20)})
        df = pd.DataFrame({"symbol": list(scores.index)})
        selection = SelectionConfig(method="decile", decile_group=0)
        weighting = WeightingConfig(method="equal")

        rows = build_signals(
            pd.Timestamp("2024-01-01"), scores.sort_values(ascending=False),
            df, selection, weighting,
        )

        # Top decile (group 0) should have 2 stocks
        assert len(rows) == 2
        assert all(r["target_weight"] > 0 for r in rows)

    def test_build_signals_topk_with_pct(self):
        """topk + top_pct=0.1 在 100 只股票上应选 10 只。"""
        from backtest.strategy.selection import build_signals
        from backtest.strategy.config import SelectionConfig, WeightingConfig

        scores = pd.Series({f"S{i:03d}": float(100 - i) for i in range(100)})
        df = pd.DataFrame({"symbol": list(scores.index)})
        selection = SelectionConfig(method="topk", top_pct=0.1)
        weighting = WeightingConfig(method="equal")

        rows = build_signals(
            pd.Timestamp("2024-01-01"), scores.sort_values(ascending=False),
            df, selection, weighting,
        )

        assert len(rows) == 10
        assert sum(r["target_weight"] for r in rows) == pytest.approx(1.0)

    def test_build_signals_long_short_with_pct(self):
        """long_short + top_pct=0.05 / bottom_pct=0.05 在 100 只上各选 5 只。"""
        from backtest.strategy.selection import build_signals
        from backtest.strategy.config import SelectionConfig, WeightingConfig

        scores = pd.Series({f"S{i:03d}": float(100 - i) for i in range(100)})
        df = pd.DataFrame({"symbol": list(scores.index)})
        selection = SelectionConfig(
            method="long_short", top_pct=0.05, bottom_pct=0.05,
        )
        weighting = WeightingConfig(method="equal")

        rows = build_signals(
            pd.Timestamp("2024-01-01"), scores.sort_values(ascending=False),
            df, selection, weighting,
        )

        longs = [r for r in rows if r["target_weight"] > 0]
        shorts = [r for r in rows if r["target_weight"] < 0]
        assert len(longs) == 5
        assert len(shorts) == 5
        assert sum(r["target_weight"] for r in longs) == pytest.approx(0.5)
        assert sum(r["target_weight"] for r in shorts) == pytest.approx(-0.5)


class MockFactorStorage:
    """Mock FactorStorage for strategy testing."""

    def __init__(self, data: pd.DataFrame | None = None):
        if data is not None:
            self._data = data
        else:
            dates = pd.date_range("2024-01-01", periods=10)
            symbols = ["A", "B", "C", "D", "E"]
            rows = []
            for d in dates:
                for sym in symbols:
                    rows.append({
                        "date": d,
                        "symbol": sym,
                        "f_001": hash(sym) % 100 / 100,  # deterministic pseudo-random
                        "f_002": (hash(sym) // 10) % 100 / 100,
                    })
            self._data = pd.DataFrame(rows)

    def get_factor(
        self,
        factor_id: str,
        start: str | None = None,
        end: str | None = None,
        *,
        variant: str | None = None,
    ):
        # Mock has a single conceptual variant; the kwarg is accepted but ignored.
        if factor_id not in self._data.columns:
            return pd.DataFrame(columns=["date", "symbol", "value"])
        df = self._data[["date", "symbol", factor_id]].copy()
        df = df.rename(columns={factor_id: "value"})
        df["date"] = pd.to_datetime(df["date"])
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df

    def get_factor_panel(
        self,
        factor_ids: list[str],
        date: str,
        *,
        variant: str | None = None,
    ):
        d = pd.Timestamp(date)
        available = [f for f in factor_ids if f in self._data.columns]
        cols = ["date", "symbol"] + available
        df = self._data[self._data["date"] == d][cols].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def close(self):
        pass


class MockMarketStorage:
    """Mock MarketStorage for strategy testing."""

    def __init__(self):
        dates = pd.date_range("2024-01-01", periods=10)
        symbols = ["A", "B", "C", "D", "E"]
        rows = []
        for d in dates:
            for sym in symbols:
                rows.append({
                    "date": d,
                    "symbol": sym,
                    "close": 100.0 + hash(sym + str(d)) % 50,
                    "circ_mv": (hash(sym) % 10 + 1) * 1e9,
                    "amount": 1e6,
                    "is_st": 0,
                    "list_date": "20200101",
                })
        self._data = pd.DataFrame(rows)

    def get_bars(self, symbols=None, start=None, end=None, columns=None):
        df = self._data.copy()
        df["date"] = pd.to_datetime(df["date"])
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        if columns:
            keep = ["date", "symbol"] + [c for c in columns if c in df.columns]
            df = df[keep]
        return df

    def get_panel(self, date, columns=None):
        df = self._data[self._data["date"] == pd.Timestamp(date)].copy()
        df["date"] = pd.to_datetime(df["date"])
        if columns:
            keep = ["date", "symbol"] + [c for c in columns if c in df.columns]
            df = df[keep]
        return df

    def close(self):
        pass


class TestSingleFactorStrategy:
    """Test SingleFactorStrategy end-to-end with mock data."""

    def test_topk_signal_generation(self, monkeypatch):
        """Test topk strategy generates correct signals."""
        from backtest.strategy import SingleFactorStrategy, StrategyConfig, FactorConfig
        from backtest.strategy import SelectionConfig, WeightingConfig
        from backtest.strategy import base as base_module

        trade_dates = pd.date_range("2024-01-01", periods=10).strftime("%Y%m%d").tolist()
        monkeypatch.setattr(base_module, "get_trade_dates", lambda s, e: trade_dates)
        monkeypatch.setattr(base_module, "get_rebalance_dates", lambda s, e, f: trade_dates)

        config = StrategyConfig(
            strategy_type="single_factor_topk",
            rebalance_freq="1D",
            delay=0,
            factors=[FactorConfig(id="f_001", direction="desc")],
            selection=SelectionConfig(method="topk", top_k=2),
            weighting=WeightingConfig(method="equal"),
        )
        strategy = SingleFactorStrategy(config)

        factor_storage = MockFactorStorage()
        market_storage = MockMarketStorage()

        signals = strategy.run("20240101", "20240105", factor_storage=factor_storage, market_storage=market_storage)

        assert not signals.empty
        assert list(signals.columns) == ["date", "symbol", "target_weight"]
        # Each day should have exactly 2 stocks (top_k=2)
        for date, group in signals.groupby("date"):
            assert len(group) == 2
            assert group["target_weight"].sum() == pytest.approx(1.0)

    def test_long_short_signal_generation(self, monkeypatch):
        """Test long-short strategy generates both long and short positions."""
        from backtest.strategy import SingleFactorStrategy, StrategyConfig, FactorConfig
        from backtest.strategy import SelectionConfig, WeightingConfig
        from backtest.strategy import base as base_module

        trade_dates = pd.date_range("2024-01-01", periods=10).strftime("%Y%m%d").tolist()
        monkeypatch.setattr(base_module, "get_trade_dates", lambda s, e: trade_dates)
        monkeypatch.setattr(base_module, "get_rebalance_dates", lambda s, e, f: trade_dates)

        config = StrategyConfig(
            strategy_type="single_factor_topk",
            rebalance_freq="1D",
            delay=0,
            factors=[FactorConfig(id="f_001", direction="desc")],
            selection=SelectionConfig(method="long_short", top_k=2, bottom_k=2),
            weighting=WeightingConfig(method="equal"),
        )
        strategy = SingleFactorStrategy(config)

        factor_storage = MockFactorStorage()
        market_storage = MockMarketStorage()

        signals = strategy.run("20240101", "20240103", factor_storage=factor_storage, market_storage=market_storage)

        assert not signals.empty
        # Each day should have 4 stocks (2 long + 2 short)
        for date, group in signals.groupby("date"):
            assert len(group) == 4
            long_sum = group[group["target_weight"] > 0]["target_weight"].sum()
            short_sum = group[group["target_weight"] < 0]["target_weight"].sum()
            assert long_sum == pytest.approx(0.5)
            assert short_sum == pytest.approx(-0.5)

    def test_with_delay(self, monkeypatch):
        """Test that delay shifts signal dates forward."""
        from backtest.strategy import SingleFactorStrategy, StrategyConfig, FactorConfig
        from backtest.strategy import SelectionConfig, WeightingConfig
        from backtest.strategy import base as base_module

        # Mock get_trade_dates to avoid Tushare API call
        trade_dates = [f"202401{i:02d}" for i in range(1, 11)]
        monkeypatch.setattr(base_module, "get_trade_dates", lambda s, e: trade_dates)
        monkeypatch.setattr(base_module, "get_rebalance_dates", lambda s, e, f: trade_dates)

        config = StrategyConfig(
            strategy_type="single_factor_topk",
            rebalance_freq="1D",
            delay=1,
            factors=[FactorConfig(id="f_001", direction="desc")],
            selection=SelectionConfig(method="topk", top_k=2),
            weighting=WeightingConfig(method="equal"),
        )
        strategy = SingleFactorStrategy(config)

        factor_storage = MockFactorStorage()
        market_storage = MockMarketStorage()

        signals = strategy.run("20240101", "20240105", factor_storage=factor_storage, market_storage=market_storage)

        # With delay=1, signals should be shifted by 1 trading day
        # First signal date should be 20240102 (not 20240101)
        min_date = signals["date"].min()
        assert min_date.strftime("%Y%m%d") == "20240102"

    def test_direction_asc(self, monkeypatch):
        """Test ascending direction selects lowest factor values."""
        from backtest.strategy import SingleFactorStrategy, StrategyConfig, FactorConfig
        from backtest.strategy import SelectionConfig, WeightingConfig
        from backtest.strategy import base as base_module

        # Mock trade dates to avoid Tushare API
        monkeypatch.setattr(base_module, "get_trade_dates", lambda s, e: ["20240102"])
        monkeypatch.setattr(base_module, "get_rebalance_dates", lambda s, e, f: ["20240102"])

        config = StrategyConfig(
            strategy_type="single_factor_topk",
            rebalance_freq="1D",
            delay=0,
            factors=[FactorConfig(id="f_001", direction="asc")],
            selection=SelectionConfig(method="topk", top_k=2),
            weighting=WeightingConfig(method="equal"),
        )
        strategy = SingleFactorStrategy(config)

        # Create deterministic factor data: A=0.1, B=0.2, C=0.3, D=0.4, E=0.5
        data = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "f_001": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        factor_storage = MockFactorStorage(data)
        market_storage = MockMarketStorage()

        signals = strategy.run("20240102", "20240102", factor_storage=factor_storage, market_storage=market_storage)

        # With asc direction, should select A and B (lowest values)
        symbols = set(signals["symbol"])
        assert symbols == {"A", "B"}

    def test_empty_factor_data(self, monkeypatch):
        """Test strategy handles empty factor data gracefully."""
        from backtest.strategy import SingleFactorStrategy, StrategyConfig, FactorConfig
        from backtest.strategy import base as base_module

        monkeypatch.setattr(base_module, "get_trade_dates", lambda s, e: ["20240102"])

        config = StrategyConfig(factors=[FactorConfig(id="f_missing")])
        strategy = SingleFactorStrategy(config)

        factor_storage = MockFactorStorage(pd.DataFrame(columns=["date", "symbol", "f_001"]))
        market_storage = MockMarketStorage()

        with pytest.raises(ValueError, match="No factor data"):
            strategy.run("20240102", "20240102", factor_storage=factor_storage, market_storage=market_storage)

    def test_nan_factor_values_dropped(self, monkeypatch):
        """Test that stocks with NaN factor values are excluded from selection."""
        from backtest.strategy import SingleFactorStrategy, StrategyConfig, FactorConfig
        from backtest.strategy import SelectionConfig, WeightingConfig
        from backtest.strategy import base as base_module

        monkeypatch.setattr(base_module, "get_trade_dates", lambda s, e: ["20240102"])
        monkeypatch.setattr(base_module, "get_rebalance_dates", lambda s, e, f: ["20240102"])

        config = StrategyConfig(
            strategy_type="single_factor_topk",
            rebalance_freq="1D",
            delay=0,
            factors=[FactorConfig(id="f_001", direction="desc")],
            selection=SelectionConfig(method="topk", top_k=3),
            weighting=WeightingConfig(method="equal"),
        )
        strategy = SingleFactorStrategy(config)

        # A=0.5, B=NaN, C=0.3, D=0.2, E=NaN
        data = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "f_001": [0.5, np.nan, 0.3, 0.2, np.nan],
        })
        factor_storage = MockFactorStorage(data)
        market_storage = MockMarketStorage()

        signals = strategy.run("20240102", "20240102",
                               factor_storage=factor_storage, market_storage=market_storage)

        # Should select top 3 from non-NaN: A(0.5), C(0.3), D(0.2)
        # B and E have NaN factor values, should be excluded
        symbols = set(signals["symbol"])
        assert symbols == {"A", "C", "D"}
        assert "B" not in symbols
        assert "E" not in symbols
        assert len(signals) == 3


class TestMultiFactorStrategy:
    """Test MultiFactorStrategy end-to-end with mock data."""

    def test_zscore_equal_combination(self, monkeypatch):
        """Test multi-factor strategy with zscore_equal combination."""
        from backtest.strategy import MultiFactorStrategy, StrategyConfig, FactorConfig
        from backtest.strategy import SelectionConfig, WeightingConfig
        from backtest.strategy import base as base_module

        trade_dates = pd.date_range("2024-01-01", periods=10).strftime("%Y%m%d").tolist()
        monkeypatch.setattr(base_module, "get_trade_dates", lambda s, e: trade_dates)
        monkeypatch.setattr(base_module, "get_rebalance_dates", lambda s, e, f: trade_dates)

        config = StrategyConfig(
            strategy_type="multi_factor",
            rebalance_freq="1D",
            delay=0,
            factors=[
                FactorConfig(id="f_001", direction="desc", weight=1.0),
                FactorConfig(id="f_002", direction="asc", weight=0.5),
            ],
            combine_method="zscore_equal",
            selection=SelectionConfig(method="topk", top_k=2),
            weighting=WeightingConfig(method="equal"),
        )
        strategy = MultiFactorStrategy(config)

        factor_storage = MockFactorStorage()
        market_storage = MockMarketStorage()

        signals = strategy.run("20240101", "20240103", factor_storage=factor_storage, market_storage=market_storage)

        assert not signals.empty
        for date, group in signals.groupby("date"):
            assert len(group) == 2
            assert group["target_weight"].sum() == pytest.approx(1.0)

    def test_requires_two_factors(self):
        """Test MultiFactorStrategy requires at least 2 factors."""
        from backtest.strategy import MultiFactorStrategy, StrategyConfig, FactorConfig

        config = StrategyConfig(factors=[FactorConfig(id="f_001")])
        with pytest.raises(ValueError, match="at least 2 factors"):
            MultiFactorStrategy(config)
