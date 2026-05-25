"""Tests for the factor mining pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.pipeline.config import PipelineConfig, StepThresholds
from backtest.pipeline.state import PipelineState, StepResult
from backtest.factor.evaluation import _ic_series, _rank_ic_series
from backtest.pipeline.steps import (
    _build_tag,
    step1_coverage_check,
    step5_build_strategy,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_daily_defaults(self):
        cfg = PipelineConfig.for_frequency(
            "D",
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        assert cfg.frequency == "D"
        assert cfg.thresholds.min_abs_ic == 0.01
        assert cfg.thresholds.min_annual_icir == 1.0
        assert cfg.thresholds.min_sharpe_simple == 0.8
        assert cfg.icir_check_horizons == [1, 5]

    def test_monthly_defaults(self):
        cfg = PipelineConfig.for_frequency(
            "M",
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        assert cfg.frequency == "M"
        assert cfg.thresholds.min_abs_ic == 0.03
        assert cfg.thresholds.min_annual_icir == 0.8
        assert cfg.thresholds.min_sharpe_simple == 1.0
        assert cfg.icir_check_horizons == [20]

    def test_state_path(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        assert cfg.state_path() == Path("results/f_001/pipeline_state.json")


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


class TestPipelineState:
    def test_save_load_roundtrip(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        state.record("step1", StepResult(passed=True, metrics={"x": 1}))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state.save(path)
            loaded = PipelineState.load(path)

        assert loaded.factor_id == "f_001"
        assert loaded.status == "running"
        assert loaded.step_results["step1"].passed is True
        assert loaded.step_results["step1"].metrics == {"x": 1}

    def test_can_proceed_to(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        assert state.can_proceed_to("step1") is True
        assert state.can_proceed_to("step2") is False  # step1 not run yet

        state.record("step1", StepResult(passed=True))
        assert state.can_proceed_to("step2") is True
        assert state.can_proceed_to("step3") is False  # step2 not run yet

    def test_rejection_blocks_downstream(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        state.record("step1", StepResult(passed=False, reason="fail"))
        assert state.is_rejected() is True
        assert state.can_proceed_to("step2") is False

    def test_get_result(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        assert state.get_result("step1") is None
        state.record("step1", StepResult(passed=True))
        assert state.get_result("step1").passed is True


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestICSeries:
    def test_perfect_positive(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _ic_series(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _ic_series(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        assert np.isnan(_ic_series(a, b))

    def test_with_nans(self):
        a = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _ic_series(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_insufficient_data(self):
        a = pd.Series([1.0, 2.0])
        b = pd.Series([1.0, 2.0])
        assert np.isnan(_ic_series(a, b))


class TestRankICSeries:
    def test_perfect_positive(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _rank_ic_series(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _rank_ic_series(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_monotonic_nonlinear(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0])
        assert _rank_ic_series(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_constant_y(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([5.0, 5.0, 5.0])
        assert np.isnan(_rank_ic_series(a, b))


class TestBuildTag:
    def test_with_top_pct(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        state.strategy_config = MagicMock()
        state.strategy_config.selection.top_pct = 0.1
        state.strategy_config.selection.top_k = None
        state.strategy_config.rebalance_freq = "1D"
        state.strategy_config.decay = 5
        assert _build_tag(state) == "top10pct_1d_d5"

    def test_with_top_k(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        state.strategy_config = MagicMock()
        state.strategy_config.selection.top_pct = None
        state.strategy_config.selection.top_k = 50
        state.strategy_config.rebalance_freq = "1W"
        state.strategy_config.decay = 0
        assert _build_tag(state) == "top50_1w_d0"

    def test_no_strategy_config(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        assert _build_tag(state) == "default"


# ---------------------------------------------------------------------------
# Step function tests (mock-based)
# ---------------------------------------------------------------------------


class TestStep1Coverage:
    @pytest.fixture(autouse=True)
    def clean_registry(self):
        from backtest.factor import registry
        registry._REGISTRY_CACHE = {}
        registry._FACTOR_FUNCTIONS.clear()
        yield
        registry._REGISTRY_CACHE = {}
        registry._FACTOR_FUNCTIONS.clear()

    def _make_factor_df(self, n_dates=10, n_symbols=100, missing_rate=0.0):
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        symbols = [f"S{i:03d}" for i in range(n_symbols)]
        df = pd.DataFrame({
            "date": np.repeat(dates, n_symbols),
            "symbol": np.tile(symbols, n_dates),
            "value": np.random.randn(n_dates * n_symbols),
        })
        if missing_rate > 0:
            n_missing = int(len(df) * missing_rate)
            mask = np.random.choice(len(df), n_missing, replace=False)
            df.loc[mask, "value"] = np.nan
        return df

    def test_pass_low_missing_rate(self, monkeypatch):
        cfg = PipelineConfig(
            factor_id="f_test",
            start_date="20240101",
            end_date="20240115",
        )
        state = PipelineState(factor_id="f_test", config=cfg)

        # Mock get_factor_meta — price/volume factor
        monkeypatch.setattr(
            "backtest.pipeline.steps.get_factor_meta",
            lambda _: {"data_sources": ["market_daily"]},
        )

        # Mock FactorStorage.get_factor
        factor_df = self._make_factor_df(n_dates=5, n_symbols=100, missing_rate=0.02)
        mock_fs = MagicMock()
        mock_fs.get_factor.return_value = factor_df
        mock_fs.__enter__.return_value = mock_fs
        mock_fs.__exit__.return_value = False
        monkeypatch.setattr(
            "backtest.pipeline.steps.FactorStorage",
            lambda: mock_fs,
        )

        # Mock MarketStorage.get_bars
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        symbols = [f"S{i:03d}" for i in range(100)]
        mock_bars = pd.DataFrame({
            "date": np.repeat(dates, len(symbols)),
            "symbol": np.tile(symbols, len(dates)),
        })
        mock_ms = MagicMock()
        mock_ms.get_bars.return_value = mock_bars
        mock_ms.__enter__.return_value = mock_ms
        mock_ms.__exit__.return_value = False
        monkeypatch.setattr(
            "backtest.pipeline.steps.MarketStorage",
            lambda **kw: mock_ms,
        )

        result = step1_coverage_check(state)
        assert result.step_results["step1"].passed is True
        assert result.step_results["step1"].metrics["max_missing_rate"] < 0.10

    def test_fail_high_missing_rate(self, monkeypatch):
        cfg = PipelineConfig(
            factor_id="f_test",
            start_date="20240101",
            end_date="20240115",
        )
        state = PipelineState(factor_id="f_test", config=cfg)

        monkeypatch.setattr(
            "backtest.pipeline.steps.get_factor_meta",
            lambda _: {"data_sources": ["market_daily"]},
        )

        factor_df = self._make_factor_df(n_dates=5, n_symbols=100, missing_rate=0.20)
        mock_fs = MagicMock()
        mock_fs.get_factor.return_value = factor_df
        mock_fs.__enter__.return_value = mock_fs
        mock_fs.__exit__.return_value = False
        monkeypatch.setattr(
            "backtest.pipeline.steps.FactorStorage",
            lambda: mock_fs,
        )

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        symbols = [f"S{i:03d}" for i in range(100)]
        mock_bars = pd.DataFrame({
            "date": np.repeat(dates, len(symbols)),
            "symbol": np.tile(symbols, len(dates)),
        })
        mock_ms = MagicMock()
        mock_ms.get_bars.return_value = mock_bars
        mock_ms.__enter__.return_value = mock_ms
        mock_ms.__exit__.return_value = False
        monkeypatch.setattr(
            "backtest.pipeline.steps.MarketStorage",
            lambda **kw: mock_ms,
        )

        result = step1_coverage_check(state)
        assert result.step_results["step1"].passed is False
        assert "missing rate" in result.step_results["step1"].reason

    def test_financial_factor_higher_threshold(self, monkeypatch):
        cfg = PipelineConfig(
            factor_id="f_test",
            start_date="20240101",
            end_date="20240115",
        )
        state = PipelineState(factor_id="f_test", config=cfg)

        monkeypatch.setattr(
            "backtest.pipeline.steps.get_factor_meta",
            lambda _: {"data_sources": ["income_q"]},
        )

        # 15% missing — above PV threshold (10%) but below FIN threshold (30%)
        factor_df = self._make_factor_df(n_dates=5, n_symbols=100, missing_rate=0.15)
        mock_fs = MagicMock()
        mock_fs.get_factor.return_value = factor_df
        mock_fs.__enter__.return_value = mock_fs
        mock_fs.__exit__.return_value = False
        monkeypatch.setattr(
            "backtest.pipeline.steps.FactorStorage",
            lambda: mock_fs,
        )

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        symbols = [f"S{i:03d}" for i in range(100)]
        mock_bars = pd.DataFrame({
            "date": np.repeat(dates, len(symbols)),
            "symbol": np.tile(symbols, len(dates)),
        })
        mock_ms = MagicMock()
        mock_ms.get_bars.return_value = mock_bars
        mock_ms.__enter__.return_value = mock_ms
        mock_ms.__exit__.return_value = False
        monkeypatch.setattr(
            "backtest.pipeline.steps.MarketStorage",
            lambda **kw: mock_ms,
        )

        result = step1_coverage_check(state)
        # Should PASS because financial threshold is 30%
        assert result.step_results["step1"].passed is True
        assert result.step_results["step1"].metrics["is_financial"] is True


class TestStep5BuildStrategy:
    def test_default_params(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        result = step5_build_strategy(state)

        assert result.step_results["step5"].passed is True
        assert result.strategy_config is not None
        assert result.strategy_config.selection.top_k == 100
        assert result.strategy_config.decay == 5
        assert result.strategy_config.rebalance_freq == "1D"

    def test_retry_params_override(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(
            factor_id="f_001",
            config=cfg,
            retry_params={"top_k": 30, "decay": 10},
        )
        result = step5_build_strategy(state)

        assert result.strategy_config.selection.top_k == 30
        assert result.strategy_config.decay == 10

    def test_cli_kwargs_override(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        result = step5_build_strategy(state, top_k=20, decay=3, universe="000300.SH")

        assert result.strategy_config.selection.top_k == 20
        assert result.strategy_config.decay == 3
        assert result.strategy_config.universe.index_members == "000300.SH"


# ---------------------------------------------------------------------------
# Threshold constant tests
# ---------------------------------------------------------------------------


class TestThresholdConstants:
    """Ensure threshold constants match PLAN.md §4 expectations."""

    def test_daily_icir_thresholds(self):
        th = StepThresholds()
        assert th.min_abs_ic == 0.01
        assert th.min_annual_icir == 1.0
        assert th.min_ic_tstat == 2.0
        assert th.min_ic_positive_ratio == 0.55

    def test_daily_backtest_thresholds(self):
        th = StepThresholds()
        assert th.min_sharpe_simple == 0.8
        assert th.min_annual_return_simple == 0.10
        assert th.max_max_drawdown == 0.50
        assert th.min_calmar_simple == 0.5
        assert th.max_annual_turnover == 50.0

    def test_detailed_backtest_thresholds(self):
        th = StepThresholds()
        assert th.min_sharpe_detailed == 0.4
        assert th.min_annual_return_detailed == 0.08
        assert th.min_calmar_detailed == 0.5

    def test_monotonicity_threshold(self):
        th = StepThresholds()
        assert th.min_monotonicity == 0.7

    def test_coverage_thresholds(self):
        th = StepThresholds()
        assert th.max_missing_rate_pv == 0.20
        assert th.max_missing_rate_fin == 0.30
