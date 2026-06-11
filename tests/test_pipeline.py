"""Tests for the factor mining pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import backtest.pipeline.steps as pipeline_steps
from backtest.pipeline.config import PipelineConfig, StepThresholds
from backtest.pipeline.state import PipelineState, StepResult
from backtest.factor.evaluation import _ic_series, _rank_ic_series
from backtest.pipeline.steps import (
    FULL_MARKET_UNIVERSE,
    _build_tag,
    run_pipeline,
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
        assert cfg.thresholds.min_sharpe_simple == 0.4
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

    def test_clear_from_step(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        state.record("step1", StepResult(passed=True))
        state.record("step2", StepResult(passed=True))
        state.record("step3", StepResult(passed=True))
        state.record("step4", StepResult(passed=True))
        state.record("step5", StepResult(passed=True, metrics={"top_k": 100}))
        state.record("step6", StepResult(passed=False, reason="mdd"))
        state.artifacts["strategy_config"] = "old.json"
        state.artifacts["report"] = "old/report.md"
        state.status = "rejected"

        state.clear_from_step("step5")

        assert "step5" not in state.step_results
        assert "step6" not in state.step_results
        assert "step1" in state.step_results
        assert "step2" in state.step_results
        assert "strategy_config" not in state.artifacts
        assert "report" not in state.artifacts
        assert state.status == "running"

    def test_save_load_restores_strategy_config(self):
        from backtest.strategy.config import (
            BacktestConfig,
            FactorConfig,
            SelectionConfig,
            StrategyConfig,
        )

        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        state.strategy_config = StrategyConfig(
            name="test_strategy",
            rebalance_freq="5D",
            factors=[FactorConfig(id="f_001", direction="desc")],
            selection=SelectionConfig(method="topk", top_k=50),
            backtest=BacktestConfig(start_date="20200101", end_date="20241231"),
            decay=10,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state.save(path)
            loaded = PipelineState.load(path)

        assert loaded.strategy_config is not None
        assert loaded.strategy_config.name == "test_strategy"
        assert loaded.strategy_config.selection.top_k == 50
        assert loaded.strategy_config.rebalance_freq == "5D"
        assert loaded.strategy_config.decay == 10

    def test_clear_from_step_removes_signals_artifact(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        for i in range(1, 7):
            state.record(f"step{i}", StepResult(passed=True))
        state.artifacts["signals"] = "old_signals.parquet"
        state.artifacts["strategy_config"] = "old_strategy.json"

        state.clear_from_step("step6")

        assert "signals" not in state.artifacts
        assert state.artifacts["strategy_config"] == "old_strategy.json"
        assert "step5" in state.step_results
        assert "step6" not in state.step_results

    def test_clear_from_step7_preserves_strategy_and_signals_artifacts(self):
        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20241231",
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        for i in range(1, 8):
            state.record(f"step{i}", StepResult(passed=True))
        state.artifacts["strategy_config"] = "strategy.json"
        state.artifacts["signals"] = "signals.parquet"
        state.artifacts["simple_bt"] = "simple"
        state.artifacts["detailed_bt"] = "detailed"
        state.artifacts["report"] = "old/report.md"

        state.clear_from_step("step7")

        assert "step6" in state.step_results
        assert "step7" not in state.step_results
        assert state.artifacts["strategy_config"] == "strategy.json"
        assert state.artifacts["signals"] == "signals.parquet"
        assert state.artifacts["simple_bt"] == "simple"
        assert "detailed_bt" not in state.artifacts
        assert "report" not in state.artifacts

    def test_step7_restores_strategy_and_signals_from_artifacts(self, tmp_path, monkeypatch):
        from backtest.strategy.config import (
            BacktestConfig,
            FactorConfig,
            SelectionConfig,
            StrategyConfig,
        )

        cfg = PipelineConfig(
            factor_id="f_001",
            start_date="20200101",
            end_date="20240131",
            results_root=str(tmp_path / "results"),
        )
        state = PipelineState(factor_id="f_001", config=cfg)
        strategy = StrategyConfig(
            name="test_strategy",
            rebalance_freq="1D",
            factors=[FactorConfig(id="f_001", direction="desc")],
            selection=SelectionConfig(method="topk", top_k=1),
            backtest=BacktestConfig(start_date="20200101", end_date="20240131"),
            decay=5,
        )
        strategy_path = tmp_path / "strategy_config.json"
        strategy_path.write_text(json.dumps(strategy.to_dict()), encoding="utf-8")
        signals_path = tmp_path / "signals.parquet"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "symbol": ["000001.SZ"],
                "target_weight": [1.0],
            }
        ).to_parquet(signals_path, index=False)
        state.artifacts["strategy_config"] = str(strategy_path)
        state.artifacts["signals"] = str(signals_path)

        class DummyDetailedSimulator:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, signals, market_data, dividends):
                assert not signals.empty
                return object()

        def fake_backtest_gate(state, result, step, sub_dir, threshold_map):
            assert state.strategy_config is not None
            assert state.signals is not None
            state.record("step7", StepResult(passed=True, metrics={"sharpe": 1.0}))
            return state

        monkeypatch.setattr(pipeline_steps, "DetailedSimulator", DummyDetailedSimulator)
        monkeypatch.setattr(
            pipeline_steps,
            "_load_market_data",
            lambda config, signals, with_dividends=False: (pd.DataFrame(), pd.DataFrame()),
        )
        monkeypatch.setattr(pipeline_steps, "_backtest_gate", fake_backtest_gate)

        result = pipeline_steps.step7_detailed_backtest(state)

        assert result.step_results["step7"].passed is True
        assert result.strategy_config is not None
        assert result.strategy_config.selection.top_k == 1
        assert result.signals is not None
        assert result.signals["date"].dtype.kind == "M"

    def test_run_pipeline_from_step_clears_rejected_state(self, tmp_path, monkeypatch):
        factor_id = "f_resume"
        cfg = PipelineConfig(
            factor_id=factor_id,
            start_date="20200101",
            end_date="20241231",
            results_root=str(tmp_path),
        )
        state = PipelineState(factor_id=factor_id, config=cfg)
        for i in range(1, 6):
            state.record(f"step{i}", StepResult(passed=True))
        state.record("step6", StepResult(passed=False, reason="old fail"))
        state.save(cfg.state_path())

        calls = []

        def _make_step(name):
            def _step(s):
                calls.append(name)
                s.record(name, StepResult(passed=True))
                return s
            return _step

        def _step5(s, **kwargs):
            calls.append("step5")
            s.record("step5", StepResult(passed=True, metrics=kwargs))
            return s

        monkeypatch.setattr(
            pipeline_steps,
            "_STEP_ORDER",
            [
                ("step1", _make_step("step1")),
                ("step2", _make_step("step2")),
                ("step3", _make_step("step3")),
                ("step4", _make_step("step4")),
                ("step5", _step5),
                ("step6", _make_step("step6")),
            ],
        )

        result = run_pipeline(
            factor_id,
            results_root=str(tmp_path),
            from_step=5,
            to_step=6,
            skip_report=True,
            skip_mark_rejected=True,
            top_k=20,
        )

        assert calls == ["step5", "step6"]
        assert result.status == "running"
        assert result.step_results["step6"].passed is True
        assert result.step_results["step5"].metrics["top_k"] == 20

    def test_run_pipeline_accumulates_state_json_for_step1_to_step10(
        self, tmp_path, monkeypatch
    ):
        factor_id = "f_pipeline_integration"
        calls: list[str] = []

        def _make_step(name):
            def _step(s):
                calls.append(name)
                s.artifacts[f"{name}_artifact"] = f"{name}.json"
                s.record(name, StepResult(passed=True, metrics={"ordinal": int(name[4:])}))
                return s
            return _step

        def _step5(s, **kwargs):
            calls.append("step5")
            assert kwargs["top_k"] == 20
            s.artifacts["strategy_config"] = "strategy_config.json"
            s.record("step5", StepResult(passed=True, metrics={"top_k": kwargs["top_k"]}))
            return s

        monkeypatch.setattr(
            pipeline_steps,
            "_STEP_ORDER",
            [
                ("step1", _make_step("step1")),
                ("step2", _make_step("step2")),
                ("step3", _make_step("step3")),
                ("step4", _make_step("step4")),
                ("step5", _step5),
                ("step6", _make_step("step6")),
                ("step7", _make_step("step7")),
                ("step8", _make_step("step8")),
                ("step9", _make_step("step9")),
                ("step10", _make_step("step10")),
            ],
        )

        result = run_pipeline(
            factor_id,
            start_date="20240101",
            end_date="20240131",
            results_root=str(tmp_path),
            skip_report=True,
            skip_mark_rejected=True,
            top_k=20,
        )

        state_path = tmp_path / factor_id / "pipeline_state.json"
        assert calls == [f"step{i}" for i in range(1, 11)]
        assert result.status == "running"
        assert result.current_step == "step10"
        assert state_path.exists()

        with state_path.open(encoding="utf-8") as f:
            data = json.load(f)

        assert list(data["step_results"]) == [f"step{i}" for i in range(1, 11)]
        assert data["step_results"]["step1"]["metrics"]["ordinal"] == 1
        assert data["step_results"]["step5"]["metrics"]["top_k"] == 20
        assert data["step_results"]["step10"]["passed"] is True
        assert data["artifacts"]["step1_artifact"] == "step1.json"
        assert data["artifacts"]["strategy_config"] == "strategy_config.json"
        assert data["artifacts"]["step10_artifact"] == "step10.json"


class TestStep5Universe:
    def test_full_market_sentinel_bypasses_default_index_universe(self):
        cfg = PipelineConfig(
            factor_id="f_full_market",
            start_date="20200101",
            end_date="20200131",
            default_universe="000300.SH",
        )
        state = PipelineState(factor_id="f_full_market", config=cfg)

        result = step5_build_strategy(
            state,
            top_k=100,
            decay=5,
            rebalance="1D",
            universe=FULL_MARKET_UNIVERSE,
        )

        universe = result.strategy_config.universe
        assert universe.index_members is None
        assert universe.exclude_st is True
        assert universe.exclude_new_ipo_days == 252
        assert universe.include_cyb is True
        assert universe.include_kcb is False
        assert universe.include_bse is False
        assert universe.min_market_cap == 500_000_000
        assert universe.min_avg_amount == 10_000_000
        assert result.strategy_config.backtest.benchmark == "000300.SH"


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
            lambda **kw: mock_fs,
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
            lambda **kw: mock_fs,
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
            lambda **kw: mock_fs,
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

    def test_strategy_config_artifact_uses_state_subdir_when_isolated(self, tmp_path):
        cfg = PipelineConfig(
            factor_id="f_iso",
            start_date="20200101",
            end_date="20241231",
            results_root=str(tmp_path / "results"),
            results_subdir="f_iso/default",
            state_subdir="f_iso/default/top100_1d_d5",
        )
        state = PipelineState(factor_id="f_iso", config=cfg)

        result = step5_build_strategy(state, top_k=100, decay=5, rebalance="1D")

        artifact_path = Path(result.artifacts["strategy_config"])
        assert artifact_path == cfg.state_path().parent / "strategy_config.json"
        assert artifact_path.exists()
        assert artifact_path.parent != cfg.results_dir()


# ---------------------------------------------------------------------------
# Threshold constant tests
# ---------------------------------------------------------------------------


class TestThresholdConstants:
    """Ensure threshold constants match config.yaml expectations."""

    def test_daily_icir_thresholds(self):
        th = StepThresholds()
        assert th.min_abs_ic == 0.01
        assert th.min_annual_icir == 1.0
        assert th.min_ic_tstat == 2.0
        assert th.min_ic_positive_ratio == 0.55

    def test_daily_backtest_thresholds(self):
        th = StepThresholds()
        assert th.min_sharpe_simple == 0.4
        assert th.min_annual_return_simple == 0.12
        assert th.max_max_drawdown == 0.50
        assert th.min_calmar_simple == 0.4
        assert th.max_annual_turnover == 50.0

    def test_detailed_backtest_thresholds(self):
        th = StepThresholds()
        assert th.min_sharpe_detailed == 0.35
        assert th.min_annual_return_detailed == 0.10
        assert th.min_calmar_detailed == 0.4

    def test_monotonicity_threshold(self):
        th = StepThresholds()
        assert th.min_monotonicity == 0.7

    def test_coverage_thresholds(self):
        th = StepThresholds()
        assert th.max_missing_rate_pv == 0.20
        assert th.max_missing_rate_fin == 0.30
