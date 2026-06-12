from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agents import sweep as sweep_mod
from agents.experiment import AutoQuantFactorExperiment
from backtest.pipeline.config import PipelineConfig
from backtest.pipeline._report import _bt_metrics_table
from backtest.pipeline.state import PipelineState, StepResult
from backtest.pipeline.steps import (
    _MARKET_CACHE_MEM,
    _load_simulation_config,
    _read_market_cache_for_step6,
    _summarize_backtest_result,
    step5_build_strategy,
)
from backtest.simulation.simple import SimpleSimulator


class _ImmediateFuture:
    def __init__(self, fn, *args):
        self._result = fn(*args)

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args):
        return _ImmediateFuture(fn, *args)


def _fake_as_completed(futures):
    return list(futures)


class _DummyRunner:
    def __init__(self, *args, **kwargs):
        self.start_date = kwargs.get("start_date") or args[0]
        self.end_date = kwargs.get("end_date") or args[1]
        self.results_root = Path(kwargs.get("results_root", "results"))
        self.results_subdir = kwargs.get("results_subdir")
        self.state_subdir = kwargs.get("state_subdir")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, experiment: AutoQuantFactorExperiment, *, from_step=1, to_step=None, **kwargs):
        cfg = PipelineConfig(
            factor_id=experiment.factor_id,
            start_date="20200101",
            end_date="20200131",
            results_root=str(self.results_root),
            results_subdir=self.results_subdir,
            state_subdir=self.state_subdir,
        )
        if from_step == 1:
            state = PipelineState(factor_id=experiment.factor_id, config=cfg)
            for i in range(1, 5):
                state.record(f"step{i}", StepResult(passed=True, metrics={"step": i}))
            state.status = "quick_pass"
            state.save(cfg.state_path())
            experiment.status = "quick_pass"
            return experiment

        if kwargs.get("top_k") is not None:
            tag = f"top{kwargs['top_k']}_{kwargs.get('rebalance', '1D').lower()}_d{kwargs.get('decay', 0)}"
        else:
            pct = int(round(float(kwargs.get("top_pct", 0.0)) * 100))
            tag = f"top{pct}pct_{kwargs.get('rebalance', '1D').lower()}_d{kwargs.get('decay', 0)}"
        report = cfg.results_dir() / tag / "pipeline_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# report\n", encoding="utf-8")
        plots = report.parent / "plots"
        plots.mkdir()
        (plots / "nav.png").write_text("plot\n", encoding="utf-8")
        experiment.report_path = str(report)
        experiment.status = "candidate" if from_step == 7 else "quick_pass"
        experiment.step_results = {
            "step5": {"passed": True, "metrics": kwargs},
            "step6": {"passed": True, "metrics": {"sharpe": 1.0}},
        }
        experiment.simple_bt_metrics = {
            "sharpe": 1.0,
            "annual_return": 0.2,
            "max_drawdown": -0.1,
            "calmar": 2.0,
        }
        return experiment


def _write_factor_file(path: Path) -> None:
    path.write_text(
        "from __future__ import annotations\n"
        "import pandas as pd\n"
        "from backtest.factor.registry import register\n\n"
        "@register('f_test_sweep', name='test', category='test', data_sources=['market_daily'])\n"
        "def test_factor(panel: pd.DataFrame) -> pd.Series:\n"
        "    return panel['close']\n",
        encoding="utf-8",
    )


def _synthetic_benchmark_navs(timestamps: pd.DatetimeIndex) -> dict[str, pd.Series]:
    n_dates = len(timestamps)
    return {
        "hs300": pd.Series([1.0 + i * 0.001 for i in range(n_dates)], index=timestamps),
        "csi500": pd.Series([1.0 + i * 0.0008 for i in range(n_dates)], index=timestamps),
        "csi1000": pd.Series([1.0 + i * 0.0006 for i in range(n_dates)], index=timestamps),
        "csi2000": pd.Series([1.0 + i * 0.0004 for i in range(n_dates)], index=timestamps),
    }


def _install_synthetic_batch_env(
    monkeypatch,
    tmp_path: Path,
    factor_id: str = "f_test_sweep",
    *,
    n_symbols: int = 6,
    n_dates: int = 10,
    market_extra_dates: int = 0,
    index_members: dict[str, set[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Install deterministic panels/calendars for sweep Step6 tests."""
    market_timestamps = pd.bdate_range(
        "2020-01-01",
        periods=n_dates + market_extra_dates,
    )
    trade_dates = [
        d.strftime("%Y%m%d")
        for d in market_timestamps[:n_dates]
    ]
    timestamps = pd.to_datetime(trade_dates)
    base_symbols = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "300001.SZ",
        "688001.SH",
        "000005.SZ",
        "000006.SZ",
        "000007.SZ",
        "000008.SZ",
        "000009.SZ",
        "000010.SZ",
    ]
    symbols = base_symbols[:n_symbols]

    market_rows: list[dict] = []
    factor_rows: list[dict] = []
    for d_idx, date in enumerate(market_timestamps):
        for s_idx, symbol in enumerate(symbols):
            close = 10.0 + s_idx + d_idx * (0.20 + s_idx * 0.03)
            is_st = symbol == "000003.SZ"
            is_new = symbol == "000004.SZ"
            is_illiquid = symbol == "000002.SZ"
            market_rows.append({
                "date": date,
                "symbol": symbol,
                "close": close,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "adj_factor": 1.0,
                "circ_mv": 100_000 + s_idx * 10_000,
                "amount": 30_000_000.0,
                "avg_amount_20": 5_000_000.0 if is_illiquid else 30_000_000.0,
                "_avg_amount_20_n": 20,
                "is_st": 1 if is_st else 0,
                "list_date": "20200106" if is_new else "20180101",
                "limit_up": close * 1.1,
                "limit_down": close * 0.9,
            })
    for d_idx, date in enumerate(timestamps):
        for s_idx, symbol in enumerate(symbols):
            if not (d_idx == 2 and symbol == symbols[0]):
                factor_rows.append({
                    "date": date,
                    "symbol": symbol,
                    factor_id: float(s_idx * 10 + d_idx),
                })

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    factor_cache = cache_dir / "factor_panel.parquet"
    market_cache = cache_dir / "market_panel.parquet"
    pd.DataFrame(factor_rows).to_parquet(factor_cache, index=False)
    pd.DataFrame(market_rows).to_parquet(market_cache, index=False)

    def _warm_shared_cache(_factor_id, _cfg, _results_root):
        return str(factor_cache), str(market_cache)

    def _between(start: str, end: str) -> list[str]:
        return [d for d in trade_dates if start <= d <= end]

    def _rebalance(start: str, end: str, freq: str) -> list[str]:
        dates = _between(start, end)
        if freq == "5D":
            return dates[::5]
        if freq in {"1M", "3M"}:
            return dates[:1]
        return dates

    class _Conn:
        def execute(self, *_args, **_kwargs):
            return self

        def fetchall(self):
            return [("index_members",)]

    class _SyntheticMarketStorage:
        conn = _Conn()

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def close(self):
            return None

        def get_index_members(self, _index_code: str, date: str) -> set[str]:
            if index_members is not None:
                return index_members.get(date, set())
            return set(symbols)

    bench_navs = _synthetic_benchmark_navs(timestamps)

    monkeypatch.setattr(sweep_mod, "_warm_shared_cache", _warm_shared_cache)
    monkeypatch.setattr(sweep_mod, "_load_benchmark_navs", lambda *_args, **_kwargs: bench_navs, raising=False)
    monkeypatch.setattr("backtest.data.storage.MarketStorage", _SyntheticMarketStorage)
    monkeypatch.setattr("backtest.strategy.base.get_trade_dates", _between)
    monkeypatch.setattr("backtest.strategy.base.get_rebalance_dates", _rebalance)
    monkeypatch.setattr("backtest.strategy.universe.get_trade_dates", _between)

    return trade_dates, symbols


def test_pipeline_report_backtest_table_includes_csi2000_column():
    lines = _bt_metrics_table({
        "annual_return": 0.10,
        "excess_annual_return_hs300": 0.01,
        "excess_annual_return_csi500": 0.02,
        "excess_annual_return_csi1000": 0.03,
        "excess_annual_return_csi2000": 0.04,
    })

    assert "相对中证2000" in lines[0]
    assert "4.00%" in "\n".join(lines)


def test_sweep_does_not_create_clone_factor_dirs(tmp_path, monkeypatch):
    _install_synthetic_batch_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sweep_mod,
        "_get_strategy_combos",
        lambda _factor_id, _universe_name: [
            sweep_mod.StrategyCombo(decay=0, rebalance="1D", top_k=2),
            sweep_mod.StrategyCombo(decay=3, rebalance="5D", top_k=3),
        ],
    )
    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(sweep_mod, "as_completed", _fake_as_completed)
    monkeypatch.setattr(sweep_mod, "AutoQuantFactorRunner", _DummyRunner)

    generated_dir = tmp_path / "alphas" / "exp" / "agent"
    generated_dir.mkdir(parents=True)
    factor_dir = generated_dir / "f_test_sweep"
    factor_dir.mkdir()
    factor_file = factor_dir / "factor.py"
    _write_factor_file(factor_file)

    results = sweep_mod.run_sweep(
        factor_id="f_test_sweep",
        factor_file=factor_file,
        generated_dir=generated_dir,
        results_root=tmp_path / "results",
        workers=2,
        universes={"default": None},
    )

    assert results["factor_id"] == "f_test_sweep"
    assert results["n_universes"] == 1
    assert not list(generated_dir.glob("f_test_sweep_sw_*"))
    assert results["best_overall"]["universe"] == "default"
    assert results["best_overall"]["combo_tag"] in {"top2_1d_d0", "top3_5d_d3"}
    assert results["best_overall"]["top_k"] in {2, 3}
    assert results["universes"]["default"]["n_combos"] == 2
    assert {r["params"].get("top_k") for r in results["universes"]["default"]["all_results"]} == {2, 3}
    assert all(
        "top_pct" not in r["params"]
        for r in results["universes"]["default"]["all_results"]
    )
    for result in results["universes"]["default"]["all_results"]:
        state_path = tmp_path / "results" / result["state_subdir"] / "pipeline_state.json"
        assert state_path.exists()
        state = PipelineState.load(state_path)
        assert state.step_results["step5"].passed is True
        assert state.step_results["step6"].passed is True
        assert "signals" in state.artifacts
        assert "simple_bt" in state.artifacts
        nav_path = Path(state.artifacts["simple_bt"]) / "nav.parquet"
        assert nav_path.exists()
    candidate_dir = tmp_path / "results" / "candidates" / "f_test_sweep"
    assert (candidate_dir / "factor.py").exists()
    assert (candidate_dir / "result.json").exists()
    assert (candidate_dir / "pipeline_report.md").exists()
    assert (candidate_dir / "plots" / "nav.png").exists()
    assert not list((tmp_path / "results" / "f_test_sweep").glob("**/f_test_sweep_sw_*"))


def test_sweep_batch_step6_matches_single_combo_reference(tmp_path, monkeypatch):
    trade_dates, _symbols = _install_synthetic_batch_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sweep_mod,
        "_get_strategy_combos",
        lambda _factor_id, _universe_name: [
            sweep_mod.StrategyCombo(decay=0, rebalance="1D", top_k=2),
        ],
    )
    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(sweep_mod, "as_completed", _fake_as_completed)
    monkeypatch.setattr(sweep_mod, "AutoQuantFactorRunner", _DummyRunner)

    generated_dir = tmp_path / "alphas" / "exp" / "agent"
    generated_dir.mkdir(parents=True)
    factor_dir = generated_dir / "f_test_sweep"
    factor_dir.mkdir()
    factor_file = factor_dir / "factor.py"
    _write_factor_file(factor_file)

    results = sweep_mod.run_sweep(
        factor_id="f_test_sweep",
        factor_file=factor_file,
        generated_dir=generated_dir,
        results_root=tmp_path / "results",
        workers=2,
        validate_top_n=0,
        universes={"default": None},
    )

    payload = results["universes"]["default"]["all_results"][0]
    state = PipelineState.load(tmp_path / "results" / payload["state_subdir"] / "pipeline_state.json")
    signals = pd.read_parquet(state.artifacts["signals"])
    market_panel = pd.read_parquet(tmp_path / "cache" / "market_panel.parquet")
    market_panel["date"] = pd.to_datetime(market_panel["date"])
    market_data = market_panel[market_panel["symbol"].isin(signals["symbol"].unique())]

    sim = SimpleSimulator(
        _load_simulation_config(overrides=state.config.simulation_overrides)
    )
    expected = sim.run(signals, market_data)
    actual_nav = pd.read_parquet(Path(state.artifacts["simple_bt"]) / "nav.parquet")
    pd.testing.assert_frame_equal(
        actual_nav.reset_index(drop=True),
        expected.nav_df.reset_index(drop=True),
    )

    bench_navs = _synthetic_benchmark_navs(pd.to_datetime(trade_dates))
    expected_metrics = _summarize_backtest_result(expected, bench_navs=bench_navs)
    persisted_metrics = state.step_results["step6"].metrics
    for key in (
        "sharpe",
        "annual_return",
        "max_drawdown",
        "calmar",
        "excess_sharpe_hs300",
        "excess_annual_return_hs300",
        "excess_max_drawdown_hs300",
        "excess_calmar_hs300",
    ):
        actual_value = persisted_metrics[key]
        expected_value = expected_metrics[key]
        if pd.isna(expected_value):
            assert pd.isna(actual_value)
        else:
            assert actual_value == pytest.approx(expected_value)


def test_sweep_batch_step6_loads_benchmarks_through_market_buffer(tmp_path, monkeypatch):
    loaded_ranges: list[tuple[str, str]] = []

    _install_synthetic_batch_env(
        monkeypatch,
        tmp_path,
        n_dates=23,
        market_extra_dates=5,
    )
    market_panel = pd.read_parquet(tmp_path / "cache" / "market_panel.parquet")
    expected_start = pd.to_datetime(market_panel["date"]).min().strftime("%Y%m%d")
    expected_end = pd.to_datetime(market_panel["date"]).max().strftime("%Y%m%d")

    def _recording_benchmark_loader(start: str, end: str) -> dict[str, pd.Series]:
        loaded_ranges.append((start, end))
        return _synthetic_benchmark_navs(pd.bdate_range(start=start, end=end))

    monkeypatch.setattr(
        sweep_mod,
        "_load_benchmark_navs",
        _recording_benchmark_loader,
        raising=False,
    )
    monkeypatch.setattr(
        sweep_mod,
        "_get_strategy_combos",
        lambda _factor_id, _universe_name: [
            sweep_mod.StrategyCombo(decay=0, rebalance="1D", top_k=2),
        ],
    )
    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(sweep_mod, "as_completed", _fake_as_completed)
    monkeypatch.setattr(sweep_mod, "AutoQuantFactorRunner", _DummyRunner)

    generated_dir = tmp_path / "alphas" / "exp" / "agent"
    generated_dir.mkdir(parents=True)
    factor_dir = generated_dir / "f_test_sweep"
    factor_dir.mkdir()
    factor_file = factor_dir / "factor.py"
    _write_factor_file(factor_file)

    sweep_mod.run_sweep(
        factor_id="f_test_sweep",
        factor_file=factor_file,
        generated_dir=generated_dir,
        results_root=tmp_path / "results",
        workers=2,
        validate_top_n=0,
        universes={"default": None},
    )

    assert loaded_ranges == [(expected_start, expected_end)]


def test_sweep_to_step_5_stops_after_strategy_config(tmp_path, monkeypatch):
    _install_synthetic_batch_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sweep_mod,
        "_get_strategy_combos",
        lambda _factor_id, _universe_name: [
            sweep_mod.StrategyCombo(decay=0, rebalance="1D", top_k=2),
        ],
    )
    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(sweep_mod, "as_completed", _fake_as_completed)
    monkeypatch.setattr(sweep_mod, "AutoQuantFactorRunner", _DummyRunner)

    generated_dir = tmp_path / "alphas" / "exp" / "agent"
    generated_dir.mkdir(parents=True)
    factor_dir = generated_dir / "f_test_sweep"
    factor_dir.mkdir()
    factor_file = factor_dir / "factor.py"
    _write_factor_file(factor_file)

    results = sweep_mod.run_sweep(
        factor_id="f_test_sweep",
        factor_file=factor_file,
        generated_dir=generated_dir,
        results_root=tmp_path / "results",
        workers=2,
        validate_top_n=0,
        universes={"default": None},
        to_step=5,
    )

    payload = results["universes"]["default"]["all_results"][0]
    state = PipelineState.load(tmp_path / "results" / payload["state_subdir"] / "pipeline_state.json")

    assert payload["status"] == "partial"
    assert state.step_results["step5"].passed is True
    assert "step6" not in state.step_results
    assert "signals" not in state.artifacts
    assert "simple_bt" not in state.artifacts


def test_sweep_validate_top_n_resumes_best_combo_from_step7(tmp_path, monkeypatch):
    calls: list[int] = []

    class TrackingRunner(_DummyRunner):
        def run(self, experiment, *, from_step=1, to_step=None, **kwargs):
            calls.append(from_step)
            return super().run(experiment, from_step=from_step, to_step=to_step, **kwargs)

    _install_synthetic_batch_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sweep_mod,
        "_get_strategy_combos",
        lambda _factor_id, _universe_name: [
            sweep_mod.StrategyCombo(decay=0, rebalance="1D", top_k=2),
            sweep_mod.StrategyCombo(decay=3, rebalance="5D", top_k=3),
        ],
    )
    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(sweep_mod, "as_completed", _fake_as_completed)
    monkeypatch.setattr(sweep_mod, "AutoQuantFactorRunner", TrackingRunner)

    generated_dir = tmp_path / "alphas" / "exp" / "agent"
    generated_dir.mkdir(parents=True)
    factor_dir = generated_dir / "f_test_sweep"
    factor_dir.mkdir()
    factor_file = factor_dir / "factor.py"
    _write_factor_file(factor_file)

    results = sweep_mod.run_sweep(
        factor_id="f_test_sweep",
        factor_file=factor_file,
        generated_dir=generated_dir,
        results_root=tmp_path / "results",
        workers=2,
        validate_top_n=1,
        universes={"default": None},
    )

    assert calls.count(1) == 1
    assert calls.count(5) == 0
    assert calls.count(7) == 1
    assert results["best_overall"]["universe"] == "default"


def test_sweep_explicit_to_step_7_uses_legacy_combo_workers(tmp_path, monkeypatch):
    calls: list[int] = []

    class TrackingRunner(_DummyRunner):
        def run(self, experiment, *, from_step=1, to_step=None, **kwargs):
            calls.append(from_step)
            return super().run(experiment, from_step=from_step, to_step=to_step, **kwargs)

    _install_synthetic_batch_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sweep_mod,
        "_get_strategy_combos",
        lambda _factor_id, _universe_name: [
            sweep_mod.StrategyCombo(decay=0, rebalance="1D", top_k=2),
            sweep_mod.StrategyCombo(decay=3, rebalance="5D", top_k=3),
        ],
    )
    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(sweep_mod, "as_completed", _fake_as_completed)
    monkeypatch.setattr(sweep_mod, "AutoQuantFactorRunner", TrackingRunner)

    generated_dir = tmp_path / "alphas" / "exp" / "agent"
    generated_dir.mkdir(parents=True)
    factor_dir = generated_dir / "f_test_sweep"
    factor_dir.mkdir()
    factor_file = factor_dir / "factor.py"
    _write_factor_file(factor_file)

    sweep_mod.run_sweep(
        factor_id="f_test_sweep",
        factor_file=factor_file,
        generated_dir=generated_dir,
        results_root=tmp_path / "results",
        workers=2,
        validate_top_n=0,
        universes={"default": None},
        to_step=7,
    )

    assert calls.count(1) == 1
    assert calls.count(5) == 2
    assert calls.count(7) == 0


def test_sweep_standard_universes_include_default_all_a():
    assert sweep_mod.UNIVERSES["default"] is None


def test_index_universe_sweep_uses_top_pct(tmp_path, monkeypatch):
    index_trade_dates = [
        x.strftime("%Y%m%d")
        for x in pd.bdate_range("2020-01-01", periods=10)
    ]
    index_member_symbols = {
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "300001.SZ",
        "688001.SH",
    }
    trade_dates, symbols = _install_synthetic_batch_env(
        monkeypatch,
        tmp_path,
        n_symbols=12,
        index_members={
            date: set(index_member_symbols)
            for date in index_trade_dates
        },
    )
    assert trade_dates
    assert symbols
    monkeypatch.setattr(
        sweep_mod,
        "_get_strategy_combos",
        lambda _factor_id, _universe_name: [
            sweep_mod.StrategyCombo(decay=0, rebalance="1D", top_pct=0.25),
        ],
    )
    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(sweep_mod, "as_completed", _fake_as_completed)
    monkeypatch.setattr(sweep_mod, "AutoQuantFactorRunner", _DummyRunner)

    generated_dir = tmp_path / "alphas" / "exp" / "agent"
    generated_dir.mkdir(parents=True)
    factor_dir = generated_dir / "f_test_sweep"
    factor_dir.mkdir()
    factor_file = factor_dir / "factor.py"
    _write_factor_file(factor_file)

    results = sweep_mod.run_sweep(
        factor_id="f_test_sweep",
        factor_file=factor_file,
        generated_dir=generated_dir,
        results_root=tmp_path / "results",
        workers=2,
        universes={"hs300": "000300.SH"},
    )

    assert results["best_overall"]["universe"] == "hs300"
    assert results["best_overall"]["combo_tag"] == "top25pct_1d_d0"
    assert results["best_overall"]["top_pct"] == 0.25
    assert results["universes"]["hs300"]["n_combos"] == 1
    assert {r["params"].get("top_pct") for r in results["universes"]["hs300"]["all_results"]} == {0.25}
    assert all(
        "top_k" not in r["params"]
        for r in results["universes"]["hs300"]["all_results"]
    )
    result = results["universes"]["hs300"]["all_results"][0]
    state = PipelineState.load(tmp_path / "results" / result["state_subdir"] / "pipeline_state.json")
    assert state.config.benchmark == "000300.SH"
    assert state.strategy_config.backtest.benchmark == "000300.SH"
    signals = pd.read_parquet(state.artifacts["signals"])
    first_rebalance_symbols = set(
        signals[signals["date"] == signals["date"].min()]["symbol"]
    )
    assert len(first_rebalance_symbols) == 1
    assert first_rebalance_symbols <= {"688001.SH", "300001.SZ", "000004.SZ"}


def test_seed_combo_state_copies_only_step1_to_step4(tmp_path):
    base_cfg = PipelineConfig(
        factor_id="f_seed",
        start_date="20200101",
        end_date="20200131",
        results_root=str(tmp_path / "results"),
    )
    base = PipelineState(factor_id="f_seed", config=base_cfg)
    for i in range(1, 7):
        base.record(f"step{i}", StepResult(passed=True, metrics={"step": i}))
    base.artifacts["eval_result"] = "eval.json"
    base.artifacts["signals"] = "old_signals.parquet"

    results_root = tmp_path / "results"
    results_subdir = "f_seed/sweep_runs"
    state_subdir = "f_seed/sweep_runs/top50_1d_d5"
    state_path = sweep_mod._seed_combo_state(
        base_state=base,
        factor_id="f_seed",
        results_root=results_root,
        results_subdir=results_subdir,
        state_subdir=state_subdir,
    )
    loaded = PipelineState.load(state_path)

    assert loaded.factor_id == "f_seed"
    assert loaded.config.results_root == str(results_root)
    assert loaded.config.results_subdir == results_subdir
    assert loaded.config.state_subdir == state_subdir
    assert loaded.config.results_dir() == results_root / results_subdir
    assert loaded.config.state_path() == results_root / state_subdir / "pipeline_state.json"
    assert set(loaded.step_results) == {"step1", "step2", "step3", "step4"}
    assert loaded.current_step == "step4"
    assert loaded.status == "running"
    assert loaded.artifacts == {"eval_result": "eval.json"}


def test_step5_explicit_top_pct_overrides_default_top_k(tmp_path):
    cfg = PipelineConfig(
        factor_id="f_seed",
        start_date="20200101",
        end_date="20200131",
        results_root=str(tmp_path / "results"),
        default_top_k=100,
        default_top_pct=None,
    )
    state = PipelineState(factor_id="f_seed", config=cfg)

    state = step5_build_strategy(
        state,
        top_pct=0.1,
        universe="000300.SH",
        decay=5,
        rebalance="1D",
    )

    assert state.step_results["step5"].metrics["top_pct"] == 0.1
    assert "top_k" not in state.step_results["step5"].metrics
    assert state.strategy_config is not None
    assert state.strategy_config.selection.top_pct == 0.1
    assert state.strategy_config.selection.top_k is None


def test_step6_market_cache_requires_adj_factor_and_filters_buffer(tmp_path, monkeypatch):
    cfg = PipelineConfig(
        factor_id="f_seed",
        start_date="20200101",
        end_date="20200103",
        results_root=str(tmp_path / "results"),
    )
    cache_path = tmp_path / "market.parquet"
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-12-31", "2020-01-01", "2020-01-13", "2020-01-20"]
            ),
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "close": [0.9, 1.0, 1.1, 1.2],
            "open": [0.9, 1.0, 1.1, 1.2],
            "high": [0.9, 1.0, 1.1, 1.2],
            "low": [0.9, 1.0, 1.1, 1.2],
            "adj_factor": [1.0, 1.0, 1.0, 1.0],
            "circ_mv": [1.0, 1.0, 1.0, 1.0],
            "amount": [1.0, 1.0, 1.0, 1.0],
            "is_st": [False, False, False, False],
            "list_date": ["20190101", "20190101", "20190101", "20190101"],
            "limit_up": [1.0, 1.1, 1.2, 1.3],
            "limit_down": [0.8, 0.9, 1.0, 1.1],
        }
    )
    market.to_parquet(cache_path, index=False)
    monkeypatch.setenv("AQ_MARKET_CACHE", str(cache_path))

    cached = _read_market_cache_for_step6(cfg)

    assert cached is not None
    assert cached["date"].min() == pd.Timestamp("2020-01-01")
    assert cached["date"].max() == pd.Timestamp("2020-01-13")
    cache_key = str(cache_path.resolve())
    assert cache_key in _MARKET_CACHE_MEM

    cache_path.unlink()
    cached_again = _read_market_cache_for_step6(cfg)

    assert cached_again is not None
    assert cached_again["date"].min() == pd.Timestamp("2020-01-01")
    assert cached_again["date"].max() == pd.Timestamp("2020-01-13")


def test_step6_market_cache_missing_required_column_returns_none(tmp_path, monkeypatch):
    cache_path = tmp_path / "market.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "symbol": ["000001.SZ"],
            "close": [1.0],
        }
    ).to_parquet(cache_path, index=False)
    monkeypatch.setenv("AQ_MARKET_CACHE", str(cache_path))

    cfg = PipelineConfig(factor_id="f_seed", start_date="20200101", end_date="20200103")

    assert _read_market_cache_for_step6(cfg) is None


def test_factor_iterate_prompt_prefers_pre_rc_sweep():
    prompt = Path(".codex/skills/factor-iterate/references/workflow.md").read_text(
        encoding="utf-8"
    )
    assert "Pre-RC Strategy Sweep Fast Path" in prompt
    assert "不要启动 RC" in prompt
    assert "--validate-top-n" in prompt
    assert "不会创建 `alphas/exp/agent/<factor_id>_sw_*`" in prompt
