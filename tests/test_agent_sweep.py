from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents import sweep as sweep_mod
from agents.experiment import AutoQuantFactorExperiment
from backtest.pipeline.config import PipelineConfig
from backtest.pipeline._report import _bt_metrics_table
from backtest.pipeline.state import PipelineState, StepResult
from backtest.pipeline.steps import _MARKET_CACHE_MEM, _read_market_cache_for_step6, step5_build_strategy


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
    assert results["best_overall"]["combo_tag"] == "top100_1d_d5"
    assert results["best_overall"]["top_k"] == 100
    assert results["universes"]["default"]["n_combos"] == 12
    assert {r["params"].get("top_k") for r in results["universes"]["default"]["all_results"]} == {100, 200}
    assert all(
        "top_pct" not in r["params"]
        for r in results["universes"]["default"]["all_results"]
    )
    candidate_dir = tmp_path / "results" / "candidates" / "f_test_sweep"
    assert (candidate_dir / "factor.py").exists()
    assert (candidate_dir / "result.json").exists()
    assert (candidate_dir / "pipeline_report.md").exists()
    assert (candidate_dir / "plots" / "nav.png").exists()
    assert not list((tmp_path / "results" / "f_test_sweep").glob("**/f_test_sweep_sw_*"))


def test_sweep_validate_top_n_resumes_best_combo_from_step7(tmp_path, monkeypatch):
    calls: list[int] = []

    class TrackingRunner(_DummyRunner):
        def run(self, experiment, *, from_step=1, to_step=None, **kwargs):
            calls.append(from_step)
            return super().run(experiment, from_step=from_step, to_step=to_step, **kwargs)

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
    assert calls.count(5) == 12
    assert calls.count(7) == 1
    assert results["best_overall"]["universe"] == "default"


def test_sweep_standard_universes_include_default_all_a():
    assert sweep_mod.UNIVERSES["default"] is None


def test_index_universe_sweep_uses_top_pct(tmp_path, monkeypatch):
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
    assert results["best_overall"]["combo_tag"] == "top10pct_1d_d5"
    assert results["best_overall"]["top_pct"] == 0.1
    assert results["universes"]["hs300"]["n_combos"] == 6
    assert {r["params"].get("top_pct") for r in results["universes"]["hs300"]["all_results"]} == {0.1}
    assert all(
        "top_k" not in r["params"]
        for r in results["universes"]["hs300"]["all_results"]
    )


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
