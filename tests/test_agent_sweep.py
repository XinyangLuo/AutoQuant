from __future__ import annotations

from pathlib import Path

from agents import sweep as sweep_mod
from agents.experiment import AutoQuantFactorExperiment
from backtest.pipeline.config import PipelineConfig
from backtest.pipeline.state import PipelineState, StepResult


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

        tag = sweep_mod._build_combo_tag(
            kwargs.get("top_k", 100), kwargs.get("decay", 0), kwargs.get("rebalance", "1D")
        )
        report = cfg.results_dir() / tag / "pipeline_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# report\n", encoding="utf-8")
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
        top_ks=[50, 100],
        decays=[5],
        rebalances=["1D"],
        workers=2,
    )

    assert len(results) == 2
    assert not list(generated_dir.glob("f_test_sweep_sw_*"))
    assert all(r["factor_id"] == "f_test_sweep" for r in results)
    assert {r["combo_tag"] for r in results} == {"top50_1d_d5", "top100_1d_d5"}
    assert all("clone_id" not in r for r in results)
    assert all(
        Path(r["result_path"]).parent
        == tmp_path / "results" / "f_test_sweep" / "sweep_runs" / r["combo_tag"]
        for r in results
    )
    assert not list((tmp_path / "results" / "f_test_sweep" / "sweep_runs").glob("*/f_test_sweep/*"))


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
        top_ks=[50, 100],
        decays=[5],
        rebalances=["1D"],
        workers=2,
        validate_top_n=1,
    )

    assert calls.count(1) == 1
    assert calls.count(5) == 2
    assert calls.count(7) == 1
    assert sum("full_result" in r for r in results) == 1


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


def test_factor_iterate_prompt_prefers_pre_rc_sweep():
    prompt = Path(".claude/commands/factor-iterate.md").read_text(encoding="utf-8")
    assert "Pre-RC Strategy Sweep Fast Path" in prompt
    assert "不要启动 RC" in prompt
    assert "--validate-top-n" in prompt
    assert "不会创建 `alphas/exp/agent/<factor_id>_sw_*`" in prompt
