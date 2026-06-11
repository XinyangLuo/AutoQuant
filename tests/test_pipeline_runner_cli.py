"""Tests for generated-factor pipeline entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

from backtest.pipeline.config import PipelineConfig
from backtest.pipeline.state import PipelineState, StepResult


def test_pipeline_run_command_uses_generated_factor_runner(
    tmp_path, monkeypatch, capsys
):
    import backtest.pipeline.__main__ as pipeline_cli
    from backtest.pipeline.runner import GeneratedFactorRun

    factor_file = tmp_path / "factor.py"
    factor_file.write_text("# factor code\n", encoding="utf-8")
    result_path = tmp_path / "result.json"
    calls: dict[str, object] = {}

    class DummyRunner:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def run_factor_code(self, **kwargs):
            calls["run"] = kwargs
            cfg = PipelineConfig(
                factor_id=kwargs["factor_id"],
                start_date="20200101",
                end_date="20200131",
                results_root=str(tmp_path),
            )
            state = PipelineState(factor_id=kwargs["factor_id"], config=cfg)
            state.record("step1", StepResult(passed=True, metrics={"ok": 1}))
            state.status = "quick_pass"
            state.artifacts["report"] = str(tmp_path / "pipeline_report.md")
            return GeneratedFactorRun(state=state, factor_file_path=factor_file)

    monkeypatch.setattr(pipeline_cli, "GeneratedFactorPipelineRunner", DummyRunner)

    rc = pipeline_cli.main(
        [
            "run",
            "f_generated",
            "--factor-file",
            str(factor_file),
            "--results-root",
            str(tmp_path),
            "--result-path",
            str(result_path),
            "--to-step",
            "1",
            "--top-k",
            "20",
        ]
    )

    assert rc == 0
    assert calls["init"]["results_root"] == str(tmp_path)
    assert calls["init"]["frequency"] == "D"
    assert calls["run"]["factor_id"] == "f_generated"
    assert calls["run"]["factor_code"] == "# factor code\n"
    assert calls["run"]["to_step"] == 1
    assert calls["run"]["top_k"] == 20

    payload = json.loads(capsys.readouterr().out)
    assert payload["factor_id"] == "f_generated"
    assert payload["status"] == "quick_pass"
    assert payload["result_path"] == str(result_path)
    assert payload["step_results"]["step1"]["metrics"] == {"ok": 1}
    assert json.loads(result_path.read_text(encoding="utf-8")) == payload


def test_agent_runner_reuses_generated_factor_pipeline_runner():
    from agents.runner import AutoQuantFactorRunner
    from backtest.pipeline.runner import GeneratedFactorPipelineRunner

    assert issubclass(AutoQuantFactorRunner, GeneratedFactorPipelineRunner)
