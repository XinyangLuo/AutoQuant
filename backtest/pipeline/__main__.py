"""CLI dispatcher for the factor mining pipeline.

Usage::

    python -m backtest.pipeline init f_001
    python -m backtest.pipeline step1 f_001
    python -m backtest.pipeline step2 f_001
    ...
    python -m backtest.pipeline step10 f_001

    python -m backtest.pipeline run-all f_001

Date range is read from ``config.yaml`` (``pipeline.start_date`` / ``pipeline.end_date``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backtest.pipeline.config import PipelineConfig
from backtest.pipeline.runner import GeneratedFactorPipelineRunner
from backtest.pipeline.state import PipelineState, StepResult
from backtest.pipeline.steps import _STEP_ORDER

STEP_FUNCTIONS = dict(_STEP_ORDER)
STEP_ORDER = [name for name, _ in _STEP_ORDER]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Factor mining pipeline (step1~step10)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialise pipeline state")
    p_init.add_argument("factor_id")
    p_init.add_argument("--frequency", choices=["D", "M"], default="D")
    p_init.add_argument("--results-root", default="results")

    # step1~step10
    for i in range(1, 11):
        p = sub.add_parser(f"step{i}", help=f"Run step {i}")
        p.add_argument("factor_id")
        p.add_argument("--results-root", default="results")
        if i == 5:
            p.add_argument("--top-k", type=int)
            p.add_argument("--top-pct", type=float)
            p.add_argument("--decay", type=int)
            p.add_argument("--universe")
            p.add_argument("--rebalance")

    # run-all
    p_run = sub.add_parser("run-all", help="Run all steps sequentially")
    p_run.add_argument("factor_id")
    p_run.add_argument("--frequency", choices=["D", "M"], default="D")
    p_run.add_argument("--from-step", type=int, default=1, choices=range(1, 6))
    p_run.add_argument("--results-root", default="results")
    p_run.add_argument("--top-k", type=int)
    p_run.add_argument("--top-pct", type=float)
    p_run.add_argument("--decay", type=int)
    p_run.add_argument("--universe")
    p_run.add_argument("--rebalance")

    # run generated factor code through register/backfill + pipeline
    p_generated = sub.add_parser(
        "run",
        help="Register/backfill a factor file, then run the pipeline",
    )
    p_generated.add_argument("factor_id")
    p_generated.add_argument("--factor-file", required=True)
    p_generated.add_argument("--frequency", choices=["D", "M"], default="D")
    p_generated.add_argument(
        "--generated-dir",
        default="alphas/exp/agent",
        help="Directory where importable generated factors are written",
    )
    p_generated.add_argument("--results-root", default="results")
    p_generated.add_argument("--result-path")
    p_generated.add_argument("--from-step", type=int, default=1, choices=range(1, 11))
    p_generated.add_argument("--to-step", type=int, choices=range(1, 11))
    p_generated.add_argument("--top-k", type=int)
    p_generated.add_argument("--top-pct", type=float)
    p_generated.add_argument("--decay", type=int)
    p_generated.add_argument("--universe")
    p_generated.add_argument("--rebalance")
    p_generated.add_argument("--keep-work-db", action="store_true")

    return parser


def _load_or_init_state(args) -> PipelineState:
    """Load existing state or create a new one."""
    state_path = Path(args.results_root) / args.factor_id / "pipeline_state.json"

    if state_path.exists():
        return PipelineState.load(state_path)

    # Create new state (for run-all without prior init)
    config = PipelineConfig.from_factor_config(
        args.factor_id,
        frequency=getattr(args, "frequency", "D"),
        results_root=args.results_root,
    )
    state = PipelineState(factor_id=args.factor_id, config=config)
    state.save(state_path)
    return state


def _save_state(state: PipelineState) -> None:
    state.save(state.config.state_path())


def _run_step(step_name: str, state: PipelineState, cli_kwargs: dict | None = None) -> tuple[bool, str | None, dict]:
    """Execute a single step.

    Returns (passed, reason, metrics).
    """
    fn = STEP_FUNCTIONS[step_name]
    state = fn(state, **(cli_kwargs or {}))
    result = state.step_results[step_name]
    return result.passed, result.reason, result.metrics


def _output_json(step_name: str, passed: bool, reason: str | None, metrics: dict) -> None:
    output = {
        "step": step_name,
        "passed": passed,
        "reason": reason,
        "metrics": metrics,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_clean_json(data), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _pipeline_result_payload(
    *,
    state: PipelineState,
    factor_file_path: Path | None,
    result_path: Path,
) -> dict:
    return {
        "factor_id": state.factor_id,
        "status": state.status,
        "result_path": str(result_path),
        "factor_file": str(factor_file_path) if factor_file_path else None,
        "report_path": state.artifacts.get("report") or None,
        "step_results": {
            name: {"passed": sr.passed, "reason": sr.reason, "metrics": sr.metrics}
            for name, sr in state.step_results.items()
        },
        "artifacts": state.artifacts,
    }


def cmd_init(args) -> int:
    config = PipelineConfig.from_factor_config(
        args.factor_id,
        frequency=args.frequency,
        results_root=args.results_root,
    )
    state = PipelineState(factor_id=args.factor_id, config=config)
    state.save(config.state_path())
    print(f"Initialised pipeline state: {config.state_path()}")
    return 0


def cmd_step(args, step_name: str) -> int:
    state = _load_or_init_state(args)

    # Check prerequisites
    idx = STEP_ORDER.index(step_name)
    for prev in STEP_ORDER[:idx]:
        prev_result = state.step_results.get(prev)
        if prev_result is None:
            print(f"Error: prerequisite step {prev} has not been run yet.", file=sys.stderr)
            return 2
        if not prev_result.passed:
            print(f"Error: prerequisite step {prev} did not pass.", file=sys.stderr)
            return 2

    # Build CLI kwargs for step5
    cli_kwargs = {}
    if step_name == "step5":
        for key in ["top_k", "top_pct", "decay", "universe", "rebalance"]:
            val = getattr(args, key, None)
            if val is not None:
                cli_kwargs[key] = val

    passed, reason, metrics = _run_step(step_name, state, cli_kwargs)
    _save_state(state)
    _output_json(step_name, passed, reason, metrics)

    return 0 if passed else 1


def cmd_run_all(args) -> int:
    from backtest.pipeline.steps import run_pipeline

    state = run_pipeline(
        factor_id=args.factor_id,
        frequency=args.frequency,
        results_root=args.results_root,
        from_step=args.from_step,
        top_k=args.top_k,
        top_pct=args.top_pct,
        decay=args.decay,
        universe=args.universe,
        rebalance=args.rebalance,
    )

    for step_name in [s for s in STEP_ORDER if s in state.step_results]:
        sr = state.step_results[step_name]
        _output_json(step_name, sr.passed, sr.reason, sr.metrics)

    report_path = state.artifacts.get("report", "")
    print(f"\nReport: {report_path}", file=sys.stderr)

    if state.is_rejected():
        last = state.last_step()
        reason = state.step_results.get(last, StepResult(False)).reason if last else "unknown"
        print(f"\nPipeline REJECTED at {last}: {reason}", file=sys.stderr)
        return 1

    print(f"\nPipeline complete: {state.status}", file=sys.stderr)
    return 0


def cmd_run_generated(args) -> int:
    factor_file = Path(args.factor_file)
    if not factor_file.exists():
        print(f"Error: factor file not found: {factor_file}", file=sys.stderr)
        return 2

    config = PipelineConfig.from_factor_config(
        args.factor_id,
        frequency=args.frequency,
        results_root=args.results_root,
    )
    factor_code = factor_file.read_text(encoding="utf-8")
    result_path = Path(args.result_path) if args.result_path else (
        Path(args.results_root) / args.factor_id / "result.json"
    )

    with GeneratedFactorPipelineRunner(
        start_date=config.start_date,
        end_date=config.end_date,
        results_root=args.results_root,
        frequency=args.frequency,
        generated_dir=args.generated_dir,
    ) as runner:
        try:
            run = runner.run_factor_code(
                factor_id=args.factor_id,
                factor_code=factor_code,
                from_step=args.from_step,
                to_step=args.to_step,
                top_k=args.top_k,
                top_pct=args.top_pct,
                decay=args.decay,
                universe=args.universe,
                rebalance=args.rebalance,
                skip_report=False,
                skip_mark_rejected=True,
            )
        except Exception:
            if not args.keep_work_db:
                runner.cleanup_work_db(args.factor_id)
            raise

    payload = _pipeline_result_payload(
        state=run.state,
        factor_file_path=run.factor_file_path or factor_file,
        result_path=result_path,
    )
    _write_json(result_path, payload)
    print(json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, default=str))

    return 1 if run.state.is_rejected() else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return cmd_init(args)

    if args.command == "run-all":
        return cmd_run_all(args)

    if args.command == "run":
        return cmd_run_generated(args)

    if args.command.startswith("step"):
        return cmd_step(args, args.command)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
