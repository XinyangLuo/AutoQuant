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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "init":
        return cmd_init(args)

    if args.command == "run-all":
        return cmd_run_all(args)

    if args.command.startswith("step"):
        return cmd_step(args, args.command)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
