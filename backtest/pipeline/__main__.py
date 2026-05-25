"""CLI dispatcher for the factor mining pipeline.

Usage::

    python -m backtest.pipeline init f_001 --start 20160101 --end 20251231
    python -m backtest.pipeline step1 f_001
    python -m backtest.pipeline step2 f_001
    ...
    python -m backtest.pipeline step10 f_001

    python -m backtest.pipeline run-all f_001 --start 20160101 --end 20251231
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backtest.pipeline.config import PipelineConfig
from backtest.pipeline.state import PipelineState, StepResult
from backtest.pipeline.steps import (
    step1_coverage_check,
    step2_neutralization_check,
    step3_icir_check,
    step4_monotonicity_check,
    step5_build_strategy,
    step6_simple_backtest,
    step7_detailed_backtest,
    step8_ridge_r2,
    step9_residual_icir,
    step10_report_and_admit,
)

STEP_FUNCTIONS = {
    "step1": step1_coverage_check,
    "step2": step2_neutralization_check,
    "step3": step3_icir_check,
    "step4": step4_monotonicity_check,
    "step5": step5_build_strategy,
    "step6": step6_simple_backtest,
    "step7": step7_detailed_backtest,
    "step8": step8_ridge_r2,
    "step9": step9_residual_icir,
    "step10": step10_report_and_admit,
}

STEP_ORDER = ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Factor mining pipeline (step1~step10)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialise pipeline state")
    p_init.add_argument("factor_id")
    p_init.add_argument("--start", default=None, help="default from config.yaml")
    p_init.add_argument("--end", default=None, help="default from config.yaml")
    p_init.add_argument("--frequency", choices=["D", "M"], default="D")
    p_init.add_argument("--results-root", default="results")
    p_init.add_argument("--ret-type", default="open", choices=["close", "open"])
    p_init.add_argument("--benchmark", default="000300.SH")

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
    p_run.add_argument("--start", default=None, help="default from config.yaml")
    p_run.add_argument("--end", default=None, help="default from config.yaml")
    p_run.add_argument("--frequency", choices=["D", "M"], default="D")
    p_run.add_argument("--from-step", type=int, default=1, choices=range(1, 11))
    p_run.add_argument("--results-root", default="results")
    p_run.add_argument("--ret-type", default="open", choices=["close", "open"])
    p_run.add_argument("--benchmark", default="000300.SH")

    return parser


def _resolve_dates(args) -> tuple[str, str]:
    """Resolve start/end dates: CLI args take priority, fall back to config.yaml."""
    from backtest.config_loader import get_section

    start = args.start or None  # treat empty string as None
    end = args.end or None
    if not start:
        start = get_section("pipeline", "start_date")
    if not end:
        end = get_section("pipeline", "end_date")
    return start, end


def _load_or_init_state(args) -> PipelineState:
    """Load existing state or create a new one."""
    state_path = Path(args.results_root) / args.factor_id / "pipeline_state.json"

    if state_path.exists():
        return PipelineState.load(state_path)

    # Create new state (for run-all without prior init)
    start_date, end_date = _resolve_dates(args)
    config = PipelineConfig.for_frequency(
        frequency=getattr(args, "frequency", "D"),
        factor_id=args.factor_id,
        start_date=start_date,
        end_date=end_date,
        results_root=args.results_root,
        ret_type=getattr(args, "ret_type", "open"),
        benchmark=getattr(args, "benchmark", "000300.SH"),
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
    start_date, end_date = _resolve_dates(args)
    config = PipelineConfig.for_frequency(
        frequency=args.frequency,
        factor_id=args.factor_id,
        start_date=start_date,
        end_date=end_date,
        results_root=args.results_root,
        ret_type=args.ret_type,
        benchmark=args.benchmark,
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
    state = _load_or_init_state(args)
    from_step = args.from_step
    rejected_step: str | None = None

    for step_name in STEP_ORDER[from_step - 1:]:
        print(f"\n--- Running {step_name} ---", file=sys.stderr)

        passed, reason, metrics = _run_step(step_name, state)
        _save_state(state)
        _output_json(step_name, passed, reason, metrics)

        if not passed:
            print(f"\nREJECTED at {step_name}: {reason}", file=sys.stderr)
            rejected_step = step_name
            # Stop running further steps but still generate report below.
            break

    # Always generate a diagnostic report — even on rejection.
    from backtest.pipeline._report import generate_pipeline_report

    report_path = generate_pipeline_report(state)
    print(f"\nReport: {report_path}", file=sys.stderr)

    if rejected_step:
        # Mark as rejected in registry so `admission status` reflects reality.
        # Use mark_rejected() instead of reject() to preserve work DB data.
        from backtest.factor.admission import mark_rejected
        reason = state.step_results.get(rejected_step, StepResult(False)).reason
        mark_rejected(
            state.config.factor_id,
            notes=f"Pipeline rejected at {rejected_step}: {reason}",
        )
        state.status = "rejected"
        _save_state(state)
        print(f"\nPipeline REJECTED at {rejected_step}. Factor data preserved in work DB.", file=sys.stderr)
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
