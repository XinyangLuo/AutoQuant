"""Claude Code helper CLI for single-factor iteration.

This module intentionally keeps the loop outside Python: Claude Code generates or
edits factor code, calls this CLI for one deterministic execution, then decides
how to iterate based on the JSON result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import AgentConfig
from .evaluator import AutoQuantFactorEvaluator, QuantFeedback
from .experiment import AutoQuantFactorExperiment
from .helpers import validate_python_code, validate_transforms_imports, force_register_factor_id
from .kb_update import KbUpdater
from .runner import AutoQuantFactorRunner
from .schema import COLUMN_ALIASES, get_panel_columns_for_data_sources
from .trace import TraceManager, TraceRecord

load_dotenv()

_CODE_ERROR_TYPES = (
    "SyntaxError",
    "NameError",
    "TypeError",
    "ImportError",
    "AttributeError",
    "ValueError",
)

_STEP_FAILURE_MAP: dict[str, str] = {
    "step1": "coverage_fail",
    "step2": "neutralization_fail",
    "step3": "icir_fail",
    "step4": "monotonicity_fail",
    "step5": "config_error",
    "step6": "backtest_fail",
    "step7": "backtest_fail",
    "step8": "ridge_fail",
    "step9": "residual_fail",
}


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp{os.getpid()}")
    tmp.write_text(
        json.dumps(_clean_json(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


def _read_factor_code(
    factor_id: str, factor_file: Path | None, generated_dir: Path,
) -> tuple[str, Path]:
    path = factor_file or generated_dir / factor_id / "factor.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Factor file not found: {path}. Pass --factor-file or create "
            f"{generated_dir / factor_id / 'factor.py'}"
        )
    return path.read_text(encoding="utf-8"), path


def _classify_failure(
    *,
    error: str | None,
    feedback: QuantFeedback | None,
) -> str | None:
    """Classify the failure reason from exception or pipeline step result.

    When a pipeline step rejects the factor, the failed_step name is mapped
    to a stable failure type for the trace JSONL.  Exception-based failures
    (code / schema / execution) are classified from the traceback text.
    """
    if error:
        if "KeyError" in error or "not in index" in error or "column" in error.lower():
            return "schema_error"
        if any(t in error for t in _CODE_ERROR_TYPES):
            return "code_error"
        return "execution_error"

    if feedback is None or feedback.decision:
        return None

    failed = feedback.failed_step
    if failed:
        return _STEP_FAILURE_MAP.get(failed, "metrics_fail")
    return "metrics_fail"


def _schema_payload(sources: list[str]) -> dict[str, Any]:
    columns = get_panel_columns_for_data_sources(sources)
    return {
        "sources": sources,
        "columns": columns,
        "aliases": COLUMN_ALIASES,
    }


def cmd_schema(args: argparse.Namespace) -> int:
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    payload = _schema_payload(sources or ["market_daily"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    factor_id = args.factor_id
    run_dir = Path(args.run_dir) if args.run_dir else None
    generated_dir = Path(args.generated_dir)
    cfg = AgentConfig()

    # Determine output layout.
    # Legacy mode: explicit --run-dir (kept for compatibility).
    # New mode: auto-layout under results/<factor_id>/.
    if run_dir:
        result_path = Path(args.result_path) if args.result_path else run_dir / "result.json"
        sanitized_code_path = run_dir / "factor_sanitized.py"
        results_root = run_dir
    else:
        factor_results_dir = Path("results") / factor_id
        factor_results_dir.mkdir(parents=True, exist_ok=True)
        result_path = Path(args.result_path) if args.result_path else factor_results_dir / "result.json"
        sanitized_code_path = Path("results") / factor_id / "factor_sanitized.py"
        results_root = Path("results")

    experiment = AutoQuantFactorExperiment(factor_id=factor_id)
    feedback: QuantFeedback | None = None
    error: str | None = None
    tb: str | None = None
    factor_file_path: Path | None = None

    # --quick is a convenience alias for --to-step 6 (simple backtest only).
    to_step = 6 if getattr(args, "quick", False) and args.to_step is None else args.to_step

    try:
        code, factor_file_path = _read_factor_code(
            factor_id,
            Path(args.factor_file) if args.factor_file else None,
            generated_dir,
        )
        validate_python_code(code)
        validate_transforms_imports(code)
        code = force_register_factor_id(code, factor_id)
        sanitized_code_path.parent.mkdir(parents=True, exist_ok=True)
        sanitized_code_path.write_text(code, encoding="utf-8")

        experiment.factor_code = code
        evaluator = AutoQuantFactorEvaluator()

        with AutoQuantFactorRunner(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            results_root=results_root,
            generated_dir=generated_dir,
        ) as runner:
            try:
                experiment = runner.run(
                    experiment,
                    from_step=args.from_step,
                    to_step=to_step,
                    top_k=args.top_k,
                    top_pct=args.top_pct,
                    decay=args.decay,
                    universe=args.universe,
                    rebalance=args.rebalance,
                )
                feedback = evaluator.evaluate(experiment)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                experiment.error = error
                if not args.keep_work_db:
                    runner.cleanup_work_db(factor_id)
            else:
                if (
                    feedback is not None
                    and not feedback.decision
                    and experiment.status != "quick_pass"
                    and not args.keep_work_db
                ):
                    runner.cleanup_work_db(factor_id)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        experiment.error = error

    failure_type = _classify_failure(error=error, feedback=feedback)

    if error:
        status = "error"
    elif experiment.status == "candidate":
        status = "pass"
    elif experiment.status == "quick_pass":
        status = "quick_pass"
        failure_type = None  # partial run is not a real failure
    else:
        status = "fail"

    # When running in auto-layout mode, place result.json next to the
    # pipeline report so that all artifacts for a given strategy variant
    # live in the same directory.
    if not args.result_path and not run_dir and experiment.report_path:
        tag_dir = Path(experiment.report_path).parent
        tag_dir.mkdir(parents=True, exist_ok=True)
        result_path = tag_dir / "result.json"

    # Build feedback output according to --feedback-format
    feedback_payload: dict[str, Any] | None = None
    if feedback is not None:
        fmt = getattr(args, "feedback_format", "layered")
        if fmt == "flat":
            feedback_payload = feedback.to_flat_dict()
        elif fmt == "relevant":
            feedback_payload = feedback.get_relevant_layer(failure_type)
        else:  # layered (default)
            feedback_payload = {
                "layered": feedback.to_layered_dict(),
                "flat": feedback.to_flat_dict(),
                "relevant": feedback.get_relevant_layer(failure_type),
            }

    result = {
        "factor_id": factor_id,
        "status": status,
        "failure_type": failure_type,
        "error": error,
        "traceback": tb,
        "factor_file": str(factor_file_path) if factor_file_path else None,
        "sanitized_factor_file": str(sanitized_code_path) if sanitized_code_path else None,
        "result_path": str(result_path),
        "run_dir": str(run_dir) if run_dir else str(result_path.parent),
        "thresholds": _read_pipeline_thresholds(),
        "metrics": feedback.metrics if feedback else {},
        "feedback": feedback_payload,
        "experiment": experiment.to_dict(),
        "report_path": experiment.report_path or None,
    }

    _write_json(result_path, result)

    # In legacy mode, copy the pipeline report into the run directory for
    # visibility. In auto-layout mode the report is already at the right place.
    if run_dir:
        _copy_report_to_round(experiment.report_path, run_dir)

    # Auto-trace: append trace record when --run-dir is specified
    # (trace is written even when runner.run() raised, so the iteration loop
    # can read the error signature from trace.jsonl on the next round.)
    if run_dir:
        tm = TraceManager(run_dir)
        arg_round = getattr(args, "round", None)
        trace_record = TraceRecord.from_result_json(
            result,
            round_num=arg_round if arg_round is not None else tm.get_next_round(),
            parent_round_id=getattr(args, "parent_round", None)
            or tm.get_default_parent_round(),
            branch_id=getattr(args, "branch_id", None) or "main",
            category=getattr(args, "category", None) or experiment.category or "",
            data_sources=(
                (getattr(args, "data_sources", None) or "").split(",")
                if getattr(args, "data_sources", None)
                else []
            ),
        )
        tm.append(trace_record)

    # Auto-KB: update knowledge base when --auto-kb-update is enabled
    if getattr(args, "auto_kb_update", False):
        updater = KbUpdater()
        if status == "pass":
            updater.update_on_pass(experiment)
        else:
            # Build a lightweight rc_output proxy from feedback so that
            # anti-patterns are persisted even when cmd_run is invoked
            # directly (without an external RC subagent).
            rc_proxy: dict[str, Any] | None = None
            if feedback:
                rc_proxy = {
                    "failure_type": failure_type,
                    "diagnosis": feedback.observation or "",
                    "fix_strategy": feedback.suggestion or "",
                    "fix_level": "factor" if feedback.failed_step in ("step1", "step2", "step3") else "strategy_only",
                }
            updater.update_on_fail(experiment, rc_output=rc_proxy)

    if status == "pass":
        # Find report for candidates/
        report_path = Path(experiment.report_path) if experiment.report_path else None
        plots_path: Path | None = None
        if report_path and report_path.exists():
            plots_dir = report_path.parent / "plots"
            if plots_dir.is_dir():
                plots_path = plots_dir
        _write_candidate(experiment, factor_id, result_path, report_path, plots_path)

    print(json.dumps(_clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if status != "error" else 1


def _safe_json_load(path: Path, label: str = "file") -> dict[str, Any] | None:
    """Load JSON from path with graceful error handling."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: {label} at {path} contains invalid JSON: {exc}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"Error: {label} not found at {path}", file=sys.stderr)
        return None


def cmd_trace_append(args: argparse.Namespace) -> int:
    """Append a trace record to trace.jsonl from a result.json file."""
    run_dir = Path(args.run_dir)
    result_path = Path(args.result)
    result = _safe_json_load(result_path, "result")
    if result is None:
        return 1

    rc_output = None
    if args.rc_output:
        rc_output = _safe_json_load(Path(args.rc_output), "rc-output")
        if rc_output is None:
            return 1

    tried_params: dict[str, Any] = {}
    if args.tried_params:
        try:
            tried_params = json.loads(args.tried_params)
        except json.JSONDecodeError as exc:
            print(f"Error: --tried-params is not valid JSON: {exc}", file=sys.stderr)
            return 1

    tm = TraceManager(run_dir)
    round_num = args.round if args.round is not None else tm.get_next_round()
    parent_round_id = (
        args.parent_round if args.parent_round is not None else tm.get_default_parent_round()
    )

    trace_record = TraceRecord.from_result_json(
        result,
        round_num=round_num,
        parent_round_id=parent_round_id,
        branch_id=args.branch_id or "main",
        rc_output=rc_output,
        code_summary=args.code_summary or "",
        tried_params=tried_params,
        category=args.category or "",
        data_sources=(args.data_sources or "").split(",") if args.data_sources else [],
    )
    tm.append(trace_record)
    print(
        json.dumps(
            {"status": "ok", "trace_path": str(tm.trace_path), "round": round_num},
            ensure_ascii=False,
        )
    )
    return 0


def cmd_kb_update(args: argparse.Namespace) -> int:
    """Update knowledge base from a result.json file."""
    result_path = Path(args.result)
    result = _safe_json_load(result_path, "result")
    if result is None:
        return 1

    experiment = AutoQuantFactorExperiment.from_dict(result.get("experiment", {}))
    rc_output = None
    if args.rc_output:
        rc_output = _safe_json_load(Path(args.rc_output), "rc-output")
        if rc_output is None:
            return 1

    updater = KbUpdater()
    if args.status == "pass":
        summary = updater.update_on_pass(experiment)
    else:
        summary = updater.update_on_fail(experiment, rc_output=rc_output)

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a parallel strategy-parameter sweep for a single factor.

    The base factor is computed once (step1-4); each parameter combo is then
    given a cloned factor ID and evaluated in a separate process so that
    pipeline state and artifacts do not conflict.
    """
    from .sweep import run_sweep

    factor_id = args.factor_id
    factor_file = Path(args.factor_file) if args.factor_file else None
    if factor_file is None:
        print("Error: --factor-file is required for sweep", file=sys.stderr)
        return 1

    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)

    def _parse_comma_ints(value: str | None) -> list[int]:
        if not value:
            return []
        return [int(x.strip()) for x in value.split(",") if x.strip()]

    def _parse_comma_strs(value: str | None) -> list[str]:
        if not value:
            return []
        return [x.strip() for x in value.split(",") if x.strip()]

    top_ks = _parse_comma_ints(args.top_k)
    decays = _parse_comma_ints(args.decay)
    rebalances = _parse_comma_strs(args.rebalance)

    if not top_ks or not rebalances:
        print(
            "Error: --top-k and --rebalance must each provide "
            "at least one comma-separated value.",
            file=sys.stderr,
        )
        return 1

    if not decays:
        decays = [0]
        print(
            "Error: --top-k, --decay, and --rebalance must each provide "
            "at least one comma-separated value.",
            file=sys.stderr,
        )
        return 1

    to_step = None if args.full else 6

    try:
        results = run_sweep(
            factor_id=factor_id,
            factor_file=factor_file,
            generated_dir=generated_dir,
            results_root=results_root,
            top_ks=top_ks,
            decays=decays,
            rebalances=rebalances,
            to_step=to_step,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"Error: sweep failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # Summary table
    print(f"\nSweep complete: {len(results)} combinations")
    print("")
    header = f"{'top_k':>6} {'decay':>6} {'rebalance':>10} {'status':>12} {'sharpe':>8} {'ann_ret':>9} {'mdd':>9} {'calmar':>8} {'result_path'}"
    print(header)
    print("-" * len(header))
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    for r in sorted(results, key=lambda x: (x["params"]["top_k"], x["params"]["decay"], x["params"]["rebalance"])):
        p = r["params"]
        m = r.get("metrics", {})
        sharpe = m.get("simple_sharpe")
        ann_ret = m.get("simple_annual_return")
        mdd = m.get("simple_mdd")
        calmar = m.get("simple_calmar")
        print(
            f"{p['top_k']:>6} {p['decay']:>6} {p['rebalance']:>10} {r['status']:>12} "
            f"{_fmt_metric(sharpe):>8} {_fmt_metric(ann_ret):>9} {_fmt_metric(mdd):>9} "
            f"{_fmt_metric(calmar):>8} {r.get('result_path') or ''}"
        )
        score = calmar if calmar is not None else (sharpe or 0)
        if not math.isnan(score) and score > best_score and r["status"] in ("pass", "quick_pass"):
            best_score = score
            best = r

    print("")
    if best:
        print(
            f"Best combo (by calmar>sharpe): top_k={best['params']['top_k']} "
            f"decay={best['params']['decay']} rebalance={best['params']['rebalance']} "
            f"→ {best['result_path']}"
        )
    else:
        print("No passing or quick-pass combinations found.")

    # Emit machine-readable summary next to the base factor results.
    summary_path = results_root / factor_id / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(summary_path, {"factor_id": factor_id, "results": results})
    print(f"Summary written to {summary_path}")
    return 0


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _copy_report_to_round(report_path: str, run_dir: Path) -> None:
    """Copy pipeline report from results/<fid>/<tag>/ to the round directory.

    The pipeline always generates a report at
    ``results_root/<factor_id>/<tag>/pipeline_report.md`` (via step10).
    This copies it into ``run_dir/pipeline_report.md`` so the user can
    review it alongside the per-round result.json.
    """
    src = Path(report_path) if report_path else None
    if not src or not src.exists():
        return
    dst = run_dir / "pipeline_report.md"
    try:
        dst.write_text(src.read_text(encoding="utf-8"))
        # Also copy plots if they exist
        plots_src = src.parent / "plots"
        plots_dst = run_dir / "plots"
        if plots_src.is_dir() and not plots_dst.exists():
            import shutil
            shutil.copytree(str(plots_src), str(plots_dst))
    except Exception:
        pass


def _write_candidate(
    experiment: AutoQuantFactorExperiment,
    factor_id: str,
    result_path: Path,
    report_path: Path | None = None,
    plots_path: Path | None = None,
) -> None:
    """Write passing factor to ``results/candidates/<factor_id>/`` for human review."""
    candidates_root = Path("results/candidates")
    candidate_dir = candidates_root / factor_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    if experiment.factor_code:
        (candidate_dir / "factor.py").write_text(experiment.factor_code, encoding="utf-8")

    if result_path.exists():
        shutil.copy2(result_path, candidate_dir / "result.json")

    if report_path and report_path.exists():
        shutil.copy2(report_path, candidate_dir / "pipeline_report.md")

    if plots_path and plots_path.is_dir():
        dst_plots = candidate_dir / "plots"
        if dst_plots.exists():
            shutil.rmtree(dst_plots)
        shutil.copytree(plots_path, dst_plots)

    state = {
        "factor_id": factor_id,
        "status": experiment.status,
        "step_results": experiment.step_results,
        "eval_result": experiment.eval_result,
        "simple_bt_metrics": experiment.simple_bt_metrics,
        "detailed_bt_metrics": experiment.detailed_bt_metrics,
        "ridge_result": experiment.ridge_result,
        "residual_icir_result": experiment.residual_icir_result,
    }
    _write_json(candidate_dir / "pipeline_state.json", state)


def _read_pipeline_thresholds() -> dict[str, Any]:
    """Read pipeline step thresholds from config.yaml via StepThresholds."""
    try:
        from dataclasses import asdict

        from backtest.pipeline.config import StepThresholds

        return asdict(StepThresholds())
    except Exception:
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agents.claude_cli",
        description="Claude Code helper CLI for AutoQuant factor iteration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    schema = sub.add_parser("schema", help="Print panel columns for selected data sources")
    schema.add_argument(
        "--sources", default="market_daily",
        help="Comma-separated data sources, e.g. market_daily,income_q",
    )
    schema.set_defaults(func=cmd_schema)

    run = sub.add_parser("run", help="Run one generated factor through the AutoQuant pipeline")
    run.add_argument("factor_id", help="Factor id to run, e.g. f_auto_run001_001")
    run.add_argument("--run-dir", help="Optional explicit round output directory (legacy). When omitted, artifacts are written to results/<factor_id>/ automatically.")
    run.add_argument("--factor-file", help="Path to the factor code file to execute")
    run.add_argument(
        "--generated-dir", default="alphas/exp/agent",
        help="Directory where the runner writes importable generated factors",
    )
    run.add_argument("--result-path", help="Optional explicit JSON result path")
    run.add_argument(
        "--keep-work-db", action="store_true",
        help="Keep failed factor values in the pending factor DB for inspection",
    )
    run.add_argument(
        "--from-step", type=int, default=1, choices=range(1, 11),
        help="Start pipeline from this step (1-10). Use >1 to skip register+backfill.",
    )
    run.add_argument(
        "--to-step", type=int, default=None, choices=range(1, 11),
        help="Stop pipeline after this step (1-10). Use 6 for quick mode: only run through simple backtest.",
    )
    run.add_argument(
        "--quick", action="store_true",
        help="Quick mode: stop after step6 (simple backtest). Equivalent to --to-step 6.",
    )
    run.add_argument("--top-k", type=int, help="Override top_k for step5 strategy")
    run.add_argument("--top-pct", type=float, help="Override top_pct for step5 strategy")
    run.add_argument("--decay", type=int, help="Override decay for step5 strategy")
    run.add_argument("--universe", type=str, help="Override universe for step5 strategy")
    run.add_argument("--rebalance", type=str, help="Override rebalance for step5 strategy")

    # Trace / iteration parameters (only meaningful with --run-dir)
    run.add_argument("--round", type=int, help="Round number for trace (auto-inferred if omitted)")
    run.add_argument("--parent-round", type=int, help="Parent round ID for DAG trace")
    run.add_argument("--branch-id", type=str, default="main", help="Branch ID for DAG trace")
    run.add_argument("--category", type=str, help="Factor category for trace/KB")
    run.add_argument("--data-sources", type=str, help="Comma-separated data sources for trace/KB")
    run.add_argument(
        "--feedback-format", type=str, default="layered",
        choices=["flat", "layered", "relevant"],
        help="Feedback output format: flat (legacy), layered (default), or relevant (RC-optimized)",
    )
    run.add_argument(
        "--auto-kb-update", action="store_true",
        help="Automatically update KB after run completes",
    )
    run.set_defaults(func=cmd_run)

    # ------------------------------------------------------------------
    # trace-append subcommand
    # ------------------------------------------------------------------
    trace_append = sub.add_parser(
        "trace-append", help="Append a trace record from result.json to trace.jsonl"
    )
    trace_append.add_argument("--run-dir", required=True, help="Run directory containing trace.jsonl")
    trace_append.add_argument("--result", required=True, help="Path to result.json")
    trace_append.add_argument("--rc-output", help="Optional path to RC diagnosis JSON")
    trace_append.add_argument("--round", type=int, help="Round number")
    trace_append.add_argument("--parent-round", type=int, help="Parent round ID")
    trace_append.add_argument("--branch-id", type=str, default="main", help="Branch ID")
    trace_append.add_argument("--code-summary", type=str, default="", help="Short formula description")
    trace_append.add_argument("--tried-params", type=str, help="JSON string of tried params")
    trace_append.add_argument("--category", type=str, help="Factor category")
    trace_append.add_argument("--data-sources", type=str, help="Comma-separated data sources")
    trace_append.set_defaults(func=cmd_trace_append)

    # ------------------------------------------------------------------
    # kb-update subcommand
    # ------------------------------------------------------------------
    kb_update = sub.add_parser(
        "kb-update", help="Update knowledge base from a result.json"
    )
    kb_update.add_argument("--result", required=True, help="Path to result.json")
    kb_update.add_argument(
        "--status", required=True, choices=["pass", "fail"],
        help="Final status of the factor run",
    )
    kb_update.add_argument("--rc-output", help="Optional path to RC diagnosis JSON (for anti-pattern extraction)")
    kb_update.set_defaults(func=cmd_kb_update)

    # ------------------------------------------------------------------
    # sweep subcommand
    # ------------------------------------------------------------------
    sweep = sub.add_parser(
        "sweep", help="Parallel strategy-parameter sweep for one factor"
    )
    sweep.add_argument("factor_id", help="Factor id to sweep, e.g. f_auto_run001_001")
    sweep.add_argument("--factor-file", required=True, help="Path to the factor code file")
    sweep.add_argument(
        "--generated-dir", default="alphas/exp/agent",
        help="Directory where generated factor modules live",
    )
    sweep.add_argument(
        "--results-root", default="results",
        help="Root directory for pipeline results",
    )
    sweep.add_argument(
        "--top-k", required=True,
        help="Comma-separated top_k values, e.g. 100,200",
    )
    sweep.add_argument(
        "--decay", default="",
        help="Comma-separated decay values, e.g. 5,10,15. Omit for fundamental-factor sweeps.",
    )
    sweep.add_argument(
        "--rebalance", required=True,
        help="Comma-separated rebalance frequencies, e.g. 1D,5D",
    )
    sweep.add_argument(
        "--full", action="store_true",
        help="Run full pipeline (step5-10) for each combo; default is quick mode (step5-6)",
    )
    sweep.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default 4)",
    )
    sweep.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
