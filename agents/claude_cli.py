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
from .runner import AutoQuantFactorRunner
from .schema import COLUMN_ALIASES, get_panel_columns_for_data_sources

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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_clean_json(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


def _read_factor_code(
    factor_id: str, factor_file: Path | None, generated_dir: Path,
) -> tuple[str, Path]:
    path = factor_file or generated_dir / f"{factor_id}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Factor file not found: {path}. Pass --factor-file or create "
            f"{generated_dir / (factor_id + '.py')}"
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
    run_dir = Path(args.run_dir)
    generated_dir = Path(args.generated_dir)
    result_path = Path(args.result_path) if args.result_path else run_dir / "result.json"
    cfg = AgentConfig()

    experiment = AutoQuantFactorExperiment(factor_id=factor_id)
    feedback: QuantFeedback | None = None
    error: str | None = None
    tb: str | None = None
    factor_file_path: Path | None = None
    sanitized_code_path: Path | None = None

    try:
        code, factor_file_path = _read_factor_code(
            factor_id,
            Path(args.factor_file) if args.factor_file else None,
            generated_dir,
        )
        validate_python_code(code)
        validate_transforms_imports(code)
        code = force_register_factor_id(code, factor_id)
        sanitized_code_path = run_dir / "factor_sanitized.py"
        sanitized_code_path.parent.mkdir(parents=True, exist_ok=True)
        sanitized_code_path.write_text(code, encoding="utf-8")

        experiment.factor_code = code
        evaluator = AutoQuantFactorEvaluator()

        with AutoQuantFactorRunner(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            results_root=run_dir,
            generated_dir=generated_dir,
        ) as runner:
            try:
                experiment = runner.run(experiment)
                feedback = evaluator.evaluate(experiment)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                experiment.error = error
                if not args.keep_work_db:
                    runner.cleanup_work_db(factor_id)
            else:
                if feedback is not None and not feedback.decision and not args.keep_work_db:
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
    else:
        status = "fail"

    result = {
        "factor_id": factor_id,
        "status": status,
        "failure_type": failure_type,
        "error": error,
        "traceback": tb,
        "factor_file": str(factor_file_path) if factor_file_path else None,
        "sanitized_factor_file": str(sanitized_code_path) if sanitized_code_path else None,
        "result_path": str(result_path),
        "run_dir": str(run_dir),
        "thresholds": _read_pipeline_thresholds(),
        "metrics": feedback.metrics if feedback else {},
        "feedback": feedback.to_dict() if feedback else None,
        "experiment": experiment.to_dict(),
    }

    _write_json(result_path, result)

    if status == "pass":
        _write_candidate(experiment, factor_id, result_path)

    print(json.dumps(_clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if status != "error" else 1


def _write_candidate(
    experiment: AutoQuantFactorExperiment,
    factor_id: str,
    result_path: Path,
) -> None:
    """Write passing factor to ``results/agent/candidates/<factor_id>/`` for human review."""
    candidates_root = Path("results/agent/candidates")
    candidate_dir = candidates_root / factor_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    if experiment.factor_code:
        (candidate_dir / "factor.py").write_text(experiment.factor_code, encoding="utf-8")

    if result_path.exists():
        shutil.copy2(result_path, candidate_dir / "result.json")

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
    run.add_argument("--run-dir", required=True, help="Round output directory")
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
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
