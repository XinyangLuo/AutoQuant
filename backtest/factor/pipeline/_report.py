"""Markdown report generation from PipelineState."""

from __future__ import annotations

from pathlib import Path

from backtest.evaluation.report import _fmt

from .state import PipelineState


def generate_pipeline_report(state: PipelineState) -> Path:
    """Generate a comprehensive markdown report from pipeline state.

    Returns the path to the written report file.
    """
    config = state.config
    factor_id = config.factor_id
    results_dir = Path(config.results_root) / factor_id
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "pipeline_report.md"

    lines: list[str] = [
        f"# Factor Pipeline Report: `{factor_id}`",
        "",
        f"- **Period**: {config.start_date} ~ {config.end_date}",
        f"- **Frequency**: {config.frequency}",
        f"- **Status**: {state.status}",
        f"- **Retry count**: {state.retry_count}",
        "",
    ]

    # Step-by-step summary table
    lines.extend(_step_summary_table(state))

    # Factor evaluation
    lines.extend(_factor_eval_section(state))

    # Backtest results
    lines.extend(_backtest_section(state))

    # Ridge R2
    lines.extend(_ridge_section(state))

    # Decision
    lines.extend(_decision_section(state))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _step_summary_table(state: PipelineState) -> list[str]:
    lines = ["## Step Results Summary", ""]
    lines.append("| Step | Name | Result | Key Metrics | Reason |")
    lines.append("|------|------|--------|-------------|--------|")

    step_names = {
        "step1": "Coverage",
        "step2": "Neutralization",
        "step3": "ICIR",
        "step4": "Monotonicity",
        "step5": "Strategy Config",
        "step6": "Simple Backtest",
        "step7": "Detailed Backtest",
        "step8": "Ridge R2",
        "step9": "Admission",
    }

    for step_key in step_names:
        result = state.step_results.get(step_key)
        if result is None:
            continue
        status = "PASS" if result.passed else "FAIL"
        name = step_names[step_key]
        metrics_str = _summarise_metrics(result.metrics)
        reason = result.reason or "-"
        lines.append(f"| {step_key} | {name} | {status} | {metrics_str} | {reason} |")

    lines.append("")
    return lines


def _summarise_metrics(metrics: dict) -> str:
    """Pick 1-2 most important metrics for the summary table."""
    if not metrics:
        return "-"
    # Map step -> key metric
    priority_keys = [
        "max_missing_rate", "size_corr", "annual_icir", "spearman",
        "sharpe", "max_drawdown", "r2", "tier",
    ]
    for key in priority_keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                return f"{key}={val:.3f}"
            return f"{key}={val}"
    return str(list(metrics.keys())[:2])


def _factor_eval_section(state: PipelineState) -> list[str]:
    lines = ["## Factor Evaluation", ""]

    step3 = state.step_results.get("step3")
    if step3 is None:
        lines.append("*No evaluation data.*")
        lines.append("")
        return lines

    all_ic = step3.metrics.get("all_ic_metrics", {})
    if all_ic:
        lines.append("### IC / RankIC by Horizon")
        lines.append("")
        lines.append("| Horizon | IC Mean | IC Std | ICIR | IC t-stat | IC+ Ratio |")
        lines.append("|---------|---------|--------|------|-----------|-----------|")
        for h_str, ic in sorted(all_ic.items(), key=lambda x: int(x[0])):
            lines.append(
                f"| {h_str}d | {_fmt(ic.get('ic_mean'), 'f4')} | "
                f"{_fmt(ic.get('ic_std'), 'f4')} | {_fmt(ic.get('icir'), 'f4')} | "
                f"{_fmt(ic.get('ic_tstat'), 'f4')} | {_fmt(ic.get('ic_positive_ratio'), 'pct')} |"
            )
        lines.append("")

    # Best horizon detail
    best_h = step3.metrics.get("best_horizon")
    if best_h:
        lines.append(f"### Best Horizon: {best_h}d")
        lines.append("")
        lines.append(f"- **Annual ICIR**: {_fmt(step3.metrics.get('annual_icir'), 'f4')}")
        lines.append(f"- **|IC|**: {_fmt(step3.metrics.get('abs_ic'), 'f4')}")
        lines.append(f"- **t-stat**: {_fmt(step3.metrics.get('tstat'), 'f4')}")
        lines.append(f"- **Positive ratio**: {_fmt(step3.metrics.get('pos_ratio'), 'pct')}")
        lines.append("")

    # Monotonicity
    step4 = state.step_results.get("step4")
    if step4:
        lines.append("### Monotonicity")
        lines.append("")
        lines.append(f"- **Spearman**: {_fmt(step4.metrics.get('spearman'), 'f4')}")
        lines.append(f"- **Threshold**: {state.config.thresholds.min_monotonicity}")
        lines.append("")

    return lines


def _backtest_section(state: PipelineState) -> list[str]:
    lines = ["## Backtest Results", ""]

    # Simple backtest
    step6 = state.step_results.get("step6")
    if step6 and step6.metrics:
        lines.append("### Simple Backtest (Vectorised, No Costs)")
        lines.append("")
        lines.extend(_metrics_table(step6.metrics))
        lines.append("")

    # Detailed backtest
    step7 = state.step_results.get("step7")
    if step7 and step7.metrics:
        lines.append("### Detailed Backtest (Event-Driven, With Costs)")
        lines.append("")
        lines.extend(_metrics_table(step7.metrics))
        lines.append("")

        # Cost drag
        simple_ann = (step6.metrics or {}).get("annual_return", 0) or 0
        detailed_ann = step7.metrics.get("annual_return", 0) or 0
        drag = simple_ann - detailed_ann
        lines.append("### Cost Drag")
        lines.append("")
        lines.append(f"- **Simple annual return**: {_fmt(simple_ann, 'pct')}")
        lines.append(f"- **Detailed annual return**: {_fmt(detailed_ann, 'pct')}")
        lines.append(f"- **Drag**: {_fmt(drag, 'pct')}")
        lines.append("")

    return lines


def _metrics_table(metrics: dict) -> list[str]:
    rows = [
        ("Annual Return", "annual_return", "pct"),
        ("Sharpe", "sharpe", "f3"),
        ("Max Drawdown", "max_drawdown", "pct"),
        ("Calmar", "calmar", "f3"),
        ("Annual Turnover", "annual_turnover", "f2"),
        ("Total Trades", "total_trades", "int"),
    ]
    lines = ["| Metric | Value |", "|--------|-------|"]
    for label, key, kind in rows:
        val = metrics.get(key)
        if val is not None:
            lines.append(f"| {label} | {_fmt(val, kind)} |")
    return lines


def _ridge_section(state: PipelineState) -> list[str]:
    lines = ["## Ridge R2 Classification", ""]

    step8 = state.step_results.get("step8")
    if step8 is None:
        lines.append("*No Ridge check data.*")
        lines.append("")
        return lines

    lines.append(f"- **R2**: {_fmt(step8.metrics.get('r2'), 'f4')}")
    lines.append(f"- **Tier**: {step8.metrics.get('tier')}")
    lines.append(f"- **Residual ICIR**: {_fmt(step8.metrics.get('residual_icir'), 'f4')}")
    lines.append(f"- **Observations**: {step8.metrics.get('n_obs')}")
    lines.append("")

    return lines


def _decision_section(state: PipelineState) -> list[str]:
    lines = ["## Decision", ""]

    if state.status == "admitted":
        lines.append("**ADMITTED** — Factor promoted to library.")
    elif state.status == "rejected":
        # Find the rejection reason
        for step_key in ["step1", "step2", "step3", "step4", "step5",
                         "step6", "step7", "step8", "step9"]:
            result = state.step_results.get(step_key)
            if result and not result.passed and result.reason:
                lines.append(f"**REJECTED** at {step_key}: {result.reason}")
                break
    else:
        lines.append(f"**Status**: {state.status}")

    lines.append("")
    return lines
