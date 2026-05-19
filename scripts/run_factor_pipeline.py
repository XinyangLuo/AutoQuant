#!/usr/bin/env python3
"""Run the full factor screening pipeline for ONE factor.

Three (or four with --decile) stages of evaluation, output layout:

    results/<factor_id>/<variant>/
        factor_eval/        # variant-scoped, shared across runs (tag-agnostic)
        decile_backtest/    # variant-scoped, only when --decile
        <tag>/              # 默认 top{n|pct}_{rebalance}_d{decay}
            pipeline.json
            simple/         # vectorised backtest on adjusted prices, no costs
            detailed/       # event-driven backtest with commission, dividends, etc.

After this script finishes, look at the three reports and run
``python -m backtest.factor.admission admit <factor_id> --variant <variant> --tag <tag>``
to promote, or ``reject ...`` to discard. The admission CLI auto-reads
``pipeline.json`` and stamps the strategy config into the registry history.

Usage:
    python scripts/run_factor_pipeline.py f_rev_05 \\
        --variant swl2_capq5 \\
        --start 20160101 --end 20251231 \\
        --top-n 50 --rebalance 1W --decay 5 \\
        --direction asc --benchmark 000300.SH

    # 分位选股 + 指数成分股 universe
    python scripts/run_factor_pipeline.py f_rev_05 \\
        --top-pct 0.1 --index-members 000300.SH

    # Skip detailed backtest (factor research mode)
    python scripts/run_factor_pipeline.py f_rev_05 --skip-detailed

    # Include decile-layered backtest in Stage 1
    python scripts/run_factor_pipeline.py f_rev_05 --decile

The CLI surface is intentionally narrow — for anything more elaborate
build your own driver using the same building blocks (StrategyConfig +
SingleFactorStrategy + SimpleSimulator / DetailedSimulator + evaluate()).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.evaluation import evaluate as bt_evaluate, render_table
from backtest.evaluation.report import _fmt
from backtest.factor import (
    RECOMMENDED_THRESHOLDS,
    check_recommended_thresholds,
    evaluate as factor_evaluate,
    print_evaluation,
)
from backtest.factor.evaluation import plot_evaluation
from backtest.simulation import (
    BacktestResult,
    DetailedSimulator,
    SimpleSimulator,
    SimulationConfig,
)
from backtest.factor.variants import BASELINE_VARIANT
from backtest.strategy import (
    BacktestConfig,
    FactorConfig,
    NeutralizeConfig,
    SelectionConfig,
    SingleFactorStrategy,
    StrategyConfig,
    UniverseConfig,
    WeightingConfig,
)


# Buffer past `end` so forward-looking simulators can resolve T+h prices.
MARKET_END_BUFFER_DAYS = 10


def _market_end(end: str) -> str:
    """End date padded with MARKET_END_BUFFER_DAYS calendar days."""
    return (pd.to_datetime(end) + pd.Timedelta(days=MARKET_END_BUFFER_DAYS)).strftime("%Y%m%d")


def _strategy_metadata(args) -> dict:
    """The 'strategy' block of the metadata.json — shared by simple/detailed."""
    return {
        "name": f"{args.factor_id}_{args.variant}_{_selection_tag(args)}_{args.rebalance.lower()}",
        "factor": args.factor_id,
        **_strategy_config_dict(args),
    }


def _strategy_config_dict(args) -> dict:
    """Strategy knobs that admission.json / pipeline.json need to record verbatim.

    Single source of truth — used by both ``_strategy_metadata`` (per-run
    metadata) and ``write_pipeline_summary`` (top-level pipeline.json that
    admission reads).
    """
    return {
        "variant": args.variant,
        "top_n": args.top_n,
        "top_pct": args.top_pct,
        "rebalance": args.rebalance,
        "direction": args.direction,
        "decay": args.decay,
        "market_cap_neutral": args.market_cap_neutral,
        "min_market_cap": args.min_market_cap,
        "min_avg_amount": args.min_avg_amount,
        "index_members": args.index_members,
        "benchmark": args.benchmark,
    }


def _selection_tag(args) -> str:
    """``top50`` / ``top10pct`` — selection portion of the run tag."""
    if args.top_pct is not None:
        return f"top{int(round(args.top_pct * 100))}pct"
    return f"top{args.top_n}"


def _build_tag(args) -> str:
    """run tag = top{n|pct%}_{rebalance}_d{decay or 0},可被 --tag 覆盖。"""
    if args.tag:
        return args.tag
    return f"{_selection_tag(args)}_{args.rebalance.lower()}_d{args.decay or 0}"


def _stage_label(stage: int, args) -> str:
    """Stage label: [1/3] or [1/4] depending on --decile."""
    total = 4 if args.decile else 3
    return f"[{stage}/{total}]"


# ---------------------------------------------------------------------------
# Stage 1 — factor-level offline evaluation
# ---------------------------------------------------------------------------


def stage_factor_eval(args, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"{_stage_label(1, args)} Factor evaluation: {args.factor_id}")
    print("=" * 70)

    horizons = [int(h) for h in args.horizons.split(",")]

    result = factor_evaluate(
        args.factor_id,
        args.start, args.end,
        variant=args.variant,
        horizons=horizons,
        ret_type=args.ret_type,
        corr_top_k=5,
        exclude_limit_up=True,
        run_decile_backtest=args.decile,
    )
    print_evaluation(result)

    plot_path = out_dir / f"{args.factor_id}_{args.plot_horizon}d.png"
    plot_evaluation(result, horizon=args.plot_horizon, output_path=str(plot_path))
    print(f"  saved: {plot_path}")

    # Decile plot
    if args.decile and result.decile_result is not None:
        from backtest.simulation.decile import plot_decile_backtest
        decile_dir = out_dir.parent / "decile_backtest"
        decile_dir.mkdir(parents=True, exist_ok=True)
        decile_plot_path = decile_dir / f"{args.factor_id}_{args.variant}_decile.png"
        plot_decile_backtest(result.decile_result, output_path=str(decile_plot_path))
        print(f"  saved: {decile_plot_path}")

    metrics = result.threshold_metrics(args.plot_horizon)
    checks = check_recommended_thresholds(metrics)

    summary = {
        "factor_id": result.factor_id,
        "start": result.start,
        "end": result.end,
        "ret_type": result.ret_type,
        "horizons": horizons,
        "primary_horizon": args.plot_horizon,
        "metrics_by_horizon": result.summary().to_dict(orient="records"),
        "threshold_metrics": metrics,
        "threshold_checks": checks,
        "max_corr": result.max_corr(),
    }
    if args.decile and result.decile_result is not None:
        dr = result.decile_result
        summary["decile"] = {
            "monotonicity_score": dr.monotonicity_score,
            "ls_annual_return": dr.ls_metrics.get("annual_return"),
            "ls_sharpe": dr.ls_metrics.get("sharpe"),
            "ls_max_drawdown": dr.ls_metrics.get("max_drawdown"),
            "d1_annual_return": dr.decile_metrics.get(0, {}).get("annual_return"),
            "d10_annual_return": dr.decile_metrics.get(9, {}).get("annual_return"),
        }

    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"  saved: {out_dir / 'eval_summary.json'}\n")
    return summary


# ---------------------------------------------------------------------------
# Stages 2 & 3 — backtest runners (shared helper)
# ---------------------------------------------------------------------------


def _build_strategy_config(args) -> StrategyConfig:
    return StrategyConfig(
        name=_strategy_metadata(args)["name"],
        strategy_type="single_factor_topk",
        rebalance_freq=args.rebalance,
        delay=1,
        universe=UniverseConfig(
            exclude_st=True,
            exclude_new_ipo_days=252,
            include_kcb=False,
            index_members=args.index_members,
            min_market_cap=args.min_market_cap,
            min_avg_amount=args.min_avg_amount,
        ),
        factors=[FactorConfig(id=args.factor_id, variant=args.variant, direction=args.direction)],
        selection=SelectionConfig(
            method="topk",
            top_k=args.top_n,
            top_pct=args.top_pct,
        ),
        weighting=WeightingConfig(method="equal"),
        neutralize=NeutralizeConfig(market_cap=args.market_cap_neutral),
        decay=args.decay,
        backtest=BacktestConfig(
            start_date=args.start, end_date=args.end, benchmark=args.benchmark,
        ),
    )


def _build_signals(config: StrategyConfig) -> pd.DataFrame:
    strategy = SingleFactorStrategy(config)
    signals = strategy.run(config.backtest.start_date, config.backtest.end_date)
    if signals.empty:
        raise RuntimeError(
            "Strategy produced no signals. Check that factor data is present "
            "in the work DB (run `python -m backtest.factor.backfill <fid>`)."
        )
    return signals


def _run_simulation(
    args,
    *,
    label: str,
    sim,
    sim_run_args: tuple,
    sim_metadata: dict,
    out_dir: Path,
) -> dict:
    """Run one simulator, persist, evaluate, return the flat metrics dict.

    ``label`` is "[2/3] Simple backtest" or "[3/3] Detailed backtest" (or [2/4] / [3/4] when --decile).
    ``sim_run_args`` is the *positional* arg tuple passed to ``sim.run(...)``
    (Simple = ``(signals, market_data)``; Detailed = ``(signals, market_data, dividends)``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"{label}: {args.factor_id}")
    print("=" * 70)

    result: BacktestResult = sim.run(*sim_run_args)
    result.save(str(out_dir), metadata={
        "strategy": _strategy_metadata(args),
        "simulation": sim_metadata,
        "period": {"start_date": args.start, "end_date": args.end},
    })

    report = bt_evaluate(out_dir, benchmark=args.benchmark, plot=True)
    print(render_table(report))
    print(f"  saved: {out_dir / 'report.png'}\n")
    return report.metrics


def stage_simple_backtest(args, signals: pd.DataFrame, market_data: pd.DataFrame,
                          out_dir: Path) -> dict:
    return _run_simulation(
        args,
        label=f"{_stage_label(2, args)} Simple backtest",
        sim=SimpleSimulator(SimulationConfig(initial_cash=args.initial_cash)),
        sim_run_args=(signals, market_data),
        sim_metadata={"engine": "SimpleSimulator", "initial_cash": args.initial_cash},
        out_dir=out_dir,
    )


def stage_detailed_backtest(args, signals: pd.DataFrame, market_data: pd.DataFrame,
                            dividends: pd.DataFrame, out_dir: Path) -> dict:
    return _run_simulation(
        args,
        label=f"{_stage_label(3, args)} Detailed backtest",
        sim=DetailedSimulator(SimulationConfig(
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            price_type=args.price_type,
            allow_short=False,
        )),
        sim_run_args=(signals, market_data, dividends),
        sim_metadata={
            "engine": "DetailedSimulator",
            "initial_cash": args.initial_cash,
            "commission_rate": args.commission_rate,
            "price_type": args.price_type,
        },
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Final pipeline summary
# ---------------------------------------------------------------------------


_KEY_METRICS = [
    "annual_return", "annual_volatility", "sharpe", "sortino",
    "max_drawdown", "calmar", "daily_win_rate", "monthly_win_rate",
    "avg_daily_turnover", "annual_turnover", "fees_pct_of_initial",
    "information_ratio", "annual_excess_return",
]


def _pick_key_metrics(metrics: dict | None) -> dict:
    if not metrics:
        return {}
    return {k: metrics.get(k) for k in _KEY_METRICS if k in metrics}


def write_pipeline_summary(args, root: Path, eval_summary: dict,
                           simple_metrics: dict, detailed_metrics: dict | None) -> None:
    payload = {
        "factor_id": args.factor_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "period": {"start": args.start, "end": args.end},
        "strategy_config": {**_strategy_config_dict(args), "tag": _build_tag(args)},
        "factor_eval": {
            "threshold_metrics": eval_summary["threshold_metrics"],
            "threshold_checks": eval_summary["threshold_checks"],
        },
        "decile_backtest": eval_summary.get("decile"),
        "simple_backtest": _pick_key_metrics(simple_metrics),
        "detailed_backtest": _pick_key_metrics(detailed_metrics) if detailed_metrics else None,
        "recommended_thresholds": RECOMMENDED_THRESHOLDS,
    }
    with open(root / "pipeline.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  saved: {root / 'pipeline.json'}")


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build markdown table lines from headers and rows."""
    lines = [f"| {' | '.join(headers)} |"]
    lines.append(f"|{'|'.join('---' for _ in headers)}|")
    for row in rows:
        lines.append(f"| {' | '.join(str(c) for c in row)} |")
    return lines


def write_pipeline_markdown(
    args,
    root: Path,
    eval_summary: dict,
    simple_metrics: dict,
    detailed_metrics: dict | None,
) -> None:
    """Generate a comprehensive Markdown report from all pipeline stages.

    The report combines factor evaluation, simple/detailed backtest metrics,
    and decile backtest results into a single human-readable document for
    decision-making.
    """
    fid = eval_summary["factor_id"]
    tag = _build_tag(args)
    period = f"{args.start} ~ {args.end}"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# Factor Pipeline Report: `{fid}`")
    lines.append("")
    lines.append(f"- **Generated**: {now}")
    lines.append(f"- **Period**: {period}")
    lines.append(f"- **Variant**: `{args.variant}`")
    lines.append(f"- **Tag**: `{tag}`")
    lines.append("")

    # ── 1. Strategy Configuration ───────────────────────────────────────
    sel = f"top{args.top_n}" if args.top_n else f"top{int(args.top_pct * 100)}pct"
    cfg_rows = [
        ["Universe selection", sel],
        ["Rebalance", args.rebalance],
        ["Direction", args.direction],
        ["Decay", str(args.decay or 0)],
        ["Market-cap neutral", "Yes" if args.market_cap_neutral else "No"],
        ["Min market cap", f"{args.min_market_cap:,.0f}"],
        ["Min avg amount", f"{args.min_avg_amount:,.0f}"],
        ["Benchmark", args.benchmark],
    ]
    if args.index_members:
        cfg_rows.append(["Index members", args.index_members])
    lines.append("## 1. Strategy Configuration")
    lines.append("")
    lines.extend(_md_table(["Parameter", "Value"], cfg_rows))
    lines.append("")

    # ── 2. Factor Static Evaluation ─────────────────────────────────────
    lines.append("## 2. Factor Static Evaluation")
    lines.append("")

    # IC / RankIC table
    metrics_by_h = eval_summary.get("metrics_by_horizon", [])
    if metrics_by_h:
        ic_headers = ["Horizon", "IC_mean", "IC_std", "ICIR", "IC_tstat",
                      "RankIC_mean", "RankIC_std", "RankICIR", "RankIC_tstat"]
        ic_rows = []
        for row in metrics_by_h:
            h = row.get("horizon", "")
            ic_rows.append([
                str(h),
                _fmt(row.get("IC_mean"), "f4"), _fmt(row.get("IC_std"), "f4"),
                _fmt(row.get("ICIR"), "f4"), _fmt(row.get("IC_tstat"), "f4"),
                _fmt(row.get("RankIC_mean"), "f4"), _fmt(row.get("RankIC_std"), "f4"),
                _fmt(row.get("RankICIR"), "f4"), _fmt(row.get("RankIC_tstat"), "f4"),
            ])
        lines.append("### 2.1 IC / RankIC by Horizon")
        lines.append("")
        lines.extend(_md_table(ic_headers, ic_rows))
        lines.append("")

    # Threshold checks
    checks = eval_summary.get("threshold_checks", {})
    tm = eval_summary.get("threshold_metrics", {})
    n_pass = sum(checks.values())
    thr_rows = [
        ["RankICIR", _fmt(tm.get("rankicir"), "f4"),
         f">= {RECOMMENDED_THRESHOLDS['min_rankicir']}", "PASS" if checks.get("rankicir") else "FAIL"],
        ["IC+ ratio", _fmt(tm.get("ic_positive_ratio"), "pct"),
         f">= {RECOMMENDED_THRESHOLDS['min_ic_positive_ratio']:.0%}", "PASS" if checks.get("ic_positive_ratio") else "FAIL"],
        ["Turnover", _fmt(tm.get("turnover"), "f4"),
         f"< {RECOMMENDED_THRESHOLDS['max_turnover']}", "PASS" if checks.get("turnover") else "FAIL"],
        ["Max |corr|", _fmt(tm.get("max_corr"), "f4"),
         f"< {RECOMMENDED_THRESHOLDS['max_corr']}", "PASS" if checks.get("max_corr") else "FAIL"],
    ]
    lines.append("### 2.2 Admission Threshold Checks")
    lines.append("")
    lines.extend(_md_table(["Check", "Value", "Threshold", "Pass"], thr_rows))
    lines.append("")
    if n_pass == 4:
        lines.append("> All 4 reference thresholds **PASSED**. Proceed to backtest review.")
    else:
        lines.append(f"> **{4 - n_pass} threshold(s) FAILED**. Consider tuning or rejecting.")
    lines.append("")

    # Turnover & max corr
    lines.append("### 2.3 Other Metrics")
    lines.append("")
    lines.append(f"- **Turnover**: {tm.get('turnover', 'N/A')}")
    max_corr = eval_summary.get("max_corr")
    if max_corr:
        lines.append(f"- **Max correlation with existing factors**: {max_corr[0]} → {_fmt(max_corr[1], 'f4')}")
    else:
        lines.append("- **Max correlation with existing factors**: N/A (no existing factors)")
    lines.append("")

    # Decile backtest
    decile = eval_summary.get("decile")
    if decile:
        lines.append("### 2.4 Decile Backtest")
        lines.append("")
        lines.append(f"- **Monotonicity score**: {decile.get('monotonicity_score', 'N/A')}")
        lines.append(f"- **Long-Short annual return**: {_fmt(decile.get('ls_annual_return'), 'pct')}")
        lines.append(f"- **Long-Short Sharpe**: {_fmt(decile.get('ls_sharpe'), 'f3')}")
        lines.append(f"- **Long-Short max drawdown**: {_fmt(decile.get('ls_max_drawdown'), 'pct')}")
        d1 = decile.get("d1_annual_return")
        d10 = decile.get("d10_annual_return")
        if d1 is not None and d10 is not None:
            lines.append(f"- **D1 annual return**: {_fmt(d1, 'pct')}")
            lines.append(f"- **D10 annual return**: {_fmt(d10, 'pct')}")
            lines.append(f"- **D10 - D1 spread**: {_fmt(d10 - d1, 'pct')}")
        lines.append("")

    # ── 3. Simple Backtest ──────────────────────────────────────────────
    lines.append("## 3. Simple Backtest (Vectorised, No Costs)")
    lines.append("")
    lines.extend(_md_metrics_section(simple_metrics))

    # ── 4. Detailed Backtest ────────────────────────────────────────────
    if detailed_metrics:
        lines.append("## 4. Detailed Backtest (Event-Driven, With Costs)")
        lines.append("")
        lines.extend(_md_metrics_section(detailed_metrics))

        # Cost drag
        simple_ann = simple_metrics.get("annual_return", 0) or 0
        detailed_ann = detailed_metrics.get("annual_return", 0) or 0
        drag = simple_ann - detailed_ann
        lines.append("### 4.1 Cost Drag")
        lines.append("")
        lines.append(f"- **Simple annual return**: {_fmt(simple_ann, 'pct')}")
        lines.append(f"- **Detailed annual return**: {_fmt(detailed_ann, 'pct')}")
        lines.append(f"- **Cost drag (simple - detailed)**: {_fmt(drag, 'pct')}")
        lines.append("")

    # ── 5. Decision Summary ─────────────────────────────────────────────
    ds_rows = [
        ["Factor eval", "Thresholds passed", f"{n_pass}/4"],
    ]
    if decile:
        ds_rows.append(
            ["Decile", "Monotonicity", _fmt(decile.get("monotonicity_score"), "f3")]
        )
    ds_rows.append(
        ["Simple BT", "Sharpe / MDD",
         f"{_fmt(simple_metrics.get('sharpe'), 'f3')} / {_fmt(simple_metrics.get('max_drawdown'), 'pct')}"]
    )
    if detailed_metrics:
        ds_rows.append(
            ["Detailed BT", "Sharpe / MDD",
             f"{_fmt(detailed_metrics.get('sharpe'), 'f3')} / {_fmt(detailed_metrics.get('max_drawdown'), 'pct')}"]
        )
    lines.append("## 5. Decision Summary")
    lines.append("")
    lines.extend(_md_table(["Stage", "Metric", "Value"], ds_rows))
    lines.append("")

    # Next steps
    lines.append("## 6. Next Steps")
    lines.append("")
    lines.append("```bash")
    lines.append(f"# Admit this factor to the library")
    lines.append(f"python -m backtest.factor.admission admit {fid} --variant {args.variant} --tag {tag}")
    lines.append("")
    lines.append(f"# Or reject it")
    lines.append(f"python -m backtest.factor.admission reject {fid} --variant {args.variant} --tag {tag}")
    lines.append("```")
    lines.append("")

    md_path = root / "pipeline_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved: {md_path}")


def _md_metrics_section(metrics: dict) -> list[str]:
    """Render a subset of key metrics as markdown lines."""
    if not metrics:
        return ["*No metrics available.*", ""]

    lines: list[str] = []
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    _rows: list[tuple[str, str, str]] = [
        ("Total Return", "total_return", "pct"),
        ("Annualised Return", "annual_return", "pct"),
        ("Annualised Volatility", "annual_volatility", "pct"),
        ("Sharpe", "sharpe", "f3"),
        ("Sortino", "sortino", "f3"),
        ("Calmar", "calmar", "f3"),
        ("Max Drawdown", "max_drawdown", "pct"),
        ("Daily Win Rate", "daily_win_rate", "pct"),
        ("Monthly Win Rate", "monthly_win_rate", "pct"),
        ("Information Ratio", "information_ratio", "f3"),
        ("Annual Excess Return", "annual_excess_return", "pct"),
        ("Avg Daily Turnover", "avg_daily_turnover", "pct"),
        ("Annual Turnover", "annual_turnover", "f2"),
        ("Total Trades", "total_trades", "int"),
        ("Fees % of Initial", "fees_pct_of_initial", "pct"),
    ]
    for label, key, kind in _rows:
        v = metrics.get(key)
        if v is not None:
            lines.append(f"| {label} | {_fmt(v, kind)} |")
    lines.append("")
    return lines


def print_decision_hint(args, eval_summary: dict, simple: dict, detailed: dict | None) -> None:
    print("=" * 70)
    print("Decision summary")
    print("=" * 70)
    checks = eval_summary["threshold_checks"]
    n_pass = sum(checks.values())
    print(f"  Factor thresholds passed : {n_pass}/4  ({checks})")
    decile = eval_summary.get("decile")
    if decile:
        print(f"  Decile monotonicity      : {decile.get('monotonicity_score', float('nan')):+.3f}")
        print(f"  Decile LS ann_ret / sharpe: "
              f"{decile.get('ls_annual_return', 0) or 0:+.2%} / {decile.get('ls_sharpe', float('nan')):.2f}")
    print(f"  Simple   Sharpe / MDD    : "
          f"{simple.get('sharpe'):.2f} / {simple.get('max_drawdown'):.2%}")
    if detailed:
        print(f"  Detailed Sharpe / MDD    : "
              f"{detailed.get('sharpe'):.2f} / {detailed.get('max_drawdown'):.2%}")
        gap = (simple.get('annual_return', 0) or 0) - (detailed.get('annual_return', 0) or 0)
        print(f"  Cost drag (simple - det) : {gap:+.2%}")
    print()
    tag = _build_tag(args)
    fid = eval_summary['factor_id']
    print("Next step:")
    print(f"  python -m backtest.factor.admission admit  {fid} "
          f"--variant {args.variant} --tag {tag}")
    print(f"  python -m backtest.factor.admission reject {fid} "
          f"--variant {args.variant} --tag {tag}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the factor screening pipeline")
    p.add_argument("factor_id")
    p.add_argument("--variant", default=BASELINE_VARIANT,
                   help=f"Neutralization variant (default: {BASELINE_VARIANT}). "
                        "Common: 'raw' (no neutralization), 'swl2_capq5' (SW-L2 + circ_mv quintile).")
    p.add_argument("--start", default="20160101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--horizons", default="1,5,10,20,60")
    p.add_argument("--ret-type", default="open", choices=["close", "open"])
    p.add_argument("--plot-horizon", type=int, default=20)

    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--top-n", type=int, default=None,
                     help="选股绝对数量;与 --top-pct 互斥;两者均未传则默认 50。")
    sel.add_argument("--top-pct", type=float, default=None,
                     help="选股分位数 (0, 1],如 0.1 表示前 10%%。")
    p.add_argument("--rebalance", default="1W",
                   choices=["1D", "5D", "1W", "2W", "1M", "EOM"])
    p.add_argument("--direction", default="desc", choices=["desc", "asc"])
    p.add_argument("--decay", type=int, default=5,
                   help="Linear decay window; pass 0 to disable")
    p.add_argument("--no-cap-neutral", action="store_true",
                   help="Disable market-cap neutralisation")
    p.add_argument("--min-market-cap", type=float, default=5e8)
    p.add_argument("--min-avg-amount", type=float, default=1e7)
    p.add_argument("--index-members", default=None,
                   help="限制 universe 到指定指数成分股,如 000300.SH。"
                        "需先跑 backfill_index_members 准备数据。")

    p.add_argument("--initial-cash", type=float, default=1e8)
    p.add_argument("--commission-rate", type=float, default=0.0003)
    p.add_argument("--price-type", default="o2o", choices=["o2o", "c2c"])
    p.add_argument("--benchmark", default="000300.SH")

    p.add_argument("--skip-detailed", action="store_true",
                   help="Skip the detailed backtest (research mode)")
    p.add_argument("--decile", action="store_true",
                   help="Run decile-layered backtest (10 equal-weight groups by factor value)")
    p.add_argument("--results-root", default="results",
                   help="Root directory for all stage outputs")
    p.add_argument("--tag", default=None,
                   help="覆盖 results/<fid>/<variant>/<tag>/ 中的 tag 段,"
                        "默认 top{n|pct}_{rebalance}_d{decay}。")
    return p


def main():
    args = _build_parser().parse_args()
    args.market_cap_neutral = not args.no_cap_neutral
    if args.decay == 0:
        args.decay = None
    # 默认行为:两者都没传 → top_n=50,保持旧脚本兼容。
    if args.top_n is None and args.top_pct is None:
        args.top_n = 50

    variant_root = Path(args.results_root) / args.factor_id / args.variant
    tag = _build_tag(args)
    run_root = variant_root / tag
    run_root.mkdir(parents=True, exist_ok=True)

    eval_summary = stage_factor_eval(args, variant_root / "factor_eval")

    config = _build_strategy_config(args)
    print(f"\nGenerating signals ({config.name}) ...")
    signals = _build_signals(config)
    print(f"  signals: {len(signals):,} rows over {signals['date'].nunique()} dates")

    # Load market panel + dividends once, share across simple / detailed stages.
    market_end = _market_end(args.end)
    symbols = signals["symbol"].unique().tolist()
    with MarketStorage() as ms:
        market_data = ms.get_bars(symbols=symbols, start=args.start, end=market_end)
        dividends = (
            ms.get_dividends(symbols=symbols, start=args.start, end=market_end)
            if not args.skip_detailed else None
        )

    simple_metrics = stage_simple_backtest(args, signals, market_data, run_root / "simple")

    detailed_metrics = None
    if not args.skip_detailed:
        detailed_metrics = stage_detailed_backtest(
            args, signals, market_data, dividends, run_root / "detailed",
        )

    write_pipeline_summary(args, run_root, eval_summary, simple_metrics, detailed_metrics)
    write_pipeline_markdown(args, run_root, eval_summary, simple_metrics, detailed_metrics)
    print_decision_hint(args, eval_summary, simple_metrics, detailed_metrics)


if __name__ == "__main__":
    main()
