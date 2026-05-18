#!/usr/bin/env python3
"""Run the full factor screening pipeline for ONE factor.

Three stages of evaluation, each written under ``results/<factor_id>/``:

1. ``factor_eval/`` — factor-level offline metrics (IC / RankIC / ICIR /
   turnover / cross-factor correlation) via ``backtest.factor.evaluate``.
2. ``simple/``      — vectorised backtest on adjusted prices, no costs,
   no limit-up filter. Net of the strategy config but optimistic.
3. ``detailed/``    — event-driven backtest with commission, stamp duty,
   limit-up / suspension filtering, board lot sizing, dividend events.

After this script finishes, look at the three reports and run
``python -m backtest.factor.admission admit <factor_id>`` to promote, or
``reject <factor_id>`` to discard.

Usage:
    python scripts/run_factor_pipeline.py f_rev_05 \\
        --start 20210101 --end 20241231 \\
        --top-n 50 --rebalance 1W --decay 5 \\
        --direction asc --benchmark 000300.SH

    # Skip detailed backtest (factor research mode)
    python scripts/run_factor_pipeline.py f_rev_05 --skip-detailed

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

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.evaluation import evaluate as bt_evaluate, render_table
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
        "name": f"{args.factor_id}_top{args.top_n}_{args.rebalance.lower()}",
        "factor": args.factor_id,
        "top_n": args.top_n,
        "rebalance": args.rebalance,
        "direction": args.direction,
        "decay": args.decay,
        "market_cap_neutral": args.market_cap_neutral,
    }


# ---------------------------------------------------------------------------
# Stage 1 — factor-level offline evaluation
# ---------------------------------------------------------------------------


def stage_factor_eval(args, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"[1/3] Factor evaluation: {args.factor_id}")
    print("=" * 70)

    horizons = [int(h) for h in args.horizons.split(",")]

    result = factor_evaluate(
        args.factor_id,
        args.start, args.end,
        horizons=horizons,
        ret_type=args.ret_type,
        corr_top_k=5,
        exclude_limit_up=True,
    )
    print_evaluation(result)

    plot_path = out_dir / f"{args.factor_id}_{args.plot_horizon}d.png"
    plot_evaluation(result, horizon=args.plot_horizon, output_path=str(plot_path))
    print(f"  saved: {plot_path}")

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
            min_market_cap=args.min_market_cap,
            min_avg_amount=args.min_avg_amount,
        ),
        factors=[FactorConfig(id=args.factor_id, direction=args.direction)],
        selection=SelectionConfig(method="topk", top_k=args.top_n),
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

    ``label`` is "[2/3] Simple backtest" or "[3/3] Detailed backtest".
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
        label="[2/3] Simple backtest",
        sim=SimpleSimulator(SimulationConfig(initial_cash=args.initial_cash)),
        sim_run_args=(signals, market_data),
        sim_metadata={"engine": "SimpleSimulator", "initial_cash": args.initial_cash},
        out_dir=out_dir,
    )


def stage_detailed_backtest(args, signals: pd.DataFrame, market_data: pd.DataFrame,
                            dividends: pd.DataFrame, out_dir: Path) -> dict:
    return _run_simulation(
        args,
        label="[3/3] Detailed backtest",
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
        "strategy_config": {
            "top_n": args.top_n,
            "rebalance": args.rebalance,
            "direction": args.direction,
            "decay": args.decay,
            "market_cap_neutral": args.market_cap_neutral,
            "min_market_cap": args.min_market_cap,
            "min_avg_amount": args.min_avg_amount,
            "benchmark": args.benchmark,
        },
        "factor_eval": {
            "threshold_metrics": eval_summary["threshold_metrics"],
            "threshold_checks": eval_summary["threshold_checks"],
        },
        "simple_backtest": _pick_key_metrics(simple_metrics),
        "detailed_backtest": _pick_key_metrics(detailed_metrics) if detailed_metrics else None,
        "recommended_thresholds": RECOMMENDED_THRESHOLDS,
    }
    with open(root / "pipeline.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  saved: {root / 'pipeline.json'}")


def print_decision_hint(eval_summary: dict, simple: dict, detailed: dict | None) -> None:
    print("=" * 70)
    print("Decision summary")
    print("=" * 70)
    checks = eval_summary["threshold_checks"]
    n_pass = sum(checks.values())
    print(f"  Factor thresholds passed : {n_pass}/4  ({checks})")
    print(f"  Simple   Sharpe / MDD    : "
          f"{simple.get('sharpe'):.2f} / {simple.get('max_drawdown'):.2%}")
    if detailed:
        print(f"  Detailed Sharpe / MDD    : "
              f"{detailed.get('sharpe'):.2f} / {detailed.get('max_drawdown'):.2%}")
        gap = (simple.get('annual_return', 0) or 0) - (detailed.get('annual_return', 0) or 0)
        print(f"  Cost drag (simple - det) : {gap:+.2%}")
    print()
    print("Next step:")
    print(f"  python -m backtest.factor.admission admit  {eval_summary['factor_id']}")
    print(f"  python -m backtest.factor.admission reject {eval_summary['factor_id']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the factor screening pipeline")
    p.add_argument("factor_id")
    p.add_argument("--start", default="20210101")
    p.add_argument("--end", default="20241231")
    p.add_argument("--horizons", default="1,5,10,20,60")
    p.add_argument("--ret-type", default="open", choices=["close", "open"])
    p.add_argument("--plot-horizon", type=int, default=20)

    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--rebalance", default="1W",
                   choices=["1D", "1W", "2W", "1M", "EOM"])
    p.add_argument("--direction", default="desc", choices=["desc", "asc"])
    p.add_argument("--decay", type=int, default=5,
                   help="Linear decay window; pass 0 to disable")
    p.add_argument("--no-cap-neutral", action="store_true",
                   help="Disable market-cap neutralisation")
    p.add_argument("--min-market-cap", type=float, default=5e8)
    p.add_argument("--min-avg-amount", type=float, default=1e7)

    p.add_argument("--initial-cash", type=float, default=1e8)
    p.add_argument("--commission-rate", type=float, default=0.0003)
    p.add_argument("--price-type", default="o2o", choices=["o2o", "c2c"])
    p.add_argument("--benchmark", default="000300.SH")

    p.add_argument("--skip-detailed", action="store_true",
                   help="Skip the detailed backtest (research mode)")
    p.add_argument("--results-root", default="results",
                   help="Root directory for all stage outputs")
    return p


def main():
    args = _build_parser().parse_args()
    args.market_cap_neutral = not args.no_cap_neutral
    if args.decay == 0:
        args.decay = None

    root = Path(args.results_root) / args.factor_id
    root.mkdir(parents=True, exist_ok=True)

    eval_summary = stage_factor_eval(args, root / "factor_eval")

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

    simple_metrics = stage_simple_backtest(args, signals, market_data, root / "simple")

    detailed_metrics = None
    if not args.skip_detailed:
        detailed_metrics = stage_detailed_backtest(
            args, signals, market_data, dividends, root / "detailed",
        )

    write_pipeline_summary(args, root, eval_summary, simple_metrics, detailed_metrics)
    print_decision_hint(eval_summary, simple_metrics, detailed_metrics)


if __name__ == "__main__":
    main()
