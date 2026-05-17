"""CLI entry point for backtest evaluation.

Usage:
    python -m backtest.evaluation <result_dir> [--benchmark 000300.SH] [--no-plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest.evaluation.report import evaluate, render_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.evaluation",
        description="Evaluate a saved BacktestResult directory.",
    )
    parser.add_argument("result_dir", help="Path to the directory containing nav.parquet")
    parser.add_argument("--benchmark", "-b", default=None,
                        help="Benchmark index code (e.g. 000300.SH). Default: no benchmark.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting (just compute metrics + write summary files).")
    parser.add_argument("--rf", type=float, default=0.0,
                        help="Annual risk-free rate for Sharpe/Sortino (default 0).")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory for summary.json / summary.csv / report.png. "
                             "Default: same as result_dir.")
    parser.add_argument("--rolling-window", type=int, default=90,
                        help="Rolling Sharpe window length in days (default 90).")
    args = parser.parse_args(argv)

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"ERROR: result_dir does not exist: {result_dir}", file=sys.stderr)
        return 2

    report = evaluate(
        result_dir,
        benchmark=args.benchmark,
        plot=not args.no_plot,
        output_dir=args.output_dir,
        rf=args.rf,
        rolling_sharpe_window=args.rolling_window,
    )

    print(render_table(report))

    out_path = Path(args.output_dir) if args.output_dir else result_dir
    print(f"\nSaved: {out_path / 'summary.json'}")
    print(f"Saved: {out_path / 'summary.csv'}")
    if not args.no_plot:
        print(f"Saved: {out_path / 'report.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
