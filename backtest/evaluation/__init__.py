"""Backtest evaluation: read saved BacktestResult parquet files, compute
metrics, plot, and persist summary files.

Public API
----------
evaluate(result_dir, benchmark=None, plot=True, ...) -> EvaluationReport
    One-shot entry: read folder, compute everything, write summary.{json,csv,png}.

render_table(report) -> str
    Markdown-style grouped table for stdout.

load_result(result_dir) -> BacktestArtifacts
    Lower-level: just read the parquet files.

compute_all_metrics(artifacts, bench_nav=None) -> dict
    Pure-function metric kernel (single source of truth — referenced by
    BacktestResult.summary() as well).

load_benchmark(code, start, end) -> pd.Series
    Pull an index NAV from the local index_daily DuckDB table.
"""

from backtest.evaluation.loader import BacktestArtifacts, load_result
from backtest.evaluation.metrics import (
    compute_all_metrics,
    compute_drawdown_series,
    compute_holdings_stats,
    compute_monthly_return_matrix,
    compute_return_metrics,
    compute_risk_adjusted,
    compute_risk_metrics,
    compute_rolling_sharpe,
    compute_trading_stats,
    compute_winrate_metrics,
    compute_yearly_returns,
)
from backtest.evaluation.report import EvaluationReport, evaluate, render_table


__all__ = [
    # data access
    "BacktestArtifacts",
    "load_result",
    # metric kernel
    "compute_all_metrics",
    "compute_return_metrics",
    "compute_risk_metrics",
    "compute_risk_adjusted",
    "compute_winrate_metrics",
    "compute_trading_stats",
    "compute_holdings_stats",
    "compute_monthly_return_matrix",
    "compute_yearly_returns",
    "compute_drawdown_series",
    "compute_rolling_sharpe",
    # report + entry
    "EvaluationReport",
    "evaluate",
    "render_table",
]
