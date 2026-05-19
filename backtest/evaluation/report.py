"""High-level evaluation orchestration: build EvaluationReport, render table, save."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.evaluation.loader import BacktestArtifacts, load_result
from backtest.evaluation.metrics import (
    compute_all_metrics,
    compute_drawdown_series,
    compute_monthly_return_matrix,
    compute_rolling_sharpe,
    compute_yearly_returns,
)


@dataclass
class EvaluationReport:
    """All artefacts needed to print + plot + persist evaluation output."""

    artifacts: BacktestArtifacts
    benchmark_code: str | None
    bench_nav: pd.Series | None
    metrics: dict                              # flat scalar metrics
    monthly_returns: pd.DataFrame              # year × month
    yearly_returns: pd.Series
    drawdown: pd.DataFrame                     # date, drawdown
    rolling_sharpe: pd.Series
    reason_histogram: pd.Series | None
    bench_metrics: dict = field(default_factory=dict)
    excess_curve: pd.Series | None = None

    # --- serialisation ---------------------------------------------------

    def to_json(self) -> dict:
        """JSON-safe nested dict for ``summary.json``."""
        out = {
            "metadata": self.artifacts.metadata,
            "result_dir": str(self.artifacts.result_dir),
            "start_date": self.artifacts.start.strftime("%Y-%m-%d"),
            "end_date": self.artifacts.end.strftime("%Y-%m-%d"),
            "benchmark": self.benchmark_code,
            "metrics": _coerce_json(self.metrics),
            "bench_metrics": _coerce_json(self.bench_metrics),
            "yearly_returns": {
                str(k): _coerce_scalar(v) for k, v in self.yearly_returns.items()
            },
            "monthly_returns": _matrix_to_json(self.monthly_returns),
        }
        if self.reason_histogram is not None and not self.reason_histogram.empty:
            out["trade_reasons"] = {str(k): int(v) for k, v in self.reason_histogram.items()}
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """One-row wide DataFrame for ``summary.csv``."""
        row = {**self.metrics}
        if self.bench_metrics:
            row.update(self.bench_metrics)
        row["benchmark"] = self.benchmark_code or ""
        row["start_date"] = self.artifacts.start.strftime("%Y-%m-%d")
        row["end_date"] = self.artifacts.end.strftime("%Y-%m-%d")
        return pd.DataFrame([row])

    def render_markdown(self) -> str:
        return render_table(self)


def _coerce_scalar(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def _coerce_json(d: dict) -> dict:
    return {k: _coerce_scalar(v) for k, v in d.items()}


def _matrix_to_json(matrix: pd.DataFrame) -> dict:
    if matrix is None or matrix.empty:
        return {}
    out: dict = {}
    for year, row in matrix.iterrows():
        out[str(int(year))] = {str(int(m)): _coerce_scalar(v) for m, v in row.items()}
    return out


# ---------------------------------------------------------------------------
# evaluate(): the single high-level entry point
# ---------------------------------------------------------------------------


def evaluate(
    result_dir: str | Path,
    benchmark: str | None = None,
    *,
    plot: bool = True,
    output_dir: str | Path | None = None,
    rf: float = 0.0,
    rolling_sharpe_window: int = 90,
) -> EvaluationReport:
    """Read a saved backtest directory and build an EvaluationReport.

    When ``plot=True`` the 8-panel figure is saved as ``report.png``;
    in any case ``summary.json`` and ``summary.csv`` are written.
    All outputs land in ``output_dir`` (default: the input ``result_dir``).
    """
    artifacts = load_result(result_dir)

    bench_nav: pd.Series | None = None
    bench_metrics: dict = {}
    excess_curve: pd.Series | None = None
    if benchmark:
        # Local import — load_benchmark touches MarketStorage which opens DuckDB.
        from backtest.evaluation.benchmark import (
            compute_benchmark_metrics, compute_excess_curve, load_benchmark,
        )
        bench_nav = load_benchmark(
            benchmark,
            start=artifacts.start.strftime("%Y%m%d"),
            end=artifacts.end.strftime("%Y%m%d"),
        )
        bench_metrics = compute_benchmark_metrics(artifacts.nav, bench_nav)
        excess_curve = compute_excess_curve(artifacts.nav, bench_nav)

    metrics = compute_all_metrics(artifacts, bench_nav=bench_nav, rf=rf)

    monthly = compute_monthly_return_matrix(artifacts.nav)
    yearly = compute_yearly_returns(artifacts.nav)
    drawdown = compute_drawdown_series(artifacts.nav)
    rolling = compute_rolling_sharpe(artifacts.nav, window=rolling_sharpe_window)

    reason_hist: pd.Series | None = None
    if artifacts.trades is not None and "reason" in artifacts.trades.columns:
        reason_hist = artifacts.trades["reason"].value_counts()

    report = EvaluationReport(
        artifacts=artifacts,
        benchmark_code=benchmark,
        bench_nav=bench_nav,
        metrics=metrics,
        monthly_returns=monthly,
        yearly_returns=yearly,
        drawdown=drawdown,
        rolling_sharpe=rolling,
        reason_histogram=reason_hist,
        bench_metrics=bench_metrics,
        excess_curve=excess_curve,
    )

    out_path = Path(output_dir) if output_dir else artifacts.result_dir
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report.to_json(), f, ensure_ascii=False, indent=2, default=str)
    report.to_dataframe().to_csv(out_path / "summary.csv", index=False)

    if plot:
        # Local import keeps matplotlib out of import-time for non-plot users
        from backtest.evaluation.plot import plot_report
        plot_report(report, output_path=out_path / "report.png")

    return report


# ---------------------------------------------------------------------------
# render_table(): markdown-ish ASCII table
# ---------------------------------------------------------------------------


_RETURN_KEYS = [
    ("Total Return", "total_return", "pct"),
    ("Annualised Return", "annual_return", "pct"),
    ("Annualised Volatility", "annual_volatility", "pct"),
    ("Best Day", "best_day", "pct"),
    ("Worst Day", "worst_day", "pct"),
    ("Best Month", "best_month", "pct"),
    ("Worst Month", "worst_month", "pct"),
    ("Skewness", "skewness", "f3"),
    ("Kurtosis", "kurtosis", "f3"),
]

_RISK_ADJ_KEYS = [
    ("Sharpe", "sharpe", "f3"),
    ("Sortino", "sortino", "f3"),
    ("Calmar", "calmar", "f3"),
    ("Information Ratio", "information_ratio", "f3"),
]

_RISK_KEYS = [
    ("Max Drawdown", "max_drawdown", "pct"),
    ("MaxDD Start", "max_drawdown_start", "str"),
    ("MaxDD End", "max_drawdown_end", "str"),
    ("Recovery Days", "recovery_days", "days"),
    ("Avg Drawdown", "avg_drawdown", "pct"),
    ("VaR 95%", "var_95", "pct"),
    ("CVaR 95%", "cvar_95", "pct"),
]

_WINRATE_KEYS = [
    ("Daily Win Rate", "daily_win_rate", "pct"),
    ("Monthly Win Rate", "monthly_win_rate", "pct"),
    ("Yearly Win Rate", "yearly_win_rate", "pct"),
    ("Profit/Loss Ratio", "profit_loss_ratio", "f3"),
]

_TRADING_KEYS = [
    ("Total Trades", "total_trades", "int"),
    ("Total Commission", "total_commission", "money"),
    ("Total Stamp Duty", "total_stamp_duty", "money"),
    ("Total Transfer Fee", "total_transfer_fee", "money"),
    ("Total Fees", "total_fees", "money"),
    ("Fees % of Initial", "fees_pct_of_initial", "pct"),
    ("Avg Daily Turnover", "avg_daily_turnover", "pct"),
    ("Annualised Turnover", "annual_turnover", "f2"),
]

_HOLDINGS_KEYS = [
    ("Avg Position Count", "avg_position_count", "f1"),
    ("Avg Long Count", "avg_long_count", "f1"),
    ("Avg Short Count", "avg_short_count", "f1"),
    ("Avg Cash Ratio", "avg_cash_ratio", "pct"),
    ("Avg Gross Exposure", "avg_gross_exposure", "pct"),
    ("Avg Net Exposure", "avg_net_exposure", "pct"),
    ("Avg Top5 Weight", "avg_top5_weight", "pct"),
    ("Avg Top10 Weight", "avg_top10_weight", "pct"),
    ("Avg Herfindahl", "avg_herfindahl", "f4"),
]

_BENCH_KEYS = [
    ("Benchmark Total Return", "bench_total_return", "pct"),
    ("Benchmark Annual Return", "bench_annual_return", "pct"),
    ("Annual Excess Return", "annual_excess_return", "pct"),
    ("Tracking Error", "tracking_error", "pct"),
    ("Information Ratio", "information_ratio", "f3"),
    ("Beta", "beta", "f3"),
    ("Alpha (annual)", "alpha_annual", "pct"),
    ("Correlation", "corr", "f3"),
    ("Excess Max Drawdown", "excess_max_drawdown", "pct"),
]


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    if kind == "pct":
        return f"{value:+.2%}"
    if kind == "f1":
        return f"{value:,.1f}"
    if kind == "f2":
        return f"{value:,.2f}"
    if kind == "f3":
        return f"{value:+.3f}"
    if kind == "f4":
        return f"{value:.4f}"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "money":
        return f"{value:,.2f}"
    if kind == "days":
        return f"{int(value)}" if value is not None else "not recovered"
    return str(value)


def _render_section(title: str, rows: list[tuple[str, str, str]], data: dict) -> list[str]:
    lines = ["", f"## {title}"]
    width = max(len(r[0]) for r in rows) + 2
    for label, key, kind in rows:
        val = data.get(key)
        lines.append(f"{label.ljust(width)}  {_fmt(val, kind)}")
    return lines


def render_table(report: EvaluationReport) -> str:
    """Pretty-print all metrics in grouped sections to a string."""
    a = report.artifacts
    header = [
        "=" * 70,
        f"Backtest Evaluation: {a.strategy_id}",
        f"Period: {a.start.strftime('%Y-%m-%d')} ~ {a.end.strftime('%Y-%m-%d')}  "
        f"({a.n_days} trading days, initial_cash={a.initial_cash:,.0f})",
    ]
    if report.benchmark_code:
        header.append(f"Benchmark: {report.benchmark_code}")
    header.append("=" * 70)

    lines = list(header)
    lines += _render_section("Return", _RETURN_KEYS, report.metrics)
    lines += _render_section("Risk-Adjusted", _RISK_ADJ_KEYS, report.metrics)
    lines += _render_section("Risk", _RISK_KEYS, report.metrics)
    lines += _render_section("Win Rate", _WINRATE_KEYS, report.metrics)

    if a.trades is not None or a.metrics is not None:
        lines += _render_section("Trading", _TRADING_KEYS, report.metrics)
    if a.metrics is not None:
        lines += _render_section("Holdings", _HOLDINGS_KEYS, report.metrics)
    if report.bench_metrics:
        lines += _render_section("Benchmark Comparison", _BENCH_KEYS, report.bench_metrics)

    # Yearly returns
    if not report.yearly_returns.empty:
        lines += ["", "## Yearly Returns"]
        for year, ret in report.yearly_returns.items():
            lines.append(f"  {year}: {_fmt(ret, 'pct')}")

    # Trade reasons
    if report.reason_histogram is not None and not report.reason_histogram.empty:
        lines += ["", "## Trade Reasons"]
        total = int(report.reason_histogram.sum())
        for reason, count in report.reason_histogram.items():
            pct = count / total if total > 0 else 0.0
            lines.append(f"  {str(reason).ljust(24)}  {int(count):>6,}  ({pct:.1%})")

    lines.append("=" * 70)
    return "\n".join(lines)
