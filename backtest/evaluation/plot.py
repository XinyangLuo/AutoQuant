"""Render the 8-panel evaluation figure (matplotlib Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.evaluation.report import EvaluationReport


_FIGSIZE = (16, 32)
_DPI = 150


def _draw_empty(ax, title: str, message: str = "no data") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def _nav_plot(ax, report: EvaluationReport) -> None:
    nav = report.artifacts.nav
    dates = pd.to_datetime(nav["date"])
    strat_nav = nav["nav"].astype(float).values
    strat_nav = strat_nav / strat_nav[0]

    ax.plot(dates, strat_nav, color="steelblue", linewidth=1.4, label="Strategy")
    if report.bench_nav is not None:
        from backtest.evaluation.benchmark import align_benchmark
        try:
            bench = align_benchmark(nav, report.bench_nav)
            ax.plot(dates, bench.values, color="darkorange", linewidth=1.2,
                    label=f"Benchmark ({report.benchmark_code})")
        except ValueError:
            pass

    total_ret = report.metrics.get("total_return", float("nan"))
    ann_ret = report.metrics.get("annual_return", float("nan"))
    ax.set_title(
        f"Strategy NAV  (total={_pct(total_ret)}, ann={_pct(ann_ret)})"
    )
    ax.set_ylabel("NAV")
    ax.set_xlabel("Date")
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")


def _drawdown_plot(ax, report: EvaluationReport) -> None:
    dd = report.drawdown
    if dd.empty:
        _draw_empty(ax, "Drawdown")
        return
    dates = pd.to_datetime(dd["date"])
    values = dd["drawdown"].astype(float).values
    ax.fill_between(dates, values, 0, color="red", alpha=0.3)
    ax.plot(dates, values, color="red", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)

    max_dd = report.metrics.get("max_drawdown", float("nan"))
    start = report.metrics.get("max_drawdown_start", "")
    end = report.metrics.get("max_drawdown_end", "")
    ax.set_title(f"Drawdown  (max={_pct(max_dd)}, {start} ~ {end})")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)


def _monthly_heatmap(ax, report: EvaluationReport) -> None:
    matrix = report.monthly_returns
    if matrix is None or matrix.empty:
        _draw_empty(ax, "Monthly Returns")
        return
    data = matrix.values.astype(float) * 100.0  # to percent for display
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(int(y)) for y in matrix.index])
    threshold = vmax / 2 if vmax > 0 else 0
    for i, year in enumerate(matrix.index):
        for j, month in enumerate(matrix.columns):
            v = matrix.loc[year, month]
            if pd.isna(v):
                continue
            color = "white" if abs(v * 100) > threshold else "black"
            ax.text(j, i, f"{v * 100:+.1f}", ha="center", va="center",
                    color=color, fontsize=8)
    ax.set_title("Monthly Returns (%)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)


def _yearly_bar(ax, report: EvaluationReport) -> None:
    yearly = report.yearly_returns
    if yearly.empty:
        _draw_empty(ax, "Yearly Returns")
        return
    years = [str(int(y)) for y in yearly.index]
    values = yearly.values.astype(float)
    colors = ["green" if v >= 0 else "red" for v in values]
    bars = ax.bar(years, values * 100.0, color=colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Yearly Returns")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 100,
                f"{v:+.1%}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)


def _holdings_plot(ax, report: EvaluationReport) -> None:
    metrics_df = report.artifacts.metrics
    if metrics_df is None or metrics_df.empty:
        _draw_empty(ax, "Positions & Cash Ratio", "no detailed metrics")
        return
    dates = pd.to_datetime(metrics_df["date"])
    if "position_count" in metrics_df.columns:
        ax.plot(dates, metrics_df["position_count"], color="steelblue",
                linewidth=1.2, label="Position count")
    ax.set_ylabel("Position count", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlabel("Date")
    ax.set_title("Positions & Cash Ratio")
    ax.grid(True, alpha=0.3)

    if "cash_ratio" in metrics_df.columns:
        ax2 = ax.twinx()
        ax2.plot(dates, metrics_df["cash_ratio"], color="gray",
                 linewidth=1.0, linestyle="--", label="Cash ratio")
        ax2.set_ylabel("Cash ratio", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.set_ylim(0, max(0.05, metrics_df["cash_ratio"].max() * 1.1))


def _turnover_plot(ax, report: EvaluationReport) -> None:
    metrics_df = report.artifacts.metrics
    if metrics_df is None or "turnover" not in metrics_df.columns or metrics_df["turnover"].sum() == 0:
        _draw_empty(ax, "Daily Turnover", "no detailed metrics")
        return
    dates = pd.to_datetime(metrics_df["date"])
    turn = metrics_df["turnover"].astype(float)
    ax.plot(dates, turn, color="purple", linewidth=0.8, alpha=0.6, label="Daily")
    rolling = turn.rolling(20, min_periods=1).mean()
    ax.plot(dates, rolling, color="black", linewidth=1.2, label="20d rolling mean")
    ax.set_title("Daily Turnover (two-side)")
    ax.set_ylabel("Turnover")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")


def _return_histogram(ax, report: EvaluationReport) -> None:
    nav = report.artifacts.nav
    if "daily_return" in nav.columns:
        r = nav["daily_return"].dropna().to_numpy()
    else:
        nav_s = nav["nav"].astype(float).to_numpy()
        r = np.diff(nav_s) / nav_s[:-1]
    if len(r) == 0:
        _draw_empty(ax, "Daily Return Distribution")
        return
    ax.hist(r, bins=60, density=True, color="steelblue", alpha=0.6,
            edgecolor="white")
    # Normal overlay
    mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
    if sigma > 0:
        xs = np.linspace(r.min(), r.max(), 200)
        pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        ax.plot(xs, pdf, color="red", linestyle="--", linewidth=1.2, label="Normal fit")

    var = report.metrics.get("var_95", float("nan"))
    cvar = report.metrics.get("cvar_95", float("nan"))
    if not np.isnan(var):
        ax.axvline(var, color="orange", linestyle=":", linewidth=1.2,
                   label=f"VaR95%={var:+.3%}")
    if not np.isnan(cvar):
        ax.axvline(cvar, color="red", linestyle=":", linewidth=1.2,
                   label=f"CVaR95%={cvar:+.3%}")
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _rolling_sharpe(ax, report: EvaluationReport) -> None:
    rs = report.rolling_sharpe
    if rs is None or rs.empty:
        _draw_empty(ax, "Rolling Sharpe (90d)", "history too short")
        return
    dates = pd.to_datetime(rs.index)
    ax.plot(dates, rs.values, color="teal", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5)
    mean_rs = float(np.nanmean(rs.values))
    if not np.isnan(mean_rs):
        ax.axhline(mean_rs, color="blue", linestyle="--", linewidth=1.0,
                   label=f"mean={mean_rs:+.2f}")

    sharpe = report.metrics.get("sharpe", float("nan"))
    ax.set_title(f"Rolling Sharpe (90d)  (full-sample Sharpe={sharpe:+.3f})")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.2%}"


def plot_report(report: EvaluationReport, output_path: str | Path) -> str:
    """Render the 8-panel evaluation figure to ``output_path`` (PNG)."""
    fig, axes = plt.subplots(8, 1, figsize=_FIGSIZE)
    title = (
        f"{report.artifacts.strategy_id}  |  "
        f"{report.artifacts.start.strftime('%Y-%m-%d')} ~ "
        f"{report.artifacts.end.strftime('%Y-%m-%d')}"
    )
    if report.benchmark_code:
        title += f"  |  benchmark={report.benchmark_code}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    _nav_plot(axes[0], report)
    _drawdown_plot(axes[1], report)
    _monthly_heatmap(axes[2], report)
    _yearly_bar(axes[3], report)
    _holdings_plot(axes[4], report)
    _turnover_plot(axes[5], report)
    _return_histogram(axes[6], report)
    _rolling_sharpe(axes[7], report)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out)
