"""Pure-function performance metrics for backtest NAV / trade / portfolio data.

This module is the single source of truth for all numeric backtest metrics in
the project.  ``BacktestResult.summary()`` delegates here.

All formulas are documented in ``backtest/evaluation/CLAUDE.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest.evaluation.loader import BacktestArtifacts


_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nav_series(nav_df: pd.DataFrame) -> pd.Series:
    """Return NAV indexed by date (Timestamp), values renormalised to start at 1."""
    s = nav_df.set_index(pd.to_datetime(nav_df["date"]))["nav"].astype(float).sort_index()
    return s / s.iloc[0]


def _daily_returns(nav_df: pd.DataFrame) -> pd.Series:
    s = _nav_series(nav_df)
    return s.pct_change().dropna()


def _empty_metrics_dict(keys: list[str]) -> dict:
    return {k: float("nan") for k in keys}


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------


def compute_return_metrics(nav_df: pd.DataFrame) -> dict:
    """Total / annual / volatility / best-worst / skew / kurtosis."""
    keys = [
        "total_return", "annual_return", "annual_volatility",
        "best_day", "worst_day", "best_month", "worst_month",
        "skewness", "kurtosis",
    ]
    if nav_df is None or len(nav_df) < 2:
        return _empty_metrics_dict(keys)

    nav = _nav_series(nav_df)
    r = nav.pct_change().dropna()
    n = len(r)

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1)
    annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (_TRADING_DAYS / n) - 1) if n > 0 else float("nan")
    annual_vol = float(r.std(ddof=1) * np.sqrt(_TRADING_DAYS)) if n > 1 else float("nan")

    monthly = (1.0 + r).resample("ME").prod() - 1.0
    best_month = float(monthly.max()) if not monthly.empty else float("nan")
    worst_month = float(monthly.min()) if not monthly.empty else float("nan")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "best_day": float(r.max()) if n > 0 else float("nan"),
        "worst_day": float(r.min()) if n > 0 else float("nan"),
        "best_month": best_month,
        "worst_month": worst_month,
        "skewness": float(r.skew()) if n > 2 else float("nan"),
        "kurtosis": float(r.kurt()) if n > 3 else float("nan"),
    }


def compute_monthly_return_matrix(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Year × month pivot of monthly returns (decimal, e.g. 0.0123 = +1.23%)."""
    if nav_df is None or len(nav_df) < 2:
        return pd.DataFrame()
    r = _daily_returns(nav_df)
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    if monthly.empty:
        return pd.DataFrame()
    matrix = monthly.to_frame("ret")
    matrix["year"] = matrix.index.year
    matrix["month"] = matrix.index.month
    return matrix.pivot(index="year", columns="month", values="ret").reindex(
        columns=range(1, 13)
    )


def compute_yearly_returns(nav_df: pd.DataFrame) -> pd.Series:
    """Per-calendar-year compounded returns."""
    if nav_df is None or len(nav_df) < 2:
        return pd.Series(dtype=float)
    r = _daily_returns(nav_df)
    yearly = (1.0 + r).resample("YE").prod() - 1.0
    yearly.index = yearly.index.year
    yearly.name = "yearly_return"
    return yearly


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


def compute_drawdown_series(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Return (date, drawdown) long DataFrame; drawdown <= 0 by construction."""
    if nav_df is None or len(nav_df) < 1:
        return pd.DataFrame(columns=["date", "drawdown"])
    nav = _nav_series(nav_df)
    dd = nav / nav.cummax() - 1.0
    out = dd.reset_index()
    out.columns = ["date", "drawdown"]
    return out


def compute_risk_metrics(nav_df: pd.DataFrame) -> dict:
    """Max drawdown + start/end + recovery, average drawdown, VaR / CVaR."""
    keys = [
        "max_drawdown", "max_drawdown_start", "max_drawdown_end", "recovery_days",
        "avg_drawdown", "var_95", "cvar_95",
    ]
    if nav_df is None or len(nav_df) < 2:
        return _empty_metrics_dict(keys)

    nav = _nav_series(nav_df)
    dd = nav / nav.cummax() - 1.0
    r = nav.pct_change().dropna()

    max_dd = float(dd.min())
    mdd_end = dd.idxmin()
    mdd_start = nav.loc[:mdd_end].idxmax()

    # Recovery: first date after mdd_end where nav >= nav[mdd_start]
    peak_val = nav.loc[mdd_start]
    post = nav.loc[mdd_end:]
    recovered = post[post >= peak_val]
    if recovered.empty:
        recovery_days: float | None = None
    else:
        recovery_days = float((recovered.index[0] - mdd_end).days)

    var_95 = float(np.percentile(r.values, 5)) if len(r) > 0 else float("nan")
    tail = r[r <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else float("nan")

    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    return {
        "max_drawdown": max_dd,
        "max_drawdown_start": mdd_start.strftime("%Y-%m-%d"),
        "max_drawdown_end": mdd_end.strftime("%Y-%m-%d"),
        "recovery_days": recovery_days,
        "avg_drawdown": avg_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


# ---------------------------------------------------------------------------
# Risk-adjusted metrics
# ---------------------------------------------------------------------------


def compute_risk_adjusted(
    nav_df: pd.DataFrame,
    rf: float = 0.0,
    *,
    bench_nav: pd.Series | None = None,
) -> dict:
    """Sharpe, Sortino, Calmar, Information Ratio.

    ``rf`` is the annual risk-free rate (e.g. 0.02 = 2%).  Information ratio
    is computed only when ``bench_nav`` is supplied (already aligned and
    normalised to start at 1.0).
    """
    keys = ["sharpe", "sortino", "calmar", "information_ratio"]
    if nav_df is None or len(nav_df) < 2:
        return _empty_metrics_dict(keys)

    nav = _nav_series(nav_df)
    r = nav.pct_change().dropna()
    n = len(r)

    annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (_TRADING_DAYS / n) - 1) if n > 0 else float("nan")
    annual_vol = float(r.std(ddof=1) * np.sqrt(_TRADING_DAYS)) if n > 1 else float("nan")
    max_dd = float((nav / nav.cummax() - 1.0).min())

    sharpe = (annual_return - rf) / annual_vol if annual_vol and not np.isnan(annual_vol) and annual_vol > 0 else float("nan")

    downside = r[r < 0]
    if len(downside) > 1:
        downside_vol = float(downside.std(ddof=1) * np.sqrt(_TRADING_DAYS))
        sortino = (annual_return - rf) / downside_vol if downside_vol > 0 else float("nan")
    else:
        sortino = float("nan")

    calmar = annual_return / abs(max_dd) if max_dd and not np.isnan(max_dd) and max_dd < 0 else float("nan")

    information_ratio = float("nan")
    if bench_nav is not None and len(bench_nav) > 1:
        bench_r = bench_nav.pct_change().dropna()
        excess = (r - bench_r).dropna()
        if len(excess) > 1 and excess.std(ddof=1) > 0:
            information_ratio = float(excess.mean() / excess.std(ddof=1) * np.sqrt(_TRADING_DAYS))

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "information_ratio": information_ratio,
    }


def compute_rolling_sharpe(nav_df: pd.DataFrame, window: int = 90) -> pd.Series:
    """Rolling annualised Sharpe over a trailing ``window``-day window."""
    if nav_df is None or len(nav_df) < window + 1:
        return pd.Series(dtype=float)
    r = _daily_returns(nav_df)
    rolling_mean = r.rolling(window).mean()
    rolling_std = r.rolling(window).std(ddof=1)
    rolling = (rolling_mean / rolling_std) * np.sqrt(_TRADING_DAYS)
    rolling.name = f"rolling_sharpe_{window}d"
    return rolling


# ---------------------------------------------------------------------------
# Win-rate metrics
# ---------------------------------------------------------------------------


def compute_winrate_metrics(nav_df: pd.DataFrame) -> dict:
    """Daily / monthly / yearly win rate and profit-loss ratio."""
    keys = ["daily_win_rate", "monthly_win_rate", "yearly_win_rate", "profit_loss_ratio"]
    if nav_df is None or len(nav_df) < 2:
        return _empty_metrics_dict(keys)

    r = _daily_returns(nav_df)
    daily_wr = float((r > 0).mean()) if len(r) > 0 else float("nan")

    monthly = (1.0 + r).resample("ME").prod() - 1.0
    monthly_wr = float((monthly > 0).mean()) if not monthly.empty else float("nan")

    yearly = (1.0 + r).resample("YE").prod() - 1.0
    yearly_wr = float((yearly > 0).mean()) if not yearly.empty else float("nan")

    wins = r[r > 0]
    losses = r[r < 0]
    if not wins.empty and not losses.empty:
        pl_ratio = float(wins.mean() / abs(losses.mean()))
    else:
        pl_ratio = float("nan")

    return {
        "daily_win_rate": daily_wr,
        "monthly_win_rate": monthly_wr,
        "yearly_win_rate": yearly_wr,
        "profit_loss_ratio": pl_ratio,
    }


# ---------------------------------------------------------------------------
# Trading statistics (require trades + metrics dataframes)
# ---------------------------------------------------------------------------


def compute_trading_stats(
    trades: pd.DataFrame | None,
    metrics_df: pd.DataFrame | None,
    initial_cash: float,
) -> dict:
    """Total trades / fees / turnover. Returns NaN-filled dict on missing data."""
    keys = [
        "total_trades", "total_commission", "total_stamp_duty",
        "total_transfer_fee", "total_fees", "fees_pct_of_initial",
        "avg_daily_turnover", "annual_turnover",
    ]
    out = _empty_metrics_dict(keys)
    if trades is None and metrics_df is None:
        return out

    if trades is not None and not trades.empty:
        out["total_trades"] = int(len(trades))
        if "commission" in trades.columns:
            out["total_commission"] = float(trades["commission"].sum())

    if metrics_df is not None and not metrics_df.empty:
        if "stamp_duty" in metrics_df.columns:
            out["total_stamp_duty"] = float(metrics_df["stamp_duty"].sum())
        if "transfer_fee" in metrics_df.columns:
            out["total_transfer_fee"] = float(metrics_df["transfer_fee"].sum())
        if "turnover" in metrics_df.columns:
            avg_turn = float(metrics_df["turnover"].mean())
            out["avg_daily_turnover"] = avg_turn
            out["annual_turnover"] = avg_turn * _TRADING_DAYS

    # Fallback stamp_duty if metrics_df missing but trades has it
    if np.isnan(out["total_stamp_duty"]) and trades is not None and not trades.empty:
        if {"direction", "amount"}.issubset(trades.columns):
            sells = trades[trades["direction"].isin(["sell", "short"])]
            out["total_stamp_duty"] = float(sells["amount"].sum() * 0.001)

    parts = [out[k] for k in ("total_commission", "total_stamp_duty", "total_transfer_fee")]
    parts = [p for p in parts if not np.isnan(p)]
    if parts:
        out["total_fees"] = float(sum(parts))
        if initial_cash > 0:
            out["fees_pct_of_initial"] = out["total_fees"] / initial_cash

    return out


# ---------------------------------------------------------------------------
# Holdings statistics (require metrics dataframe)
# ---------------------------------------------------------------------------


_HOLDINGS_FIELDS = [
    "position_count", "long_count", "short_count",
    "cash_ratio", "gross_exposure", "net_exposure",
    "herfindahl", "top5_weight", "top10_weight",
]


def compute_holdings_stats(metrics_df: pd.DataFrame | None) -> dict:
    """Average of the daily-portfolio statistics in ``metrics.parquet``."""
    keys = [f"avg_{k}" for k in _HOLDINGS_FIELDS]
    out = _empty_metrics_dict(keys)
    if metrics_df is None or metrics_df.empty:
        return out
    for field in _HOLDINGS_FIELDS:
        if field in metrics_df.columns:
            out[f"avg_{field}"] = float(metrics_df[field].mean())
    return out


# ---------------------------------------------------------------------------
# Top-level merge
# ---------------------------------------------------------------------------


def compute_all_metrics(
    artifacts: "BacktestArtifacts",
    bench_nav: pd.Series | None = None,
    rf: float = 0.0,
    bench_navs: dict[str, pd.Series] | None = None,
) -> dict:
    """Compute every flat scalar metric and return one merged dict.

    DataFrame-shaped derivatives (monthly/yearly returns, drawdown series,
    rolling sharpe, excess curve, trade-reason histogram) are produced by
    ``report.evaluate()``, not by this function.
    """
    nav_df = artifacts.nav
    out: dict = {}
    out.update(compute_return_metrics(nav_df))
    out.update(compute_risk_metrics(nav_df))
    out.update(compute_risk_adjusted(nav_df, rf=rf, bench_nav=bench_nav))
    out.update(compute_winrate_metrics(nav_df))
    out.update(compute_trading_stats(artifacts.trades, artifacts.metrics, artifacts.initial_cash))
    out.update(compute_holdings_stats(artifacts.metrics))

    # Multi-benchmark excess metrics (computed post-simulation)
    if bench_navs:
        # Local import avoids circular dependency at import time.
        from backtest.evaluation.benchmark import (
            _BENCH_METRIC_MAP,
            compute_benchmark_metrics,
        )
        for alias, nav in bench_navs.items():
            bm = compute_benchmark_metrics(nav_df, nav)
            for out_key, src_key in _BENCH_METRIC_MAP.items():
                out[f"{out_key}_{alias}"] = bm.get(src_key, float("nan"))

    out["n_days"] = artifacts.n_days
    out["start_date"] = artifacts.start.strftime("%Y-%m-%d")
    out["end_date"] = artifacts.end.strftime("%Y-%m-%d")
    out["initial_cash"] = artifacts.initial_cash
    return out


def compute_single_nav_metrics(nav_df: pd.DataFrame) -> dict:
    """Compute return + risk + risk-adjusted metrics for a single NAV series.

    Thin wrapper that merges the three core metric functions into one dict.
    """
    m = compute_return_metrics(nav_df)
    m.update(compute_risk_metrics(nav_df))
    m.update(compute_risk_adjusted(nav_df))
    return m
