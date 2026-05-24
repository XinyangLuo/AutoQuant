"""Benchmark index loading and comparison metrics.

The benchmark series is sourced from ``MarketStorage.get_index_bars()`` —
backed by the ``index_daily`` table that ``backtest/data/backfill_indices.py``
populates from Tushare's ``pro.index_daily`` endpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage


_TRADING_DAYS = 252


def load_benchmark(
    code: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    storage: MarketStorage | None = None,
) -> pd.Series:
    """Return the index NAV series indexed by date, normalised so NAV[0] = 1.0.

    Parameters
    ----------
    code : str
        Tushare index ts_code, e.g. ``"000300.SH"``.
    start, end : str | Timestamp | None
        YYYYMMDD strings or pandas Timestamps; passed through to ``get_index_bars``.
    storage : MarketStorage, optional
        Pre-opened storage handle; if None a new one is opened and closed.
    """
    own = storage is None
    storage = storage or MarketStorage(read_only=True)
    try:
        s, e = _fmt_date(start), _fmt_date(end)
        df = storage.get_index_bars([code], start=s, end=e)
    finally:
        if own:
            storage.close()

    if df is None or df.empty:
        raise ValueError(
            f"No index data found for {code} in [{start}, {end}]. "
            f"Run `python -m backtest.data.backfill_indices --symbols {code}` first."
        )

    df = df.sort_values("date")
    nav = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
    nav.name = code
    return nav / nav.iloc[0]


def _fmt_date(d) -> str | None:
    if d is None:
        return None
    if isinstance(d, str):
        return d
    return pd.Timestamp(d).strftime("%Y%m%d")


def align_benchmark(strat_nav_df: pd.DataFrame, bench_nav: pd.Series) -> pd.Series:
    """Forward-fill the benchmark onto the strategy's trading dates, renormalise to 1.0."""
    strat_dates = pd.to_datetime(strat_nav_df["date"])
    aligned = bench_nav.reindex(strat_dates).ffill()
    if aligned.iloc[0] == 0 or pd.isna(aligned.iloc[0]):
        raise ValueError(
            "Benchmark series has no data at strategy start date — "
            "ensure the index is backfilled before the backtest start."
        )
    return aligned / aligned.iloc[0]


def compute_benchmark_metrics(
    strat_nav_df: pd.DataFrame,
    bench_nav: pd.Series,
) -> dict:
    """Beta / alpha / IR / tracking-error / excess drawdown vs. ``bench_nav``.

    ``bench_nav`` is expected to be the raw index NAV (from ``load_benchmark``).
    The function aligns it onto the strategy's trading dates and renormalises.
    """
    keys = [
        "bench_total_return", "bench_annual_return",
        "annual_excess_return", "tracking_error", "information_ratio",
        "beta", "alpha_annual", "corr",
        "excess_max_drawdown",
    ]
    out = {k: float("nan") for k in keys}

    if strat_nav_df is None or len(strat_nav_df) < 2 or bench_nav is None or len(bench_nav) < 2:
        return out

    aligned = align_benchmark(strat_nav_df, bench_nav)
    strat_s = strat_nav_df.set_index(pd.to_datetime(strat_nav_df["date"]))["nav"].astype(float)
    strat_s = strat_s / strat_s.iloc[0]

    r_strat = strat_s.pct_change()
    r_bench = aligned.pct_change()
    valid = pd.concat([r_strat, r_bench], axis=1, keys=["s", "b"]).dropna()
    if valid.empty:
        return out

    rs = valid["s"].values
    rb = valid["b"].values
    excess = rs - rb

    out["bench_total_return"] = float(aligned.iloc[-1] / aligned.iloc[0] - 1)
    n = len(valid)
    out["bench_annual_return"] = float((aligned.iloc[-1] / aligned.iloc[0]) ** (_TRADING_DAYS / n) - 1) if n > 0 else float("nan")

    out["annual_excess_return"] = float(excess.mean() * _TRADING_DAYS)
    excess_std = float(excess.std(ddof=1)) if n > 1 else 0.0
    out["tracking_error"] = excess_std * np.sqrt(_TRADING_DAYS)
    out["information_ratio"] = (
        out["annual_excess_return"] / out["tracking_error"]
        if out["tracking_error"] > 0 else float("nan")
    )

    # CAPM-style regression: r_strat = alpha + beta * r_bench
    if np.std(rb) > 0:
        beta, alpha_daily = np.polyfit(rb, rs, 1)
        out["beta"] = float(beta)
        out["alpha_annual"] = float(alpha_daily * _TRADING_DAYS)
        out["corr"] = float(np.corrcoef(rs, rb)[0, 1])

    cum_excess = (1.0 + pd.Series(excess, index=valid.index)).cumprod() - 1.0
    excess_nav = 1.0 + cum_excess
    excess_dd = excess_nav / excess_nav.cummax() - 1.0
    out["excess_max_drawdown"] = float(excess_dd.min())

    return out


def compute_excess_curve(strat_nav_df: pd.DataFrame, bench_nav: pd.Series) -> pd.Series:
    """Cumulative excess return as a pd.Series (date-indexed)."""
    if (
        strat_nav_df is None or len(strat_nav_df) < 2
        or bench_nav is None or len(bench_nav) < 2
    ):
        return pd.Series(dtype=float, name="cum_excess")
    aligned = align_benchmark(strat_nav_df, bench_nav)
    strat_s = strat_nav_df.set_index(pd.to_datetime(strat_nav_df["date"]))["nav"].astype(float)
    strat_s = strat_s / strat_s.iloc[0]
    excess_daily = (strat_s.pct_change() - aligned.pct_change()).dropna()
    cum = (1.0 + excess_daily).cumprod() - 1.0
    cum.name = "cum_excess"
    return cum
