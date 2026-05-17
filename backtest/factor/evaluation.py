"""Offline factor evaluation: IC, RankIC, ICIR, turnover, decay, group returns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.registry import get_registry
from backtest.factor.storage import FactorStorage


DEFAULT_HORIZONS = [1, 5, 10, 20, 60]
_CORR_COLUMNS = ["factor_id", "corr", "n_dates"]


def _load_market_data(
    symbols: list[str],
    start: str,
    end: str,
    horizons: list[int],
    ret_type: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pre-load market data and compute forward returns once.

    Returns (market_df, returns_df, limit_df) for reuse across multiple factors.
    """
    with MarketStorage() as ms:
        market_df = ms.get_bars(
            symbols=symbols,
            start=start,
            end=end,
        )

    if market_df.empty:
        raise ValueError("No market data available for return calculation")

    returns_df = _compute_forward_returns(market_df, horizons, ret_type)

    limit_cols = ["date", "symbol", "close", "open", "limit_up"]
    if all(c in market_df.columns for c in limit_cols):
        limit_df = market_df[limit_cols].copy()
    else:
        limit_df = pd.DataFrame(columns=limit_cols)

    return market_df, returns_df, limit_df


def _load_factor_and_returns(
    factor_id: str,
    start: str,
    end: str,
    horizons: list[int],
    ret_type: str = "close",
    returns_df: pd.DataFrame | None = None,
    limit_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load factor values, compute forward returns, and load limit prices for filtering.

    If ``returns_df`` and ``limit_df`` are provided (from a prior
    :func:`_load_market_data` call), they are reused instead of re-querying
    the database.
    """
    with FactorStorage() as fs:
        factor_df = fs.get_factor(factor_id, start, end)

    if factor_df.empty:
        raise ValueError(f"No factor data for {factor_id} in range {start}~{end}")

    if returns_df is not None and limit_df is not None:
        return factor_df, returns_df, limit_df

    max_h = max(horizons)
    factor_end = factor_df["date"].max()
    returns_end = (factor_end + pd.Timedelta(days=max_h + 5)).strftime("%Y%m%d")

    symbols = factor_df["symbol"].unique().tolist()
    _, returns_df, limit_df = _load_market_data(
        symbols=symbols, start=start, end=returns_end,
        horizons=horizons, ret_type=ret_type,
    )
    return factor_df, returns_df, limit_df


def _compute_forward_returns(
    df: pd.DataFrame, horizons: list[int], ret_type: str
) -> pd.DataFrame:
    """Compute forward returns for all horizons in a single pass."""
    df = df[["date", "symbol", "close", "open"]].copy()
    df = df.sort_values(["symbol", "date"])

    for h in horizons:
        if ret_type == "close":
            df[f"ret_{h}"] = df.groupby("symbol")["close"].shift(-h) / df["close"] - 1
        else:  # open
            df[f"ret_{h}"] = (
                df.groupby("symbol")["open"].shift(-(h + 1))
                / df.groupby("symbol")["open"].shift(-1)
                - 1
            )

    ret_cols = [f"ret_{h}" for h in horizons]
    return df[["date", "symbol"] + ret_cols].dropna(subset=ret_cols, how="all")


def _ic_series(factor_vals: pd.Series, returns: pd.Series) -> float:
    """Pearson correlation (single day's IC)."""
    mask = factor_vals.notna() & returns.notna()
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(factor_vals[mask], returns[mask])[0, 1])


def _rank_ic_series(factor_vals: pd.Series, returns: pd.Series) -> float:
    """Spearman correlation (single day's RankIC) — pure numpy, no scipy."""
    mask = factor_vals.notna() & returns.notna()
    if mask.sum() < 3:
        return np.nan
    f_rank = factor_vals[mask].rank().values
    r_rank = returns[mask].rank().values
    f_rank = f_rank - f_rank.mean()
    r_rank = r_rank - r_rank.mean()
    denom = np.sqrt((f_rank**2).sum() * (r_rank**2).sum())
    if denom == 0:
        return np.nan
    return float((f_rank * r_rank).sum() / denom)


def _compute_ic_stats(ic_series: pd.Series) -> dict:
    """Compute IC mean, std, ICIR, t-stat, positive ratio."""
    valid = ic_series.dropna()
    if len(valid) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_tstat": np.nan,
            "ic_positive_ratio": np.nan,
            "ic_count": 0,
        }

    mean = valid.mean()
    std = valid.std()
    n = len(valid)
    icir = mean / std if std != 0 else np.nan
    tstat = mean / (std / np.sqrt(n)) if std != 0 else np.nan
    pos_ratio = (valid > 0).sum() / n

    return {
        "ic_mean": float(mean),
        "ic_std": float(std),
        "icir": float(icir),
        "ic_tstat": float(tstat),
        "ic_positive_ratio": float(pos_ratio),
        "ic_count": n,
    }


def _turnover(factor_df: pd.DataFrame) -> float:
    """Average rank turnover between consecutive periods."""
    df = factor_df[["date", "symbol", "value"]].copy()
    df["rank"] = df.groupby("date")["value"].rank(pct=True)

    # Use wide format but avoid materializing full dense matrix
    wide = df.pivot(index="date", columns="symbol", values="rank")
    wide = wide.fillna(0.5)
    turnover = wide.diff().abs().mean().mean() * 2
    return float(turnover)


def _group_returns(
    merged: pd.DataFrame,
    ret_col: str,
    n_groups: int = 10,
) -> pd.DataFrame:
    """Compute mean future return per quantile group."""
    merged = merged[["date", "symbol", "value", ret_col]].dropna()
    merged["group"] = merged.groupby("date")["value"].transform(
        lambda x: pd.qcut(x, n_groups, labels=False, duplicates="drop")
    )

    grouped = merged.groupby("group")[ret_col].agg(["mean", "std", "count"])
    grouped = grouped.reset_index()
    grouped.columns = ["group", "mean_ret", "std_ret", "count"]
    return grouped


def _corr_with_existing(
    factor_df: pd.DataFrame,
    factor_id: str,
    storage: FactorStorage,
    top_k: int = 5,
) -> pd.DataFrame:
    """Average daily cross-sectional rank correlation with *admitted* factors only.

    Only factors with ``status='admitted'`` in the registry participate in the
    comparison. Rejected / pending factors are ignored — their data may have
    been purged from ``factors_daily`` anyway.

    Pass ``top_k=0`` to skip the comparison entirely.
    """
    if top_k <= 0 or factor_df.empty:
        return pd.DataFrame(columns=_CORR_COLUMNS)

    admitted = {
        fid for fid, meta in get_registry().items()
        if meta.get("status") == "admitted" and fid != factor_id
    }
    if not admitted:
        return pd.DataFrame(columns=_CORR_COLUMNS)

    start = factor_df["date"].min().strftime("%Y%m%d")
    end = factor_df["date"].max().strftime("%Y%m%d")
    others = storage.get_factors_long(
        factor_ids=list(admitted), start=start, end=end
    )
    if others.empty:
        return pd.DataFrame(columns=_CORR_COLUMNS)

    merged = others.merge(
        factor_df[["date", "symbol", "value"]].rename(columns={"value": "value_self"}),
        on=["date", "symbol"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=_CORR_COLUMNS)

    daily = merged.groupby(["factor_id", "date"]).apply(
        lambda g: _rank_ic_series(g["value_self"], g["value"]),
        include_groups=False,
    ).dropna()
    if daily.empty:
        return pd.DataFrame(columns=_CORR_COLUMNS)

    stats = daily.groupby(level=0).agg(["mean", "count"])
    stats.columns = ["corr", "n_dates"]
    stats = stats.reset_index()
    stats["n_dates"] = stats["n_dates"].astype(int)

    order = stats["corr"].abs().sort_values(ascending=False).index
    return stats.loc[order, _CORR_COLUMNS].head(top_k).reset_index(drop=True)


@dataclass
class EvaluationResult:
    factor_id: str
    ret_type: str
    horizons: list[int]
    start: str
    end: str
    ic_metrics: dict[int, dict]
    rank_ic_metrics: dict[int, dict]
    decay: dict[int, float]
    turnover: float
    group_returns: dict[int, pd.DataFrame]
    corr_with_existing: pd.DataFrame
    ic_series: dict[int, pd.Series] = field(default_factory=dict)
    rank_ic_series: dict[int, pd.Series] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all metrics by horizon."""
        rows = []
        for h in self.horizons:
            ic = self.ic_metrics.get(h, {})
            ric = self.rank_ic_metrics.get(h, {})
            rows.append({
                "horizon": h,
                "IC_mean": ic.get("ic_mean"),
                "IC_std": ic.get("ic_std"),
                "ICIR": ic.get("icir"),
                "IC_tstat": ic.get("ic_tstat"),
                "IC_pos_ratio": ic.get("ic_positive_ratio"),
                "RankIC_mean": ric.get("ic_mean"),
                "RankIC_std": ric.get("ic_std"),
                "RankICIR": ric.get("icir"),
                "RankIC_tstat": ric.get("ic_tstat"),
                "RankIC_pos_ratio": ric.get("ic_positive_ratio"),
            })
        return pd.DataFrame(rows)

    def max_corr(self) -> tuple[str, float] | None:
        """Return ``(factor_id, corr)`` of the most similar existing factor.

        Returns ``None`` if no other factors are stored.
        """
        if self.corr_with_existing.empty:
            return None
        row = self.corr_with_existing.iloc[0]
        return str(row["factor_id"]), float(row["corr"])

    def __repr__(self) -> str:
        return f"EvaluationResult({self.factor_id}, ret_type={self.ret_type}, horizons={self.horizons})"


_LIMIT_EPS = 1e-6


def _exclude_limit_up(
    merged: pd.DataFrame,
    limit_df: pd.DataFrame,
    ret_type: str,
) -> pd.DataFrame:
    """Drop rows where execution is blocked by limit-up.

    - close-to-close: exclude if close_t >= limit_up_t (can't buy at close)
    - open-to-open  : exclude if open_{t+1} >= limit_up_{t+1} (can't buy at next open)
    """
    if limit_df.empty:
        return merged

    if ret_type == "close":
        limit_sub = limit_df[["date", "symbol", "close", "limit_up"]]
        merged = merged.merge(limit_sub, on=["date", "symbol"], how="left")
        mask = merged["close"] < merged["limit_up"] - _LIMIT_EPS
        n_dropped = int((~mask).sum())
        merged = merged[mask].drop(columns=["close", "limit_up"], errors="ignore")
    else:  # open
        # Get T+1 open and limit_up via groupby shift (trading-day aligned)
        sorted_limits = limit_df.sort_values(["symbol", "date"])
        sorted_limits["next_open"] = sorted_limits.groupby("symbol")["open"].shift(-1)
        sorted_limits["next_limit_up"] = sorted_limits.groupby("symbol")["limit_up"].shift(-1)
        next_day = sorted_limits[["date", "symbol", "next_open", "next_limit_up"]]
        merged = merged.merge(next_day, on=["date", "symbol"], how="left")
        mask = merged["next_open"] < merged["next_limit_up"] - _LIMIT_EPS
        n_dropped = int((~mask).sum())
        merged = merged[mask].drop(
            columns=["next_open", "next_limit_up"], errors="ignore"
        )

    if n_dropped > 0:
        print(f"  Excluded {n_dropped:,} limit-up rows ({ret_type})")
    return merged


def evaluate(
    factor_id: str,
    start: str,
    end: str,
    *,
    horizons: list[int] | None = None,
    ret_type: str = "close",
    n_groups: int = 10,
    corr_top_k: int = 5,
    exclude_limit_up: bool = True,
    _returns_df: pd.DataFrame | None = None,
    _limit_df: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Evaluate a factor's predictive power.

    Computes IC/RankIC across the requested horizons, turnover, grouped returns,
    and the cross-sectional rank correlation against every other factor in
    ``FactorStorage`` — sorted by ``|corr|`` descending and truncated to
    ``corr_top_k`` rows. Pass ``corr_top_k=0`` to skip the correlation step.
    Use :meth:`EvaluationResult.max_corr` to gate factor admission against
    duplicates.

    Parameters
    ----------
    exclude_limit_up : bool, default True
        For ``ret_type='close'``, drop rows where the signal-day close hits
        limit-up (unbuyable). For ``ret_type='open'``, drop rows where the
        next-day open hits limit-up.
    _returns_df, _limit_df : pd.DataFrame, optional
        Pre-computed forward returns and limit prices (internal optimisation
        for batch evaluation — reuse market data across factors).
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    factor_df, returns_df, limit_df = _load_factor_and_returns(
        factor_id, start, end, horizons, ret_type,
        returns_df=_returns_df, limit_df=_limit_df,
    )

    merged = factor_df.merge(returns_df, on=["date", "symbol"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between factor and returns")

    if exclude_limit_up:
        merged = _exclude_limit_up(merged, limit_df, ret_type)
        if merged.empty:
            raise ValueError("All rows excluded by limit-up filter")

    ic_metrics: dict[int, dict] = {}
    rank_ic_metrics: dict[int, dict] = {}
    decay: dict[int, float] = {}
    group_rets: dict[int, pd.DataFrame] = {}
    ic_series: dict[int, pd.Series] = {}
    rank_ic_series: dict[int, pd.Series] = {}

    for h in horizons:
        ret_col = f"ret_{h}"
        if ret_col not in merged.columns:
            continue

        daily = merged.groupby("date").apply(
            lambda g: pd.Series({
                "ic": _ic_series(g["value"], g[ret_col]),
                "rank_ic": _rank_ic_series(g["value"], g[ret_col]),
            }),
            include_groups=False,
        )

        ic_metrics[h] = _compute_ic_stats(daily["ic"])
        rank_ic_metrics[h] = _compute_ic_stats(daily["rank_ic"])
        ic_series[h] = daily["ic"]
        rank_ic_series[h] = daily["rank_ic"]
        decay[h] = ic_metrics[h].get("ic_mean", np.nan)
        group_rets[h] = _group_returns(merged, ret_col, n_groups)

    turnover = _turnover(factor_df)

    if corr_top_k > 0:
        with FactorStorage() as fs:
            corr_df = _corr_with_existing(factor_df, factor_id, fs, top_k=corr_top_k)
    else:
        corr_df = pd.DataFrame(columns=_CORR_COLUMNS)

    return EvaluationResult(
        factor_id=factor_id,
        ret_type=ret_type,
        horizons=horizons,
        start=start,
        end=end,
        ic_metrics=ic_metrics,
        rank_ic_metrics=rank_ic_metrics,
        decay=decay,
        turnover=turnover,
        group_returns=group_rets,
        corr_with_existing=corr_df,
        ic_series=ic_series,
        rank_ic_series=rank_ic_series,
    )


def _print_comparison_table(results: list[EvaluationResult]) -> None:
    """Print a side-by-side comparison of multiple factors (primary horizon only)."""
    if not results:
        return

    # Pick the primary horizon (first one in the result's horizons list)
    primary_h = results[0].horizons[0] if results[0].horizons else 1

    rows = []
    for r in results:
        ic = r.ic_metrics.get(primary_h, {})
        ric = r.rank_ic_metrics.get(primary_h, {})
        max_corr = r.max_corr()
        rows.append({
            "factor_id": r.factor_id,
            "IC_mean": ic.get("ic_mean"),
            "IC_std": ic.get("ic_std"),
            "ICIR": ic.get("icir"),
            "IC_tstat": ic.get("ic_tstat"),
            "IC+_ratio": ic.get("ic_positive_ratio"),
            "RankIC_mean": ric.get("ic_mean"),
            "RankIC_std": ric.get("ic_std"),
            "RankICIR": ric.get("icir"),
            "RankIC_tstat": ric.get("ic_tstat"),
            "RankIC+_ratio": ric.get("ic_positive_ratio"),
            "turnover": r.turnover,
            "max_corr": max_corr[1] if max_corr else None,
        })

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 100}")
    print(f"Factor Comparison  |  ret_type={results[0].ret_type}  |  horizon={primary_h}d")
    print(f"{'=' * 100}")
    # Round floats for readability
    float_cols = [c for c in df.columns if c != "factor_id"]
    for c in float_cols:
        df[c] = df[c].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    print(df.to_string(index=False))
    print(f"{'=' * 100}\n")


def print_evaluation(result: EvaluationResult) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"Factor Evaluation: {result.factor_id}")
    print(f"Return type: {result.ret_type}")
    print(f"Period: {result.start} ~ {result.end}")
    print(f"Turnover: {result.turnover:.4f}")
    print(f"{'=' * 60}")

    print("\n--- IC / RankIC Summary ---")
    print(result.summary().to_string(index=False))

    print("\n--- Decay (IC mean by horizon) ---")
    for h, ic in sorted(result.decay.items()):
        print(f"  {h:3d}d: {ic:+.4f}")

    print("\n--- Group Returns (top vs bottom) ---")
    for h in result.horizons:
        if h not in result.group_returns:
            continue
        gr = result.group_returns[h]
        if gr.empty:
            continue
        top = gr[gr["group"] == gr["group"].max()]["mean_ret"].values
        bot = gr[gr["group"] == gr["group"].min()]["mean_ret"].values
        spread = top[0] - bot[0] if len(top) > 0 and len(bot) > 0 else np.nan
        print(f"  {h:3d}d: top={top[0]:+.4f}, bot={bot[0]:+.4f}, spread={spread:+.4f}")

    print("\n--- Correlation with existing factors (RankIC, daily mean) ---")
    top = result.max_corr()
    if top is None:
        print("  (no other factors in storage)")
    else:
        print(f"  max |corr|: {top[0]} -> {top[1]:+.4f}")
        print(result.corr_with_existing.to_string(index=False))

    print(f"{'=' * 60}\n")


def plot_evaluation(
    result: EvaluationResult,
    horizon: int = 20,
    output_path: str | None = None,
) -> str:
    """Plot daily IC / RankIC series and cumulative curves for a given horizon.

    Parameters
    ----------
    horizon : int
        Which forward-return horizon to plot (default 20).
    output_path : str, optional
        File path to save the figure.  If None, writes to
        ``results/evaluation/{factor_id}_{horizon}d.png``.

    Returns
    -------
    str
        Path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ic = result.ic_series.get(horizon)
    ric = result.rank_ic_series.get(horizon)
    if ic is None or ric is None:
        raise ValueError(f"No daily series for horizon={horizon}")

    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    fig.suptitle(
        f"{result.factor_id}  |  horizon={horizon}d  |  {result.start}~{result.end}",
        fontsize=14,
        fontweight="bold",
    )

    dates = pd.to_datetime(ic.index)
    cum_ic = ic.fillna(0).cumsum()
    cum_ric = ric.fillna(0).cumsum()

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    # --- daily IC ---
    ax = axes[0]
    colors = ["green" if v >= 0 else "red" for v in ic.values]
    ax.bar(dates, ic.values, color=colors, width=1.5, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(ic.mean(), color="blue", linestyle="--", linewidth=1, label=f"mean={ic.mean():+.4f}")
    ax.set_ylabel("Daily IC")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.set_title("Daily IC")

    # --- daily RankIC ---
    ax = axes[1]
    colors = ["green" if v >= 0 else "red" for v in ric.values]
    ax.bar(dates, ric.values, color=colors, width=1.5, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(ric.mean(), color="blue", linestyle="--", linewidth=1, label=f"mean={ric.mean():+.4f}")
    ax.set_ylabel("Daily RankIC")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.set_title("Daily RankIC")

    # --- cumulative IC ---
    ax = axes[2]
    ax.plot(dates, cum_ic.values, color="steelblue", linewidth=1.2)
    ax.fill_between(dates, cum_ic.values, 0, alpha=0.15, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative IC")
    ax.set_xlabel("Date")
    ax.set_title(f"Cumulative IC  (end={cum_ic.iloc[-1]:+.2f})")

    # --- cumulative RankIC ---
    ax = axes[3]
    ax.plot(dates, cum_ric.values, color="darkorange", linewidth=1.2)
    ax.fill_between(dates, cum_ric.values, 0, alpha=0.15, color="darkorange")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative RankIC")
    ax.set_xlabel("Date")
    ax.set_title(f"Cumulative RankIC  (end={cum_ric.iloc[-1]:+.2f})")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        from pathlib import Path
        out_dir = Path("results/evaluation")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{result.factor_id}_{horizon}d.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate factor predictive power")
    parser.add_argument("factor_id", nargs="?", help="Factor ID to evaluate (e.g. f_001)")
    parser.add_argument("--all", action="store_true", help="Evaluate all registered factors")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument(
        "--horizons",
        default="1,5,10,20,60",
        help="Comma-separated forward return horizons",
    )
    parser.add_argument(
        "--ret-type",
        choices=["close", "open"],
        default="open",
        help="Return calculation type (default: open)",
    )
    parser.add_argument(
        "--corr-top-k",
        type=int,
        default=5,
        help="Number of most-correlated existing factors to report (0 to skip)",
    )
    parser.add_argument(
        "--no-exclude-limit-up",
        action="store_true",
        help="Do NOT exclude limit-up rows from the evaluation",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save IC/RankIC time-series plots (single-factor mode only)",
    )
    parser.add_argument(
        "--plot-horizon",
        type=int,
        default=20,
        help="Horizon to plot (default: 20)",
    )
    args = parser.parse_args()

    if not args.factor_id and not args.all:
        parser.error("Specify a factor_id or --all")

    from backtest.factor.registry import get_registry, list_factors

    if args.all:
        factor_ids = [f["factor_id"] for f in list_factors()]
    else:
        factor_ids = [args.factor_id]

    if not factor_ids:
        print("No factors registered.")
        return

    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    results: list[EvaluationResult] = []

    # Batch mode: preload market data once to avoid N database round-trips
    preloaded_returns: pd.DataFrame | None = None
    preloaded_limit: pd.DataFrame | None = None
    if len(factor_ids) > 1:
        max_h = max(horizons)
        end_dt = pd.to_datetime(args.end, format="%Y%m%d")
        returns_end = (end_dt + pd.Timedelta(days=max_h + 5)).strftime("%Y%m%d")
        try:
            _, preloaded_returns, preloaded_limit = _load_market_data(
                symbols=None,  # all symbols
                start=args.start,
                end=returns_end,
                horizons=horizons,
                ret_type=args.ret_type,
            )
            print(f"  Pre-loaded market data: {len(preloaded_returns):,} return rows")
        except Exception as exc:
            print(f"  Warning: could not pre-load market data ({exc}), falling back to per-factor load")

    for fid in factor_ids:
        try:
            result = evaluate(
                fid,
                args.start,
                args.end,
                horizons=horizons,
                ret_type=args.ret_type,
                corr_top_k=args.corr_top_k,
                exclude_limit_up=not args.no_exclude_limit_up,
                _returns_df=preloaded_returns,
                _limit_df=preloaded_limit,
            )
            results.append(result)
        except Exception as exc:
            print(f"\nERROR evaluating {fid}: {exc}\n")
            continue

    if len(results) == 1:
        print_evaluation(results[0])
        if args.plot:
            try:
                path = plot_evaluation(results[0], horizon=args.plot_horizon)
                print(f"Plot saved: {path}")
            except Exception as exc:
                print(f"Plot error: {exc}")
    elif len(results) > 1:
        _print_comparison_table(results)


if __name__ == "__main__":
    main()
