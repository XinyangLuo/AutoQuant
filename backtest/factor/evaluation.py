"""Offline factor evaluation: IC, RankIC, ICIR, turnover, decay, group returns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.storage import FactorStorage


DEFAULT_HORIZONS = [1, 5, 10, 20, 60]
_CORR_COLUMNS = ["factor_id", "corr", "n_dates"]


def _load_factor_and_returns(
    factor_id: str,
    start: str,
    end: str,
    horizons: list[int],
    ret_type: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load factor values, compute forward returns, and load limit prices for filtering."""
    with FactorStorage() as fs:
        factor_df = fs.get_factor(factor_id, start, end)

    if factor_df.empty:
        raise ValueError(f"No factor data for {factor_id} in range {start}~{end}")

    max_h = max(horizons)
    factor_end = factor_df["date"].max()
    returns_end = (factor_end + pd.Timedelta(days=max_h + 5)).strftime("%Y%m%d")

    with MarketStorage() as ms:
        symbols = factor_df["symbol"].unique().tolist()
        market_df = ms.get_bars(
            symbols=symbols,
            start=start,
            end=returns_end,
        )

    if market_df.empty:
        raise ValueError("No market data available for return calculation")

    # Compute forward returns for all horizons in one pass
    returns_df = _compute_forward_returns(market_df, horizons, ret_type)

    # Extract limit prices for limit-up filtering
    limit_cols = ["date", "symbol", "close", "open", "limit_up"]
    if all(c in market_df.columns for c in limit_cols):
        limit_df = market_df[limit_cols].copy()
    else:
        limit_df = pd.DataFrame(columns=limit_cols)

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
    icir = mean / std * np.sqrt(252) if std != 0 else np.nan
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
    """Average daily cross-sectional rank correlation with every other factor.

    Used to flag near-duplicate factors at evaluation time — if the maximum
    absolute correlation is above ~0.9, the new factor probably duplicates an
    existing one and shouldn't be admitted to the library. Pass ``top_k=0``
    to skip the comparison entirely.
    """
    if top_k <= 0 or factor_df.empty:
        return pd.DataFrame(columns=_CORR_COLUMNS)

    start = factor_df["date"].min().strftime("%Y%m%d")
    end = factor_df["date"].max().strftime("%Y%m%d")
    others = storage.get_factors_long(start=start, end=end, exclude=factor_id)
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
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    factor_df, returns_df, limit_df = _load_factor_and_returns(
        factor_id, start, end, horizons, ret_type
    )

    merged = factor_df.merge(returns_df, on=["date", "symbol"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between factor and returns")

    if exclude_limit_up:
        merged = _exclude_limit_up(merged, limit_df, ret_type)
        if merged.empty:
            raise ValueError("All rows excluded by limit-up filter")

    ic_metrics = {}
    rank_ic_metrics = {}
    decay = {}
    group_rets = {}

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
    )


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate factor predictive power")
    parser.add_argument("factor_id", help="Factor ID to evaluate (e.g. f_001)")
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
        default="close",
        help="Return calculation type",
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
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    result = evaluate(
        args.factor_id,
        args.start,
        args.end,
        horizons=horizons,
        ret_type=args.ret_type,
        corr_top_k=args.corr_top_k,
        exclude_limit_up=not args.no_exclude_limit_up,
    )
    print_evaluation(result)


if __name__ == "__main__":
    main()
