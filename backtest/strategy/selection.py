"""Shared stock selection and signal-building utilities."""

from __future__ import annotations

import pandas as pd

from backtest.strategy.config import SelectionConfig, WeightingConfig
from backtest.strategy.signals import normalize_weights
from backtest.strategy.weight import WeightAllocator


def build_signals(
    date: pd.Timestamp,
    sorted_scores: pd.Series,
    filtered_df: pd.DataFrame,
    selection: SelectionConfig,
    weighting: WeightingConfig,
    factor_col: str | None = None,
) -> list[dict]:
    """Build signal rows for a single rebalancing date from sorted scores.

    Parameters
    ----------
    date : pd.Timestamp
        Rebalancing date.
    sorted_scores : pd.Series
        Index = symbol, values = score (already sorted best-to-worst).
    filtered_df : pd.DataFrame
        Full filtered panel for this date (contains symbol, circ_mv, etc.).
    selection : SelectionConfig
    weighting : WeightingConfig
    factor_col : str | None
        Factor column name for factor-value weighting.

    Returns
    -------
    list[dict]
        Signal rows, each a dict with date, symbol, target_weight, and
        optionally decile_group.
    """
    allocator = WeightAllocator(weighting)
    rows: list[dict] = []
    method = selection.method

    if method == "topk":
        selected = sorted_scores.head(selection.top_k)
        selected_df = filtered_df[filtered_df["symbol"].isin(selected.index)]
        weights = allocator.allocate(selected_df, factor_col=factor_col)
        weights = normalize_weights(weights, long_sum=1.0)
        for sym, w in weights.items():
            rows.append({"date": date, "symbol": sym, "target_weight": w})
        return rows

    if method == "long_short":
        longs = sorted_scores.head(selection.top_k)
        shorts = sorted_scores.tail(selection.bottom_k)

        long_df = filtered_df[filtered_df["symbol"].isin(longs.index)]
        long_weights = allocator.allocate(long_df, factor_col=factor_col)
        long_weights = normalize_weights(long_weights, long_sum=0.5)
        for sym, w in long_weights.items():
            rows.append({"date": date, "symbol": sym, "target_weight": w})

        short_df = filtered_df[filtered_df["symbol"].isin(shorts.index)]
        short_weights = allocator.allocate(short_df, factor_col=factor_col)
        short_weights = normalize_weights(short_weights, long_sum=0.5)
        for sym, w in short_weights.items():
            rows.append({"date": date, "symbol": sym, "target_weight": -w})
        return rows

    if method == "decile":
        n = len(sorted_scores)
        if n < 10:
            return rows
        decile_labels = pd.qcut(range(n), 10, labels=False, duplicates="drop")
        decile_series = pd.Series(decile_labels, index=sorted_scores.index)

        target_group = selection.decile_group
        if target_group is not None:
            group_symbols = decile_series[decile_series == target_group].index
            group_df = filtered_df[filtered_df["symbol"].isin(group_symbols)]
            weights = allocator.allocate(group_df, factor_col=factor_col)
            weights = normalize_weights(weights, long_sum=1.0)
            for sym, w in weights.items():
                rows.append({"date": date, "symbol": sym, "target_weight": w})
            return rows

        # Return all deciles (for analysis)
        for group_id in range(10):
            group_symbols = decile_series[decile_series == group_id].index
            group_df = filtered_df[filtered_df["symbol"].isin(group_symbols)]
            weights = allocator.allocate(group_df, factor_col=factor_col)
            weights = normalize_weights(weights, long_sum=1.0)
            for sym, w in weights.items():
                rows.append({
                    "date": date,
                    "symbol": sym,
                    "target_weight": w,
                    "decile_group": group_id,
                })
        return rows

    raise ValueError(f"Unknown selection method: {method}")
