"""Signal formatter: normalize strategy output to engine input format."""

from __future__ import annotations

import pandas as pd


def format_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a signals DataFrame.

    Ensures columns are ``[date, symbol, target_weight]`` with correct dtypes.
    This is the bridge between strategy output and engine input.

    Parameters
    ----------
    signals : pd.DataFrame
        Raw strategy output. Must contain ``date``, ``symbol``, ``target_weight``.

    Returns
    -------
    pd.DataFrame
        Normalized signals with sorted rows.
    """
    required = {"date", "symbol", "target_weight"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Signals DataFrame missing columns: {missing}")

    df = signals[["date", "symbol", "target_weight"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce")
    df = df.dropna(subset=["target_weight"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def group_by_date(signals: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Group signals by effective date, returning a dict of (date → sub-DataFrame).

    Useful for the engine to process one rebalance day at a time.
    """
    formatted = format_signals(signals)
    groups = {}
    for date, g in formatted.groupby("date"):
        groups[date] = g[["symbol", "target_weight"]].set_index("symbol")
    return groups


def normalize_weights(
    weights: pd.Series,
    long_sum: float = 1.0,
    short_sum: float | None = None,
) -> pd.Series:
    """Normalize weights so that long positions sum to ``long_sum``.

    If ``short_sum`` is provided, negative weights are normalized separately
    to sum to ``-abs(short_sum)``.

    Parameters
    ----------
    weights : pd.Series
        Index = symbol, values = raw weights.
    long_sum : float
        Target sum for positive weights.
    short_sum : float | None
        Target absolute sum for negative weights.

    Returns
    -------
    pd.Series
        Normalized weights.
    """
    pos = weights[weights > 0]
    neg = weights[weights < 0]

    result = weights.astype(float).copy()

    if len(pos) > 0 and pos.sum() > 0:
        result[pos.index] = pos / pos.sum() * long_sum

    if short_sum is not None and len(neg) > 0 and neg.sum() < 0:
        result[neg.index] = neg / abs(neg.sum()) * short_sum

    return result
