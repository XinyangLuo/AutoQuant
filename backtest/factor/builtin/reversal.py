"""Reversal factors: recent high returns + high turnover = likely overbought.

Six variants exploring different ways to combine return and turnover:
1. raw      : -ret_20d * ts_mean(turnover_rate, 20)
2. rank_both: rank(-ret_20d) * rank(ts_mean(turnover_rate, 20))
3. rank_prod: rank(-ret_20d * ts_mean(turnover_rate, 20))
4. log_turnover: -ret_20d * log(1 + ts_mean(turnover_rate, 20))
5. zscore_combo: z_score(-ret_20d, 60) * z_score(ts_mean(turnover_rate, 20), 60)
6. pure_reversal: -ret_20d (control, no turnover)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.registry import register
from backtest.factor.transforms import rank, z_score


def _prepare(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
) -> pd.DataFrame:
    """Shared preprocessing: adj_close, ret_Nd, turnover_mean_Nd."""
    df = panel[["date", "symbol", "close", "turnover_rate"]].copy()

    if "adj_factor" in panel.columns:
        df["adj_close"] = df["close"] * panel["adj_factor"]
    else:
        df["adj_close"] = df["close"]

    df = df.sort_values(["symbol", "date"])

    df[f"ret_{ret_window}d"] = df.groupby("symbol")["adj_close"].pct_change(ret_window)

    min_periods = max(turnover_window // 2, 1)
    df[f"turnover_mean_{turnover_window}d"] = (
        df.groupby("symbol")["turnover_rate"]
        .transform(lambda x: x.rolling(turnover_window, min_periods=min_periods).mean())
    )

    return df


# ------------------------------------------------------------------
# Variant 1: raw product
# ------------------------------------------------------------------
@register(
    "f_rev_01",
    name="reversal_raw",
    category="reversal",
    data_sources=["market_daily"],
    description="原始乘积: -ret_20d * ts_mean(turnover_rate, 20)",
    parameters={"ret_window": 20, "turnover_window": 20},
)
def reversal_raw(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
) -> pd.Series:
    df = _prepare(panel, ret_window=ret_window, turnover_window=turnover_window)
    df["factor"] = -df[f"ret_{ret_window}d"] * df[f"turnover_mean_{turnover_window}d"]
    return df.set_index(["date", "symbol"])["factor"]


# ------------------------------------------------------------------
# Variant 2: rank both sides then multiply
# ------------------------------------------------------------------
@register(
    "f_rev_02",
    name="reversal_rank_both",
    category="reversal",
    data_sources=["market_daily"],
    description="双边rank: rank(-ret_20d) * rank(ts_mean(turnover_rate, 20))",
    parameters={"ret_window": 20, "turnover_window": 20},
)
def reversal_rank_both(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
) -> pd.Series:
    df = _prepare(panel, ret_window=ret_window, turnover_window=turnover_window)
    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])

    ret_neg = pd.Series(-df[f"ret_{ret_window}d"].values, index=idx)
    to_mean = pd.Series(df[f"turnover_mean_{turnover_window}d"].values, index=idx)

    df["factor"] = (rank(ret_neg) * rank(to_mean)).values
    return df.set_index(["date", "symbol"])["factor"]


# ------------------------------------------------------------------
# Variant 3: rank the raw product
# ------------------------------------------------------------------
@register(
    "f_rev_03",
    name="reversal_rank_prod",
    category="reversal",
    data_sources=["market_daily"],
    description="先乘后rank: rank(-ret_20d * ts_mean(turnover_rate, 20))",
    parameters={"ret_window": 20, "turnover_window": 20},
)
def reversal_rank_prod(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
) -> pd.Series:
    df = _prepare(panel, ret_window=ret_window, turnover_window=turnover_window)
    raw = pd.Series(
        (-df[f"ret_{ret_window}d"] * df[f"turnover_mean_{turnover_window}d"]).values,
        index=pd.MultiIndex.from_arrays([df["date"], df["symbol"]]),
    )
    df["factor"] = rank(raw).values
    return df.set_index(["date", "symbol"])["factor"]


# ------------------------------------------------------------------
# Variant 4: log-transform turnover to compress tails
# ------------------------------------------------------------------
@register(
    "f_rev_04",
    name="reversal_log_turnover",
    category="reversal",
    data_sources=["market_daily"],
    description="对换手率取log: -ret_20d * log(1 + ts_mean(turnover_rate, 20))",
    parameters={"ret_window": 20, "turnover_window": 20},
)
def reversal_log_turnover(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
) -> pd.Series:
    df = _prepare(panel, ret_window=ret_window, turnover_window=turnover_window)
    df["factor"] = -df[f"ret_{ret_window}d"] * np.log1p(df[f"turnover_mean_{turnover_window}d"])
    return df.set_index(["date", "symbol"])["factor"]


# ------------------------------------------------------------------
# Variant 5: z-score both sides then multiply
# ------------------------------------------------------------------
@register(
    "f_rev_05",
    name="reversal_zscore_combo",
    category="reversal",
    data_sources=["market_daily"],
    description="时序zscore标准化后相乘",
    parameters={"ret_window": 20, "turnover_window": 20, "z_window": 60},
)
def reversal_zscore_combo(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
    z_window: int = 60,
) -> pd.Series:
    df = _prepare(panel, ret_window=ret_window, turnover_window=turnover_window)
    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])

    ret_neg = pd.Series(-df[f"ret_{ret_window}d"].values, index=idx)
    to_mean = pd.Series(df[f"turnover_mean_{turnover_window}d"].values, index=idx)

    min_periods_z = max(z_window // 3, 2)
    df["factor"] = (
        z_score(ret_neg, window=z_window, min_periods=min_periods_z) *
        z_score(to_mean, window=z_window, min_periods=min_periods_z)
    ).values
    return df.set_index(["date", "symbol"])["factor"]


# ------------------------------------------------------------------
# Variant 6: pure reversal (control, no turnover)
# ------------------------------------------------------------------
@register(
    "f_rev_06",
    name="reversal_pure",
    category="reversal",
    data_sources=["market_daily"],
    description="纯反转对照: -ret_20d",
    parameters={"ret_window": 20},
)
def reversal_pure(
    panel: pd.DataFrame,
    ret_window: int = 20,
) -> pd.Series:
    df = _prepare(panel, ret_window=ret_window, turnover_window=ret_window)
    df["factor"] = -df[f"ret_{ret_window}d"]
    return df.set_index(["date", "symbol"])["factor"]
