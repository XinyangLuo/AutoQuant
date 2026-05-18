"""f_rev_05 — reversal × turnover, both z-scored before the product.

idea : recent winners with abnormally high turnover tend to mean-revert.
form : z(-ret_20d, 60) * z(ts_mean(turnover_rate, 20), 60)

Both legs are time-series z-scored over a 60-day window so the product is
comparable across stocks of very different volatility / float profiles.
"""

from __future__ import annotations

import pandas as pd

from backtest.factor.registry import register
from backtest.factor.transforms import z_score


@register(
    "f_rev_05",
    name="reversal_zscore_combo",
    category="reversal",
    data_sources=["market_daily"],
    description="时序zscore标准化后相乘的反转×换手率因子",
    parameters={"ret_window": 20, "turnover_window": 20, "z_window": 60},
)
def reversal_zscore_combo(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
    z_window: int = 60,
) -> pd.Series:
    df = panel[["date", "symbol", "close", "turnover_rate"]].copy()

    if "adj_factor" in panel.columns:
        df["adj_close"] = df["close"] * panel["adj_factor"]
    else:
        df["adj_close"] = df["close"]

    df = df.sort_values(["symbol", "date"])
    df[f"ret_{ret_window}d"] = df.groupby("symbol")["adj_close"].pct_change(ret_window)

    min_periods_to = max(turnover_window // 2, 1)
    df[f"turnover_mean_{turnover_window}d"] = (
        df.groupby("symbol")["turnover_rate"]
        .transform(lambda x: x.rolling(turnover_window, min_periods=min_periods_to).mean())
    )

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    ret_neg = pd.Series(-df[f"ret_{ret_window}d"].values, index=idx)
    to_mean = pd.Series(df[f"turnover_mean_{turnover_window}d"].values, index=idx)

    min_periods_z = max(z_window // 3, 2)
    factor = (
        z_score(ret_neg, window=z_window, min_periods=min_periods_z)
        * z_score(to_mean, window=z_window, min_periods=min_periods_z)
    )
    return factor.rename("f_rev_05")
