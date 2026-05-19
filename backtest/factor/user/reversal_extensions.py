"""反转因子扩展家族 — 以 -ret_20 为基础，叠加成交量/换手率/振幅等维度。

设计思路：
- 纯反转 (-ret_20) 在 A 股长期表现不稳定，因为实质押注小盘风格。
- 叠加流动性/情绪维度，试图捕捉"特定条件下的反转"：
  1. 放量大跌 → 恐慌抛售后的反弹 (f_rev_06)
  2. 缩量大跌 → 抛压枯竭后的企稳 (f_rev_07)
  3. 换手异常放大 + 大跌 → 资金博弈后的均值回归 (f_rev_08)
  4. 成交额异常放大 + 大跌 → 大资金换手后的修复 (f_rev_09)
  5. 高振幅 + 大跌 → 日内波动放大后的收敛 (f_rev_10)
  6. 综合版：同时考虑换手和成交偏离 (f_rev_11)
  7. 综合版2：反转 + 低换手 + 低波动 (f_rev_12)

所有因子统一使用截面 rank 标准化（抗厚尾、抗量纲差异），方向均为 desc
（因子值越大 = 越符合该条件 = 越应该买入）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.factor.registry import register
from backtest.factor.transforms import rank, ts_mean, ts_std


def _compute_ret(panel: pd.DataFrame, window: int) -> pd.Series:
    """计算 window 日收益率 (pct_change)，使用复权价。"""
    df = panel[["date", "symbol", "close"]].copy()
    if "adj_factor" in panel.columns:
        df["adj_close"] = df["close"] * panel["adj_factor"]
    else:
        df["adj_close"] = df["close"]

    df = df.sort_values(["symbol", "date"])
    df[f"ret_{window}d"] = df.groupby("symbol")["adj_close"].pct_change(window)

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    return pd.Series(df[f"ret_{window}d"].values, index=idx)


def _compute_ts_mean(panel: pd.DataFrame, field: str, window: int) -> pd.Series:
    """计算某字段的滚动均值。"""
    df = panel[["date", "symbol", field]].copy()
    df = df.sort_values(["symbol", "date"])
    df[f"{field}_mean_{window}d"] = (
        df.groupby("symbol")[field]
        .transform(lambda x: x.rolling(window, min_periods=max(window // 2, 1)).mean())
    )
    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    return pd.Series(df[f"{field}_mean_{window}d"].values, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_06: 反转 + 放量（量比放大）
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_06",
    name="reversal_volume_ratio",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × 量比(volume_ratio)：大跌伴随放量 → 恐慌抛售后的反弹",
    parameters={"ret_window": 20},
)
def reversal_volume_ratio(panel: pd.DataFrame, ret_window: int = 20) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)
    idx = pd.MultiIndex.from_arrays([panel["date"], panel["symbol"]])
    vr = pd.Series(panel["volume_ratio"].values, index=idx)
    # 处理量比异常值
    vr = vr.replace([np.inf, -np.inf], np.nan)
    vr = vr.clip(lower=0.1, upper=10.0)
    return (rank(ret_neg) * rank(vr)).rename("f_rev_06")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_07: 反转 + 缩量（量比缩小）
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_07",
    name="reversal_volume_contraction",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × (-量比)：大跌伴随缩量 → 抛压枯竭后的企稳",
    parameters={"ret_window": 20},
)
def reversal_volume_contraction(panel: pd.DataFrame, ret_window: int = 20) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)
    idx = pd.MultiIndex.from_arrays([panel["date"], panel["symbol"]])
    vr = pd.Series(panel["volume_ratio"].values, index=idx)
    vr = vr.replace([np.inf, -np.inf], np.nan)
    vr = vr.clip(lower=0.1, upper=10.0)
    return (rank(ret_neg) * rank(-vr)).rename("f_rev_07")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_08: 反转 + 换手率偏离（当前换手 vs 20日均值）
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_08",
    name="reversal_turnover_spike",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × 换手率偏离(当前/20日均)：换手异常放大伴随大跌 → 均值回归",
    parameters={"ret_window": 20, "turnover_window": 20},
)
def reversal_turnover_spike(
    panel: pd.DataFrame, ret_window: int = 20, turnover_window: int = 20
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)

    df = panel[["date", "symbol", "turnover_rate"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["to_mean"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(turnover_window, min_periods=max(turnover_window // 2, 1)).mean()
    )
    df["to_dev"] = df["turnover_rate"] / df["to_mean"].replace(0, np.nan)
    df["to_dev"] = df["to_dev"].replace([np.inf, -np.inf], np.nan).clip(upper=5.0)

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    to_dev = pd.Series(df["to_dev"].values, index=idx)
    return (rank(ret_neg) * rank(to_dev)).rename("f_rev_08")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_09: 反转 + 成交额偏离
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_09",
    name="reversal_amount_spike",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × 成交额偏离(当前/20日均)：大资金换手后的修复",
    parameters={"ret_window": 20, "amount_window": 20},
)
def reversal_amount_spike(
    panel: pd.DataFrame, ret_window: int = 20, amount_window: int = 20
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)

    df = panel[["date", "symbol", "amount"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["amt_mean"] = df.groupby("symbol")["amount"].transform(
        lambda x: x.rolling(amount_window, min_periods=max(amount_window // 2, 1)).mean()
    )
    df["amt_dev"] = df["amount"] / df["amt_mean"].replace(0, np.nan)
    df["amt_dev"] = df["amt_dev"].replace([np.inf, -np.inf], np.nan).clip(upper=5.0)

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    amt_dev = pd.Series(df["amt_dev"].values, index=idx)
    return (rank(ret_neg) * rank(amt_dev)).rename("f_rev_09")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_10: 反转 + 日内振幅
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_10",
    name="reversal_intraday_range",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × 日内振幅((high-low)/close)：高波动伴随大跌后的收敛",
    parameters={"ret_window": 20},
)
def reversal_intraday_range(panel: pd.DataFrame, ret_window: int = 20) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)

    idx = pd.MultiIndex.from_arrays([panel["date"], panel["symbol"]])
    hl = panel["high"] - panel["low"]
    close = panel["close"]
    amplitude = (hl / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    amplitude = amplitude.clip(upper=0.5)

    amp = pd.Series(amplitude.values, index=idx)
    return (rank(ret_neg) * rank(amp)).rename("f_rev_10")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_11: 综合版 — 反转 × 换手偏离 × 成交偏离
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_11",
    name="reversal_liquidity_combo",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × 换手偏离 × 成交偏离：多重流动性异常的复合反转",
    parameters={"ret_window": 20, "liquidity_window": 20},
)
def reversal_liquidity_combo(
    panel: pd.DataFrame, ret_window: int = 20, liquidity_window: int = 20
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)

    df = panel[["date", "symbol", "turnover_rate", "amount"]].copy()
    df = df.sort_values(["symbol", "date"])

    df["to_mean"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(liquidity_window, min_periods=max(liquidity_window // 2, 1)).mean()
    )
    df["to_dev"] = (df["turnover_rate"] / df["to_mean"].replace(0, np.nan)).clip(upper=5.0)

    df["amt_mean"] = df.groupby("symbol")["amount"].transform(
        lambda x: x.rolling(liquidity_window, min_periods=max(liquidity_window // 2, 1)).mean()
    )
    df["amt_dev"] = (df["amount"] / df["amt_mean"].replace(0, np.nan)).clip(upper=5.0)

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    to_dev = pd.Series(df["to_dev"].values, index=idx).replace([np.inf, -np.inf], np.nan)
    amt_dev = pd.Series(df["amt_dev"].values, index=idx).replace([np.inf, -np.inf], np.nan)

    return (rank(ret_neg) * rank(to_dev) * rank(amt_dev)).rename("f_rev_11")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_12: 反转 + 低换手 + 低波动（与 f_rev_11 相反：找"安静"的下跌）
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_12",
    name="reversal_quiet_decline",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × 低换手 × 低振幅：安静下跌后的价值修复",
    parameters={"ret_window": 20, "liquidity_window": 20},
)
def reversal_quiet_decline(
    panel: pd.DataFrame, ret_window: int = 20, liquidity_window: int = 20
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)

    df = panel[["date", "symbol", "turnover_rate", "high", "low", "close"]].copy()
    df = df.sort_values(["symbol", "date"])

    df["to_mean"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(liquidity_window, min_periods=max(liquidity_window // 2, 1)).mean()
    )
    df["to_ratio"] = (df["turnover_rate"] / df["to_mean"].replace(0, np.nan)).clip(upper=5.0)

    amplitude = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).clip(upper=0.5)
    df["amp"] = amplitude

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    to_ratio = pd.Series(df["to_ratio"].values, index=idx).replace([np.inf, -np.inf], np.nan)
    amp = pd.Series(df["amp"].values, index=idx).replace([np.inf, -np.inf], np.nan)

    # 低换手、低振幅 → 用负号
    return (rank(ret_neg) * rank(-to_ratio) * rank(-amp)).rename("f_rev_12")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_13: 反转 + 换手偏离 —— 加法组合 + 时序平滑
# 针对 f_rev_08 乘法版换手过高(fees 104%)的问题:
# 1. ret_neg 做 10 日移动平均,截面排名变化更慢
# 2. 加法(0.6 ret + 0.4 to_dev)替代乘法,避免极端值导致的频繁调仓
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_13",
    name="reversal_turnover_additive",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 + 换手偏离:加法组合+时序平滑,降低换手和波动",
    parameters={"ret_window": 20, "turnover_window": 20, "smooth_window": 10},
)
def reversal_turnover_additive(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
    smooth_window: int = 10,
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)
    ret_neg_smooth = ts_mean(ret_neg, window=smooth_window)

    df = panel[["date", "symbol", "turnover_rate"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["to_mean"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(turnover_window, min_periods=max(turnover_window // 2, 1)).mean()
    )
    df["to_dev"] = (df["turnover_rate"] / df["to_mean"].replace(0, np.nan)).clip(upper=5.0)

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    to_dev = pd.Series(df["to_dev"].values, index=idx).replace([np.inf, -np.inf], np.nan)

    return (0.6 * rank(ret_neg_smooth) + 0.4 * rank(to_dev)).rename("f_rev_13")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_14: 反转 + 高波动 + 换手放大 —— 多维度加法
# 核心逻辑:高波动 + 大跌 + 换手异常放大 = 情绪驱动的下跌,反弹概率更高
# 叠加日内振幅和收益波动两个维度,期望改善 simple Sharpe
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_14",
    name="reversal_volatility_turnover",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 + 日内振幅 + 换手偏离 + 收益波动:捕捉情绪驱动型反转",
    parameters={"ret_window": 20, "turnover_window": 20, "vol_window": 20},
)
def reversal_volatility_turnover(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
    vol_window: int = 20,
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)

    idx = pd.MultiIndex.from_arrays([panel["date"], panel["symbol"]])

    amplitude = (
        ((panel["high"] - panel["low"]) / panel["close"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .clip(upper=0.5)
    )
    amp = pd.Series(amplitude.values, index=idx)

    df = panel[["date", "symbol", "turnover_rate"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["to_mean"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(turnover_window, min_periods=max(turnover_window // 2, 1)).mean()
    )
    df["to_dev"] = (df["turnover_rate"] / df["to_mean"].replace(0, np.nan)).clip(upper=5.0)
    idx_to = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    to_dev = pd.Series(df["to_dev"].values, index=idx_to).replace([np.inf, -np.inf], np.nan)

    daily_ret = _compute_ret(panel, 1)
    ret_vol = ts_std(daily_ret, window=vol_window, min_periods=max(vol_window // 2, 1))

    return (
        0.35 * rank(ret_neg)
        + 0.25 * rank(amp)
        + 0.25 * rank(to_dev)
        + 0.15 * rank(ret_vol)
    ).rename("f_rev_14")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_15: 反转 × 换手偏离 + 大市值倾斜
# 动机:
#   f_rev_08 (rank(ret_neg) * rank(to_dev)) 静态 IC 极强 (RankICIR 0.82),
#   但 2017-2018 detailed -13% / -28% 暴露明显的小盘风险敞口 — 换手率偏离
#   信号在小市值股票上结构性放大,流动性紧缩期亏损被放大。
# 改进:
#   在 f_rev_08 的乘法基础上,**叠加** rank(log(circ_mv)) 做大市值倾斜。
#   核心信号 (ret_neg × to_dev) 维持原状,加性引入 size tilt 在不破坏信号
#   单调性的前提下下移小盘 weight。
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_15",
    name="reversal_turnover_size_tilt",
    category="reversal",
    data_sources=["market_daily"],
    description="f_rev_08 核心信号 (rank(ret_neg) * rank(to_dev)) + rank(log(circ_mv)) 大市值倾斜",
    parameters={"ret_window": 20, "turnover_window": 20, "size_weight": 0.25},
)
def reversal_turnover_size_tilt(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
    size_weight: float = 0.25,
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)

    df = panel[["date", "symbol", "turnover_rate", "circ_mv"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["to_mean"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(turnover_window, min_periods=max(turnover_window // 2, 1)).mean()
    )
    df["to_dev"] = (df["turnover_rate"] / df["to_mean"].replace(0, np.nan)).clip(upper=5.0)
    df["log_cap"] = np.log(df["circ_mv"].clip(lower=1.0))

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    to_dev = pd.Series(df["to_dev"].values, index=idx).replace([np.inf, -np.inf], np.nan)
    log_cap = pd.Series(df["log_cap"].values, index=idx).replace([np.inf, -np.inf], np.nan)

    core = rank(ret_neg) * rank(to_dev)
    return ((1.0 - size_weight) * core + size_weight * rank(log_cap)).rename("f_rev_15")


# ─────────────────────────────────────────────────────────────────────────────
# f_rev_16: 短期反转 × 换手放大 × 长期上行 (avoid falling knives)
# 动机:
#   f_rev_08 在 2017-2018 deleveraging 期亏损巨大,因为 short-term reversal +
#   turnover spike 抓到的多是"持续下跌+恐慌抛售"的弱势股 (falling knives),
#   不是真正的"短期超跌+长期健康"的反转候选。
# 改进:
#   在原信号上 *乘以* rank(60d return) — 只有当 60d 还相对强 / 没崩盘时,
#   20d 反转 + 换手放大 才是高质量的均值回归 setup。
#   这等价于在因子内部做了 trend-quality 过滤。
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "f_rev_16",
    name="reversal_turnover_trend_filter",
    category="reversal",
    data_sources=["market_daily"],
    description="20日反转 × 换手偏离 × 60日趋势 (avoid falling knives)",
    parameters={"ret_window": 20, "turnover_window": 20, "trend_window": 60},
)
def reversal_turnover_trend_filter(
    panel: pd.DataFrame,
    ret_window: int = 20,
    turnover_window: int = 20,
    trend_window: int = 60,
) -> pd.Series:
    ret_neg = -_compute_ret(panel, ret_window)
    ret_trend = _compute_ret(panel, trend_window)

    df = panel[["date", "symbol", "turnover_rate"]].copy()
    df = df.sort_values(["symbol", "date"])
    df["to_mean"] = df.groupby("symbol")["turnover_rate"].transform(
        lambda x: x.rolling(turnover_window, min_periods=max(turnover_window // 2, 1)).mean()
    )
    df["to_dev"] = (df["turnover_rate"] / df["to_mean"].replace(0, np.nan)).clip(upper=5.0)

    idx = pd.MultiIndex.from_arrays([df["date"], df["symbol"]])
    to_dev = pd.Series(df["to_dev"].values, index=idx).replace([np.inf, -np.inf], np.nan)

    return (rank(ret_neg) * rank(to_dev) * rank(ret_trend)).rename("f_rev_16")
