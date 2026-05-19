from __future__ import annotations

import pandas as pd
import numpy as np

from backtest.simulation.config import SimulationConfig
from backtest.simulation.models import BacktestResult
from backtest.simulation.utils import compute_adj_price, cumulate_nav


class SimpleSimulator:
    """向量化快速回测。使用复权价格，不模拟金额，直接 weight × return 算净值。"""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()

    def run(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> BacktestResult:
        """向量化快速回测。

        Parameters
        ----------
        signals : pd.DataFrame
            [date, symbol, target_weight]
        market_data : pd.DataFrame
            [date, symbol, close, adj_factor, ...]

        Returns
        -------
        BacktestResult
            只有 nav_df，trades/snapshots 为空
        """
        # 1. 计算复权价格（按配置选择 o2o/c2c）
        df = market_data.copy()
        df["adj_price"] = compute_adj_price(df, self.config.price_type)

        # 2. 将数据 pivot 成 wide
        adj_price_wide = df.pivot(index="date", columns="symbol", values="adj_price")
        returns_wide = adj_price_wide.pct_change()

        # 3. 将 signals pivot 成 wide，对齐日期/股票
        weight_wide = signals.pivot(index="date", columns="symbol", values="target_weight")
        # delay=1: T 日信号 → T+1 日生效，weight 向后错一天配 returns
        weight_wide = weight_wide.shift(1)
        weight_wide = weight_wide.reindex_like(returns_wide)
        # 调仓日：不在持仓中的股票 weight 显式置为 0
        signal_dates = set(signals["date"])
        mask = weight_wide.index.isin(signal_dates)
        if mask.any():
            weight_wide.loc[mask] = weight_wide.loc[mask].fillna(0)
        # 非调仓日前向填充保持上期权重
        weight_wide = weight_wide.ffill()
        # 首次调仓前 weight = 0
        weight_wide = weight_wide.fillna(0)

        # 4. 组合日收益 = sum(weight * return)
        # 对缺失数据（停牌/退市），returns_wide 为 NaN，weight 可能非零
        # fillna(0) 后该股票当天收益为 0，不影响组合
        daily_return = (weight_wide * returns_wide.fillna(0)).sum(axis=1)

        # 5. 累积净值
        nav = cumulate_nav(daily_return)

        nav_df = pd.DataFrame({
            "date": nav.index,
            "nav": nav.values,
            "daily_return": daily_return.values,
        }).reset_index(drop=True)

        return BacktestResult(nav_df=nav_df, initial_cash=self.config.initial_cash)
