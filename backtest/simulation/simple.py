from __future__ import annotations

import pandas as pd
import numpy as np

from backtest.simulation.config import SimulationConfig
from backtest.simulation.models import BacktestResult
from backtest.simulation.utils import compute_adj_price, cumulate_nav, validate_columns


class SimpleSimulator:
    """向量化快速回测。使用复权价格，不模拟金额，直接 weight × return 算净值。

    使用稀疏点积替代密集矩阵乘法：不将权重 pivot 成密集 wide 矩阵，
    而是通过 numpy 高级索引直接取出对应 (date, symbol) 的收益，
    用 ``bincount`` 按日累加。避免了 98% 的零乘法。
    """

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
            [date, symbol, target_weight] — 日频，已含 delay=1 和 ffill。
        market_data : pd.DataFrame
            [date, symbol, close, adj_factor, ...]

        Returns
        -------
        BacktestResult
            只有 nav_df，trades/snapshots 为空
        """
        # 1. 计算长表 forward return，避免 materialise dense date × symbol 矩阵。
        required = {"date", "symbol", "close", "adj_factor"}
        validate_columns(market_data, required, label="market_data")
        df = market_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["adj_price"] = compute_adj_price(df, self.config.price_type)
        df = df.sort_values(["symbol", "date"])
        df["forward_return"] = (
            df.groupby("symbol", sort=False)["adj_price"].shift(-1)
            / df["adj_price"]
            - 1.0
        )
        returns_long = df[["date", "symbol", "forward_return"]]

        all_dates = pd.Index(sorted(df["date"].drop_duplicates()))
        if all_dates.empty:
            return BacktestResult(initial_cash=self.config.initial_cash)

        # 2. 将 signals 与长表收益按 (date, symbol) 对齐。缺失行情/末日收益
        #    与旧 sparse dense-matrix 路径一致，按 0 处理。
        sig = signals[["date", "symbol", "target_weight"]].copy()
        sig["date"] = pd.to_datetime(sig["date"])
        merged = sig.merge(returns_long, on=["date", "symbol"], how="left")
        if merged.empty:
            nav_df = pd.DataFrame({
                "date": list(all_dates),
                "nav": 1.0,
                "daily_return": 0.0,
            })
            return BacktestResult(nav_df=nav_df, initial_cash=self.config.initial_cash)

        w = merged["target_weight"].to_numpy(dtype=float)
        r = merged["forward_return"].to_numpy(dtype=float)
        np.nan_to_num(w, copy=False, nan=0.0)
        np.nan_to_num(r, copy=False, nan=0.0)
        merged["weighted_return"] = w * r

        daily_return = (
            merged.groupby("date", sort=False)["weighted_return"]
            .sum()
            .reindex(all_dates, fill_value=0.0)
            .astype(float)
        )

        # 3. signals 的 date 是持仓生效日，forward_return 是 t -> t+1。
        #    净值行 date=t 应代表 t 日收盘后的组合状态，因此 t -> t+1
        #    的收益要体现在下一条净值记录上，首日 NAV 固定为 1.0。
        realised_return = daily_return.shift(1).fillna(0.0)
        nav_series = cumulate_nav(realised_return)

        nav_df = pd.DataFrame({
            "date": list(all_dates),
            "nav": nav_series.values,
            "daily_return": realised_return.values,
        })

        return BacktestResult(nav_df=nav_df, initial_cash=self.config.initial_cash)
