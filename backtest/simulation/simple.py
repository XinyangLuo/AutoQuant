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
        # 1. 计算复权价格并 pivot 成 wide 收益矩阵 R[T, N]。
        required = {"date", "symbol", "close", "adj_factor"}
        validate_columns(market_data, required, label="market_data")
        df = market_data.copy()
        df["adj_price"] = compute_adj_price(df, self.config.price_type)
        adj_price_wide = df.pivot(index="date", columns="symbol", values="adj_price")
        returns_wide = adj_price_wide.shift(-1) / adj_price_wide - 1.0

        # 2. 构建 date/symbol → 整数索引映射。
        all_dates = returns_wide.index
        all_symbols = returns_wide.columns
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
        sym_to_idx = {s: i for i, s in enumerate(all_symbols)}
        R = returns_wide.values  # [T, N] numpy array
        T = len(all_dates)

        # 3. 将 signals 映射为整数索引数组。
        s_dates = signals["date"].values
        s_symbols = signals["symbol"].values
        s_weights = signals["target_weight"].values

        t_idx = np.array([date_to_idx.get(pd.Timestamp(d), -1) for d in s_dates],
                         dtype=np.intp)
        s_idx = np.array([sym_to_idx.get(s, -1) for s in s_symbols],
                         dtype=np.intp)

        valid = (t_idx >= 0) & (s_idx >= 0)
        if not valid.any():
            nav_df = pd.DataFrame({
                "date": list(all_dates),
                "nav": 1.0,
                "daily_return": 0.0,
            })
            return BacktestResult(nav_df=nav_df, initial_cash=self.config.initial_cash)

        t_idx = t_idx[valid]
        s_idx = s_idx[valid]
        w = s_weights[valid].astype(float)

        # 4. 稀疏点积：只对有效 (date, symbol) 取值。
        #    Sanitise both weight and return — a single NaN in w×r
        #    cascades through bincount → cumprod → entire NAV.
        r = R[t_idx, s_idx]
        np.nan_to_num(r, copy=False, nan=0.0)
        np.nan_to_num(w, copy=False, nan=0.0)

        # 5. 按日累加 daily_return[t] = Σ w × r。
        daily_return = np.bincount(t_idx, weights=w * r, minlength=T).astype(float)

        # 6. signals 的 date 是持仓生效日，returns_wide[t] 是 t -> t+1。
        #    净值行 date=t 应代表 t 日收盘后的组合状态，因此 t -> t+1
        #    的收益要体现在下一条净值记录上，首日 NAV 固定为 1.0。
        dr_series = pd.Series(daily_return, index=all_dates)
        realised_return = dr_series.shift(1).fillna(0.0)
        nav_series = cumulate_nav(realised_return)

        nav_df = pd.DataFrame({
            "date": list(all_dates),
            "nav": nav_series.values,
            "daily_return": realised_return.values,
        })

        return BacktestResult(nav_df=nav_df, initial_cash=self.config.initial_cash)
