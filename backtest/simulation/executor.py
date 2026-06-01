from __future__ import annotations

from typing import Literal

import pandas as pd


EPS = 0.01  # 涨停/跌停判断容差（元）


class OrderExecutor:
    """判断单笔订单能否成交，计算成交价格和费用。"""

    def __init__(
        self,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.001,
        transfer_fee_rate: float = 0.00002,
        price_type: Literal["o2o", "c2c"] = "o2o",
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.price_type = price_type

    def can_trade(
        self,
        symbol: str,
        direction: Literal["buy", "sell", "short", "cover"],
        bar: pd.Series,
    ) -> tuple[bool, float | None, str]:
        """返回 (能否交易, 成交价格, 原因)。

        停牌检测由调用方在 engine 主循环中通过"有无数据"判断，
        此处 bar 来自有数据的股票。
        """
        if self.price_type == "o2o":
            return self._can_trade_o2o(symbol, direction, bar)
        return self._can_trade_c2c(symbol, direction, bar)

    def _can_trade_o2o(
        self,
        symbol: str,
        direction: Literal["buy", "sell", "short", "cover"],
        bar: pd.Series,
    ) -> tuple[bool, float | None, str]:
        open_p = float(bar["open"])
        high_p = float(bar["high"])
        low_p = float(bar["low"])
        limit_up = float(bar["limit_up"])
        limit_down = float(bar["limit_down"])

        if direction in ("buy", "cover"):
            if abs(open_p - limit_up) <= EPS:
                if low_p < limit_up - EPS:
                    return True, limit_up, "limit_up_traded"
                return False, None, "limit_up_blocked"
            return True, open_p, "normal"

        if direction in ("sell", "short"):
            if abs(open_p - limit_down) <= EPS:
                if high_p > limit_down + EPS:
                    return True, limit_down, "limit_down_traded"
                return False, None, "limit_down_blocked"
            return True, open_p, "normal"

        return False, None, "unknown_direction"

    def _can_trade_c2c(
        self,
        symbol: str,
        direction: Literal["buy", "sell", "short", "cover"],
        bar: pd.Series,
    ) -> tuple[bool, float | None, str]:
        close_p = float(bar["close"])
        limit_up = float(bar["limit_up"])
        limit_down = float(bar["limit_down"])

        if direction in ("buy", "cover"):
            if abs(close_p - limit_up) <= EPS:
                return False, None, "limit_up_blocked"
            return True, close_p, "normal"

        if direction in ("sell", "short"):
            if abs(close_p - limit_down) <= EPS:
                return False, None, "limit_down_blocked"
            return True, close_p, "normal"

        return False, None, "unknown_direction"

    def calculate_cost(self, amount: float, direction: str) -> float:
        """计算总费用 = 佣金 + 印花税(仅卖出/short) + 过户费。"""
        commission, stamp_duty, transfer_fee = self.calculate_cost_breakdown(
            amount, direction
        )
        return commission + stamp_duty + transfer_fee

    def calculate_cost_breakdown(
        self, amount: float, direction: str
    ) -> tuple[float, float, float]:
        """拆分计算费用，返回 (佣金, 印花税, 过户费)。

        佣金 = max(amount * commission_rate, min_commission)
        印花税 = amount * stamp_duty_rate（仅卖出/short）
        过户费 = amount * transfer_fee_rate（双向）
        """
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate if direction in ("sell", "short") else 0.0
        transfer_fee = amount * self.transfer_fee_rate
        return commission, stamp_duty, transfer_fee
