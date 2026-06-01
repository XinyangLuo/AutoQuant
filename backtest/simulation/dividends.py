from __future__ import annotations

from datetime import date as Date
from typing import TYPE_CHECKING

import pandas as pd

from backtest.simulation.utils import round_lot_for_symbol

if TYPE_CHECKING:
    from backtest.simulation.models import DailySnapshot


class DividendHandler:
    """处理分红送转事件对 portfolio 的影响。"""

    def apply(
        self,
        date: Date,
        snapshot: "DailySnapshot",
        dividends: pd.DataFrame | None,
    ) -> list[dict]:
        """对给定日期，查找所有影响 portfolio 的分红事件并应用。

        Parameters
        ----------
        date : Date
            当前日期
        snapshot : DailySnapshot
            当前 portfolio 快照（会被原地修改 cash 和 positions）
        dividends : pd.DataFrame | None
            分红数据 [symbol, ex_date, pay_date, cash_div, stk_div]

        Returns
        -------
        list[dict]
            事件列表，供日志记录
        """
        events: list[dict] = []
        if dividends is None or dividends.empty:
            return events

        # 送转股：ex_date 当天生效
        ex_mask = dividends["ex_date"] == date.strftime("%Y%m%d")
        if ex_mask.any():
            for _, row in dividends[ex_mask].iterrows():
                symbol = row["symbol"]
                if symbol not in snapshot.positions:
                    continue
                stk_div = float(row.get("stk_div", 0) or 0)
                if stk_div <= 0:
                    continue
                pos = snapshot.positions[symbol]
                # 送转股产生的股数保留精确值（不取整），仅交易时按板块规则取整
                new_shares = int(pos.shares * (1 + stk_div))
                if new_shares != pos.shares:
                    events.append({
                        "date": date,
                        "symbol": symbol,
                        "type": "stk_div",
                        "old_shares": pos.shares,
                        "new_shares": new_shares,
                        "stk_div": stk_div,
                    })
                    # 同步调整持仓成本：送转后每股成本稀释
                    if pos.avg_cost > 0:
                        pos.avg_cost = pos.avg_cost / (1 + stk_div)
                    pos.shares = new_shares

        # 现金分红：pay_date 当天到账
        pay_mask = dividends["pay_date"] == date.strftime("%Y%m%d")
        if pay_mask.any():
            for _, row in dividends[pay_mask].iterrows():
                symbol = row["symbol"]
                if symbol not in snapshot.positions:
                    continue
                cash_div = float(row.get("cash_div", 0) or 0)
                if cash_div <= 0:
                    continue
                pos = snapshot.positions[symbol]
                dividend_cash = pos.shares * cash_div
                snapshot.cash += dividend_cash
                events.append({
                    "date": date,
                    "symbol": symbol,
                    "type": "cash_div",
                    "shares": pos.shares,
                    "cash_div": cash_div,
                    "dividend_cash": dividend_cash,
                })

        return events
