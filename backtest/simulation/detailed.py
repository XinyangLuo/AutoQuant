from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Literal

import pandas as pd
import numpy as np

from backtest.simulation.config import SimulationConfig
from backtest.simulation.models import BacktestResult, DailySnapshot, Position, Trade
from backtest.simulation.executor import OrderExecutor
from backtest.simulation.dividends import DividendHandler
from backtest.simulation.utils import round_lot_for_symbol


@dataclass
class DailyMetrics:
    """单日汇总指标。"""

    date: Date
    turnover: float = 0.0               # 双边换手率
    buy_turnover: float = 0.0           # 买入单边换手率
    sell_turnover: float = 0.0          # 卖出单边换手率
    position_count: int = 0             # 持仓股票数
    long_count: int = 0                 # 多仓股票数
    short_count: int = 0                # 空仓股票数
    cash_ratio: float = 0.0             # 现金占比
    long_value: float = 0.0             # 多仓总市值
    short_value: float = 0.0            # 空仓总市值（绝对值）
    gross_exposure: float = 0.0         # 总敞口
    net_exposure: float = 0.0           # 净敞口
    top5_weight: float = 0.0            # 前5大持仓权重
    top10_weight: float = 0.0           # 前10大持仓权重
    herfindahl: float = 0.0             # 赫芬达尔指数
    commission: float = 0.0             # 当日总佣金
    stamp_duty: float = 0.0             # 当日总印花税
    transfer_fee: float = 0.0           # 当日总过户费
    trade_count: int = 0                # 当日成交笔数
    avg_trade_size: float = 0.0         # 当日成交均额


class Portfolio:
    """内部持仓管理，持有现金和股票头寸。"""

    def __init__(self, cash: float):
        self.cash = cash
        self.positions: dict[str, Position] = {}  # symbol -> Position

    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.position_value

    def update_market_value(self, bar_by_symbol: dict[str, pd.Series]) -> None:
        """用当日行情更新所有持仓市值。有数据的用 close，无数据的保持上日市值。"""
        for sym, pos in self.positions.items():
            if sym in bar_by_symbol:
                close_p = float(bar_by_symbol[sym]["close"])
                pos.market_value = pos.shares * close_p
            # 无数据：停牌，保持上日市值不变

    def execute_trade(
        self,
        symbol: str,
        direction: Literal["buy", "sell", "short", "cover"],
        shares: int,
        price: float,
        commission: float,
    ) -> None:
        """执行一笔成交，更新持仓和现金。"""
        amount = shares * price
        if direction in ("buy", "cover"):
            # 买入：减少现金，增加/减少持仓
            self.cash -= amount + commission
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=0,
                    avg_cost=0.0,
                    market_value=0.0,
                )
            pos = self.positions[symbol]
            old_shares = pos.shares
            old_cost = pos.avg_cost * abs(old_shares) if old_shares != 0 else 0.0
            # 买入方向增加 shares
            if direction == "buy":
                new_shares = old_shares + shares
                new_cost = old_cost + amount
                pos.shares = new_shares
                pos.avg_cost = new_cost / new_shares if new_shares != 0 else 0.0
            else:  # cover: 平空仓，减少负持仓
                pos.shares += shares  # shares 是正数，负持仓绝对值减小
                if pos.shares == 0:
                    pos.avg_cost = 0.0
                elif pos.shares > 0:
                    # 超平，转为多仓
                    pos.avg_cost = price
            pos.market_value = pos.shares * price

        else:  # sell, short
            # 卖出：增加现金，减少/增加持仓
            self.cash += amount - commission
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=0,
                    avg_cost=0.0,
                    market_value=0.0,
                )
            pos = self.positions[symbol]
            if direction == "sell":
                pos.shares -= shares
                if pos.shares == 0:
                    pos.avg_cost = 0.0
                    del self.positions[symbol]
                elif pos.shares < 0:
                    # 超卖，转为空仓
                    pos.avg_cost = price
                    pos.market_value = pos.shares * price
                else:
                    pos.market_value = pos.shares * price
            else:  # short: 开空仓
                old_shares = pos.shares
                old_short = abs(old_shares) if old_shares < 0 else 0
                old_cost = pos.avg_cost * old_short
                new_short = old_short + shares
                new_cost = old_cost + amount
                pos.shares = -new_short
                pos.avg_cost = new_cost / new_short if new_short != 0 else 0.0
                pos.market_value = pos.shares * price

    def remove_position(self, symbol: str) -> float:
        """移除某持仓（退市清零），返回被清零的市值。"""
        if symbol not in self.positions:
            return 0.0
        mv = self.positions[symbol].market_value
        del self.positions[symbol]
        return mv


class DetailedSimulator:
    """逐日事件驱动详细回测。"""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.executor = OrderExecutor(
            commission_rate=self.config.commission_rate,
            min_commission=self.config.min_commission,
            stamp_duty_rate=self.config.stamp_duty_rate,
            transfer_fee_rate=self.config.transfer_fee_rate,
            price_type=self.config.price_type,
        )
        self.dividend_handler = DividendHandler()

    def run(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
        dividends_data: pd.DataFrame | None = None,
        delisted_symbols: list[str] | None = None,
    ) -> BacktestResult:
        """逐日事件驱动回测。

        Parameters
        ----------
        signals : pd.DataFrame
            [date, symbol, target_weight]
        market_data : pd.DataFrame
            [date, symbol, open, high, low, close, volume, limit_up, limit_down, ...]
        dividends_data : pd.DataFrame | None
            [symbol, ex_date, pay_date, cash_div, stk_div]
        delisted_symbols : list[str] | None
            已退市股票列表，回测结束时这些持仓清零

        Returns
        -------
        BacktestResult
        """
        # 前置准备
        signal_by_date = {
            d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d): g
            for d, g in signals.groupby("date")
        }
        bar_by_date = {
            d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d): g
            for d, g in market_data.groupby("date")
        }

        # 在所有交易日运行（不只是调仓日），非调仓日保持持仓只更新市值
        dates = sorted(set(bar_by_date.keys()))
        if not dates:
            return BacktestResult(initial_cash=self.config.initial_cash)

        # 退市股票集合
        delisted = set(delisted_symbols or [])

        portfolio = Portfolio(cash=self.config.initial_cash)
        trades: list[Trade] = []
        snapshots: list[DailySnapshot] = []
        daily_metrics: list[DailyMetrics] = []
        last_signal_weights: dict[str, float] = {}  # 用于检测信号是否变化

        for date_str in dates:
            date = Date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            bars_df = bar_by_date[date_str]
            bar_by_symbol = bars_df.set_index("symbol").to_dict("index")
            available_symbols = set(bar_by_symbol.keys())
            daily_trades: list[Trade] = []

            # 1. 应用分红送转
            snapshot = DailySnapshot(
                date=date,
                cash=portfolio.cash,
                positions={
                    sym: Position(
                        symbol=sym,
                        shares=pos.shares,
                        avg_cost=pos.avg_cost,
                        market_value=pos.market_value,
                    )
                    for sym, pos in portfolio.positions.items()
                },
                total_value=portfolio.total_value,
                nav=portfolio.total_value / self.config.initial_cash,
            )
            self.dividend_handler.apply(date, snapshot, dividends_data)
            # 同步回 portfolio
            portfolio.cash = snapshot.cash
            for sym, pos in snapshot.positions.items():
                if sym in portfolio.positions:
                    portfolio.positions[sym].shares = pos.shares
                else:
                    portfolio.positions[sym] = pos

            # 2. 更新持仓市值（用当日 close）
            portfolio.update_market_value(bar_by_symbol)

            # 3. 检查退市：持仓中有退市股票的，清零
            for sym in list(portfolio.positions.keys()):
                if sym in delisted:
                    lost_value = portfolio.remove_position(sym)
                    trades.append(Trade(
                        trade_date=date,
                        symbol=sym,
                        direction="sell",
                        shares=0,
                        price=0.0,
                        amount=0.0,
                        commission=0.0,
                        reason="delisted",
                    ))

            # 4. 调仓信号（只在权重发生变化时执行）
            sig_df = signal_by_date.get(date_str)
            if sig_df is not None and not sig_df.empty:
                current_weights = {
                    row["symbol"]: float(row["target_weight"])
                    for _, row in sig_df.iterrows()
                }
                if current_weights != last_signal_weights:
                    last_signal_weights = current_weights
                    pre_trade_count = len(trades)
                    self._rebalance(
                        date=date,
                        portfolio=portfolio,
                        sig_df=sig_df,
                        bar_by_symbol=bar_by_symbol,
                        available_symbols=available_symbols,
                        trades=trades,
                    )
                    daily_trades = trades[pre_trade_count:]

            # 5. 记录每日快照
            total_value = portfolio.total_value
            snapshots.append(DailySnapshot(
                date=date,
                cash=portfolio.cash,
                positions={
                    sym: Position(
                        symbol=sym,
                        shares=pos.shares,
                        avg_cost=pos.avg_cost,
                        market_value=pos.market_value,
                    )
                    for sym, pos in portfolio.positions.items()
                },
                total_value=total_value,
                nav=total_value / self.config.initial_cash,
            ))

            # 6. 计算当日 metrics
            metrics = self._compute_daily_metrics(date, portfolio, daily_trades, total_value)
            daily_metrics.append(metrics)

        # 回测结束：未退市的停牌持仓保持到最后
        # 构建 nav_df
        nav_df = pd.DataFrame([
            {
                "date": s.date,
                "nav": s.nav,
                "daily_return": 0.0,  # 后面计算
                "total_value": s.total_value,
                "cash": s.cash,
                "position_value": s.total_value - s.cash,
            }
            for s in snapshots
        ])
        if len(nav_df) > 1:
            nav_df["daily_return"] = nav_df["nav"].pct_change()

        metrics_df = pd.DataFrame([
            {
                "date": m.date,
                "turnover": m.turnover,
                "buy_turnover": m.buy_turnover,
                "sell_turnover": m.sell_turnover,
                "position_count": m.position_count,
                "long_count": m.long_count,
                "short_count": m.short_count,
                "cash_ratio": m.cash_ratio,
                "long_value": m.long_value,
                "short_value": m.short_value,
                "gross_exposure": m.gross_exposure,
                "net_exposure": m.net_exposure,
                "top5_weight": m.top5_weight,
                "top10_weight": m.top10_weight,
                "herfindahl": m.herfindahl,
                "commission": m.commission,
                "stamp_duty": m.stamp_duty,
                "transfer_fee": m.transfer_fee,
                "trade_count": m.trade_count,
                "avg_trade_size": m.avg_trade_size,
            }
            for m in daily_metrics
        ]) if daily_metrics else None

        return BacktestResult(
            nav_df=nav_df,
            trades=trades,
            snapshots=snapshots,
            metrics_df=metrics_df,
            initial_cash=self.config.initial_cash,
        )

    def _compute_daily_metrics(
        self,
        date: Date,
        portfolio: Portfolio,
        daily_trades: list[Trade],
        total_value: float,
    ) -> DailyMetrics:
        """计算单日汇总指标。"""
        metrics = DailyMetrics(date=date)

        if total_value <= 0:
            return metrics

        # 交易统计 —— 单次遍历
        buy_amount = sell_amount = commission = stamp_duty = transfer_fee = total_amount = 0.0
        for t in daily_trades:
            if t.direction in ("buy", "cover"):
                buy_amount += t.amount
            else:
                sell_amount += t.amount
            commission += t.commission
            transfer_fee += t.amount * self.config.transfer_fee_rate
            if t.direction in ("sell", "short"):
                stamp_duty += t.amount * self.config.stamp_duty_rate
            total_amount += t.amount

        metrics.turnover = (buy_amount + sell_amount) / total_value
        metrics.buy_turnover = buy_amount / total_value
        metrics.sell_turnover = sell_amount / total_value
        metrics.commission = commission
        metrics.stamp_duty = stamp_duty
        metrics.transfer_fee = transfer_fee
        metrics.trade_count = len(daily_trades)
        if daily_trades:
            metrics.avg_trade_size = total_amount / len(daily_trades)

        # 持仓统计 + 集中度 —— 单次遍历
        long_value = short_value = 0.0
        weights: list[float] = []
        for p in portfolio.positions.values():
            if p.shares > 0:
                metrics.long_count += 1
                long_value += p.market_value
            elif p.shares < 0:
                metrics.short_count += 1
                short_value += abs(p.market_value)
            weights.append(abs(p.market_value) / total_value)

        metrics.position_count = metrics.long_count + metrics.short_count
        metrics.cash_ratio = portfolio.cash / total_value
        metrics.long_value = long_value
        metrics.short_value = short_value
        metrics.gross_exposure = (long_value + short_value) / total_value
        metrics.net_exposure = (long_value - short_value) / total_value

        weights_sorted = sorted(weights, reverse=True)
        metrics.top5_weight = sum(weights_sorted[:5])
        metrics.top10_weight = sum(weights_sorted[:10])
        metrics.herfindahl = sum(w * w for w in weights)

        return metrics

    def _rebalance(
        self,
        date: Date,
        portfolio: Portfolio,
        sig_df: pd.DataFrame,
        bar_by_symbol: dict[str, pd.Series],
        available_symbols: set[str],
        trades: list[Trade],
    ) -> None:
        """执行单日调仓。"""
        total_value = portfolio.total_value

        # 计算目标持仓股数（只对有行情数据的股票）
        target_shares: dict[str, int] = {}
        for _, row in sig_df.iterrows():
            symbol = row["symbol"]
            weight = float(row["target_weight"])
            if symbol not in available_symbols:
                continue
            bar = bar_by_symbol[symbol]

            # 确定成交价格（用于计算目标股数）
            if self.config.price_type == "o2o":
                trade_price = float(bar["open"])
            else:
                trade_price = float(bar["close"])

            if trade_price <= 0:
                continue

            target_value = total_value * weight
            raw_shares = target_value / trade_price
            shares = round_lot_for_symbol(abs(raw_shares), symbol)
            if shares == 0:
                continue
            target_shares[symbol] = shares if weight >= 0 else -shares

        # 生成交易订单
        orders: list[tuple[str, Literal["buy", "sell", "short", "cover"], int]] = []
        current_symbols = set(portfolio.positions.keys())
        all_relevant = set(target_shares.keys()) | current_symbols

        for symbol in all_relevant:
            pos = portfolio.positions.get(symbol)
            current = pos.shares if pos is not None else 0
            target = target_shares.get(symbol, 0)
            delta = target - current
            if delta == 0:
                continue
            if symbol not in available_symbols and current != 0:
                continue
            if delta > 0:
                direction: Literal["buy", "sell", "short", "cover"] = "buy" if current >= 0 else "cover"
                orders.append((symbol, direction, delta))
            else:
                direction = "sell" if current > 0 else "short"
                orders.append((symbol, direction, abs(delta)))

        # 先执行卖出（释放现金），再执行买入
        sell_orders = [o for o in orders if o[1] in ("sell", "short")]
        buy_orders = [o for o in orders if o[1] in ("buy", "cover")]

        # 执行卖出
        for symbol, direction, shares in sell_orders:
            if symbol not in available_symbols:
                continue
            bar = bar_by_symbol[symbol]
            can_trade, price, reason = self.executor.can_trade(symbol, direction, bar)
            if not can_trade:
                continue
            amount = shares * price
            commission = self.executor.calculate_cost(amount, direction)
            portfolio.execute_trade(symbol, direction, shares, price, commission)
            trades.append(Trade(
                trade_date=date,
                symbol=symbol,
                direction=direction,
                shares=shares,
                price=price,
                amount=amount,
                commission=commission,
                reason=reason,
            ))

        # 执行买入（检查现金不足）
        total_buy_cost = 0.0
        buy_candidates: list[tuple[str, Literal["buy", "cover"], int, float, float, str]] = []
        for symbol, direction, shares in buy_orders:
            if symbol not in available_symbols:
                continue
            bar = bar_by_symbol[symbol]
            can_trade, price, reason = self.executor.can_trade(symbol, direction, bar)
            if not can_trade:
                continue
            amount = shares * price
            commission = self.executor.calculate_cost(amount, direction)
            total_cost = amount + commission
            total_buy_cost += total_cost
            buy_candidates.append((symbol, direction, shares, price, total_cost, reason))

        if total_buy_cost > portfolio.cash and total_buy_cost > 0:
            # 现金不足，按比例缩减
            scale = portfolio.cash / total_buy_cost
            for i, (symbol, direction, shares, price, _, reason) in enumerate(buy_candidates):
                scaled_shares = int(shares * scale)
                if scaled_shares == 0:
                    continue
                # 重新按板块取整
                if scaled_shares != shares:
                    scaled_shares = round_lot_for_symbol(scaled_shares, symbol)
                if scaled_shares == 0:
                    continue
                amount = scaled_shares * price
                commission = self.executor.calculate_cost(amount, direction)
                total_cost = amount + commission
                if total_cost > portfolio.cash:
                    continue
                portfolio.execute_trade(symbol, direction, scaled_shares, price, commission)
                trades.append(Trade(
                    trade_date=date,
                    symbol=symbol,
                    direction=direction,
                    shares=scaled_shares,
                    price=price,
                    amount=amount,
                    commission=commission,
                    reason=reason,
                ))
        else:
            for symbol, direction, shares, price, total_cost, reason in buy_candidates:
                if total_cost > portfolio.cash:
                    continue
                amount = shares * price
                commission = self.executor.calculate_cost(amount, direction)
                portfolio.execute_trade(symbol, direction, shares, price, commission)
                trades.append(Trade(
                    trade_date=date,
                    symbol=symbol,
                    direction=direction,
                    shares=shares,
                    price=price,
                    amount=amount,
                    commission=commission,
                    reason=reason,
                ))
