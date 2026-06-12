"""Tests for simulation (backtest engine) module."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.simulation import (
    SimulationConfig,
    SimpleSimulator,
    DetailedSimulator,
    OrderExecutor,
    DividendHandler,
    detect_board,
    round_lot,
    round_lot_for_symbol,
)
from backtest.simulation.models import DailySnapshot, Position


class TestUtils:
    """Test board detection and lot rounding."""

    def test_detect_board_kcb(self):
        assert detect_board("688001.SH") == "kcb"
        assert detect_board("688999.SH") == "kcb"

    def test_detect_board_bj(self):
        assert detect_board("830009.BJ") == "bj"
        assert detect_board("430047.BJ") == "bj"

    def test_detect_board_default(self):
        assert detect_board("000001.SZ") == "default"
        assert detect_board("600000.SH") == "default"
        assert detect_board("300001.SZ") == "default"

    def test_round_lot_default(self):
        assert round_lot(50, "default") == 0    # < 100, skip
        assert round_lot(100, "default") == 100
        assert round_lot(153, "default") == 100 # floor to 100
        assert round_lot(250, "default") == 200

    def test_round_lot_kcb(self):
        assert round_lot(150, "kcb") == 0       # < 200, skip
        assert round_lot(200, "kcb") == 200
        assert round_lot(253, "kcb") == 253     # 1-share increment

    def test_round_lot_bj(self):
        assert round_lot(50, "bj") == 0         # < 100, skip
        assert round_lot(100, "bj") == 100
        assert round_lot(153, "bj") == 153      # 1-share increment

    def test_round_lot_for_symbol(self):
        assert round_lot_for_symbol(150, "688001.SH") == 0
        assert round_lot_for_symbol(253, "688001.SH") == 253
        assert round_lot_for_symbol(153, "000001.SZ") == 100


class TestOrderExecutor:
    """Test limit up/down and trading rules."""

    def _make_bar(self, open_p, high_p, low_p, close_p, limit_up, limit_down):
        return pd.Series({
            "open": open_p, "high": high_p, "low": low_p, "close": close_p,
            "limit_up": limit_up, "limit_down": limit_down,
        })

    # ---- o2o mode ----

    def test_o2o_buy_normal(self):
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(10.0, 10.5, 9.8, 10.2, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "buy", bar)
        assert ok is True
        assert price == 10.0
        assert reason == "normal"

    def test_o2o_buy_limit_up_blocked(self):
        """涨停开盘且全天未打开 → 不能买入"""
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(11.0, 11.0, 11.0, 11.0, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "buy", bar)
        assert ok is False
        assert reason == "limit_up_blocked"

    def test_o2o_buy_limit_up_traded(self):
        """涨停开盘但盘中打开 → 可用 limit_up 成交"""
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(11.0, 11.0, 10.5, 10.8, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "buy", bar)
        assert ok is True
        assert price == 11.0
        assert reason == "limit_up_traded"

    def test_o2o_sell_normal(self):
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(10.0, 10.5, 9.8, 10.2, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "sell", bar)
        assert ok is True
        assert price == 10.0

    def test_o2o_sell_limit_down_blocked(self):
        """跌停开盘且全天未打开 → 不能卖出"""
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(9.0, 9.0, 9.0, 9.0, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "sell", bar)
        assert ok is False
        assert reason == "limit_down_blocked"

    def test_o2o_sell_limit_down_traded(self):
        """跌停开盘但盘中打开 → 可用 limit_down 成交"""
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(9.0, 9.5, 9.0, 9.3, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "sell", bar)
        assert ok is True
        assert price == 9.0
        assert reason == "limit_down_traded"

    def test_o2o_short_blocked_by_limit_down(self):
        """做空开仓（short）和 sell 一样受跌停限制"""
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(9.0, 9.0, 9.0, 9.0, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "short", bar)
        assert ok is False
        assert reason == "limit_down_blocked"

    def test_o2o_cover_blocked_by_limit_up(self):
        """做空平仓（cover）和 buy 一样受涨停限制"""
        ex = OrderExecutor(price_type="o2o")
        bar = self._make_bar(11.0, 11.0, 11.0, 11.0, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "cover", bar)
        assert ok is False
        assert reason == "limit_up_blocked"

    # ---- c2c mode ----

    def test_c2c_buy_limit_up_blocked(self):
        ex = OrderExecutor(price_type="c2c")
        bar = self._make_bar(10.5, 10.8, 10.2, 11.0, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "buy", bar)
        assert ok is False
        assert reason == "limit_up_blocked"

    def test_c2c_sell_limit_down_blocked(self):
        ex = OrderExecutor(price_type="c2c")
        bar = self._make_bar(9.5, 9.8, 9.0, 9.0, 11.0, 9.0)
        ok, price, reason = ex.can_trade("000001.SZ", "sell", bar)
        assert ok is False
        assert reason == "limit_down_blocked"

    # ---- fees ----

    def test_calculate_cost_buy(self):
        ex = OrderExecutor(commission_rate=0.0003, min_commission=5.0)
        # 买入：佣金(min 5元) + 过户费（无印花税）
        cost = ex.calculate_cost(10000, "buy")
        # 10000 * 0.0003 = 3 < 5, 按 5 元收
        assert cost == pytest.approx(5.0 + 10000 * 0.00002)

    def test_calculate_cost_sell(self):
        ex = OrderExecutor(commission_rate=0.0003, min_commission=5.0)
        # 卖出：佣金(min 5元) + 印花税 + 过户费
        cost = ex.calculate_cost(10000, "sell")
        assert cost == pytest.approx(5.0 + 10000 * 0.001 + 10000 * 0.00002)

    def test_calculate_cost_min_commission(self):
        ex = OrderExecutor(commission_rate=0.0003, min_commission=5.0)
        # 小额交易，佣金低于5元按5元收
        cost = ex.calculate_cost(1000, "buy")
        assert cost == pytest.approx(5.0 + 1000 * 0.00002)


class TestDividendHandler:
    """Test dividend application."""

    def test_stk_div(self):
        handler = DividendHandler()
        snapshot = DailySnapshot(
            date=date(2024, 6, 1),
            cash=100000,
            positions={"000001.SZ": Position("000001.SZ", 1000, 10.0, 10000)},
            total_value=110000,
            nav=1.1,
        )
        divs = pd.DataFrame({
            "symbol": ["000001.SZ"],
            "ex_date": ["20240601"],
            "pay_date": ["20240605"],
            "cash_div": [0.5],
            "stk_div": [0.3],
        })
        events = handler.apply(date(2024, 6, 1), snapshot, divs)
        assert len(events) == 1
        assert events[0]["type"] == "stk_div"
        assert snapshot.positions["000001.SZ"].shares == 1300

    def test_cash_div(self):
        handler = DividendHandler()
        snapshot = DailySnapshot(
            date=date(2024, 6, 5),
            cash=100000,
            positions={"000001.SZ": Position("000001.SZ", 1000, 10.0, 10000)},
            total_value=110000,
            nav=1.1,
        )
        divs = pd.DataFrame({
            "symbol": ["000001.SZ"],
            "ex_date": ["20240601"],
            "pay_date": ["20240605"],
            "cash_div": [0.5],
            "stk_div": [0.3],
        })
        events = handler.apply(date(2024, 6, 5), snapshot, divs)
        assert len(events) == 1
        assert events[0]["type"] == "cash_div"
        assert snapshot.cash == 100500  # 100000 + 1000 * 0.5

    def test_detailed_simulator_syncs_stk_div_avg_cost(self):
        sim = DetailedSimulator(SimulationConfig(
            initial_cash=10000,
            commission_rate=0.0,
            min_commission=0.0,
            stamp_duty_rate=0.0,
            transfer_fee_rate=0.0,
        ))
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-03"]),
            "symbol": ["A"],
            "target_weight": [1.0],
        })
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
            "symbol": ["A", "A"],
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [10000, 10000],
            "limit_up": [11.0, 11.0],
            "limit_down": [9.0, 9.0],
        })
        dividends = pd.DataFrame({
            "symbol": ["A"],
            "ex_date": ["20240604"],
            "pay_date": ["20240610"],
            "cash_div": [0.0],
            "stk_div": [1.0],
        })

        result = sim.run(signals, market, dividends)
        positions = result.positions_df
        day2 = positions[positions["date"] == date(2024, 6, 4)].iloc[0]

        assert day2["shares"] == 2000
        assert day2["avg_cost"] == pytest.approx(5.0)

    def test_no_position_no_event(self):
        handler = DividendHandler()
        snapshot = DailySnapshot(
            date=date(2024, 6, 1),
            cash=100000,
            positions={},
            total_value=100000,
            nav=1.0,
        )
        divs = pd.DataFrame({
            "symbol": ["000001.SZ"],
            "ex_date": ["20240601"],
            "pay_date": ["20240605"],
            "cash_div": [0.5],
            "stk_div": [0.3],
        })
        events = handler.apply(date(2024, 6, 1), snapshot, divs)
        assert len(events) == 0


class TestSimpleSimulator:
    """Test vectorized simple backtest."""

    def _make_market_data(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A"] * 3 + ["B"] * 3,
            "close": [100.0, 110.0, 99.0, 100.0, 100.0, 100.0],
            "adj_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })

    def test_single_stock_constant_weight(self):
        """单股票恒权重，验证净值计算。"""
        sim = SimpleSimulator()
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["A", "A", "A"],
            "target_weight": [1.0, 1.0, 1.0],
        })
        market = self._make_market_data()
        result = sim.run(signals, market)

        nav = result.nav_df["nav"].values
        assert nav[0] == pytest.approx(1.0)
        assert nav[1] == pytest.approx(1.1)   # 100 -> 110, +10%
        assert nav[2] == pytest.approx(0.99)  # 110 -> 99, -10%

    def test_equal_weight_two_stocks(self):
        """两只股票等权，组合收益为平均。"""
        sim = SimpleSimulator()
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "target_weight": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        })
        market = self._make_market_data()
        result = sim.run(signals, market)

        nav = result.nav_df["nav"].values
        # Day1: A=+10%, B=0% → avg = +5%
        assert nav[1] == pytest.approx(1.05, abs=1e-6)
        # Day2: A=-10%, B=0% → avg = -5%
        assert nav[2] == pytest.approx(0.9975, abs=1e-6)  # 1.05 * 0.95

    def test_missing_data(self):
        """某股票某天无数据，该位置收益为0，不影响其他。"""
        sim = SimpleSimulator()
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A", "A", "A", "C", "C", "C"],
            "target_weight": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        })
        # market 中没有 C 的数据
        market = self._make_market_data()
        result = sim.run(signals, market)

        nav = result.nav_df["nav"].values
        # C 无数据，收益为0，只有 A 贡献
        assert nav[1] == pytest.approx(1.05, abs=1e-6)

    def test_does_not_require_dense_pivot_matrix(self, monkeypatch):
        """SimpleSimulator computes sparse long-form returns without pivoting."""
        sim = SimpleSimulator()
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "target_weight": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        })
        market = self._make_market_data()

        def _fail_pivot(self, *args, **kwargs):
            raise AssertionError("SimpleSimulator should not build a dense pivot matrix")

        monkeypatch.setattr(pd.DataFrame, "pivot", _fail_pivot)

        result = sim.run(signals, market)

        nav = result.nav_df["nav"].values
        assert nav[0] == pytest.approx(1.0, abs=1e-6)
        assert nav[1] == pytest.approx(1.05, abs=1e-6)
        assert nav[2] == pytest.approx(0.9975, abs=1e-6)

    def test_run_batch_matches_individual_sparse_runs(self):
        """Batch simple backtest is numerically identical to per-combo runs."""
        sim = SimpleSimulator()
        market = self._make_market_data()
        combo_a = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "target_weight": [0.7, 0.7, 0.7, 0.3, 0.3, 0.3],
            "combo_tag": ["combo_a"] * 6,
        })
        combo_b = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "target_weight": [0.2, 0.4, 0.6, 0.8, 0.6, 0.4],
            "combo_tag": ["combo_b"] * 6,
        })
        signals = pd.concat([combo_a, combo_b], ignore_index=True)

        actual = sim.run_batch(signals, market, strategy_col="combo_tag")

        assert set(actual) == {"combo_a", "combo_b"}
        for tag in sorted(actual):
            single_signals = (
                signals[signals["combo_tag"] == tag]
                .drop(columns=["combo_tag"])
                .reset_index(drop=True)
            )
            expected = sim.run(single_signals, market)
            pd.testing.assert_frame_equal(
                actual[tag].nav_df.reset_index(drop=True),
                expected.nav_df.reset_index(drop=True),
            )

    def test_run_batch_does_not_require_dense_pivot_matrix(self, monkeypatch):
        """Batch path keeps the sparse long-form contract and avoids pivot."""
        sim = SimpleSimulator()
        market = self._make_market_data()
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "target_weight": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "combo_tag": ["combo_a"] * 3 + ["combo_b"] * 3,
        })

        def _fail_pivot(self, *args, **kwargs):
            raise AssertionError("SimpleSimulator.run_batch should not build a dense pivot matrix")

        monkeypatch.setattr(pd.DataFrame, "pivot", _fail_pivot)

        result = sim.run_batch(signals, market, strategy_col="combo_tag")

        assert sorted(result) == ["combo_a", "combo_b"]


class TestDetailedSimulator:
    """Test event-driven detailed backtest."""

    def _make_market_data(self, price=10.0, limit_pct=0.1):
        limit_up = round(price * (1 + limit_pct), 2)
        limit_down = round(price * (1 - limit_pct), 2)
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["A"] * 3,
            "open": [price, price, price],
            "high": [price, price, price],
            "low": [price, price, price],
            "close": [price, price, price],
            "volume": [10000, 10000, 10000],
            "limit_up": [limit_up, limit_up, limit_up],
            "limit_down": [limit_down, limit_down, limit_down],
        })

    def _zero_fee_config(self, **kwargs):
        """返回零费用配置，用于测试核心逻辑。"""
        return SimulationConfig(
            initial_cash=kwargs.get("initial_cash", 100000),
            commission_rate=0.0,
            min_commission=0.0,
            stamp_duty_rate=0.0,
            transfer_fee_rate=0.0,
            price_type=kwargs.get("price_type", "o2o"),
        )

    def test_basic_buy_and_hold(self):
        """买入持有，验证持仓和现金。"""
        sim = DetailedSimulator(self._zero_fee_config())

        # Day1: 买入 A，weight=1.0
        # Day2-3: 持有不变
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["A", "A", "A"],
            "target_weight": [1.0, 1.0, 1.0],
        })
        market = self._make_market_data(price=10.0)
        result = sim.run(signals, market)

        nav = result.nav_df["nav"].values
        assert nav[0] == pytest.approx(1.0, abs=1e-6)
        # 无涨跌，净值保持 1.0
        assert nav[1] == pytest.approx(1.0, abs=1e-6)

        # 验证持仓：Day1 买入 10000 股（10万/10元 = 10000股，主板100股整数倍）
        positions = result.positions_df
        day1_pos = positions[positions["date"] == date(2024, 1, 1)]
        assert len(day1_pos) == 1
        assert day1_pos.iloc[0]["shares"] == 10000

    def test_commission_deducted(self):
        """验证手续费从现金中扣除。"""
        config = SimulationConfig(initial_cash=100000, commission_rate=0.001)
        sim = DetailedSimulator(config)

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "target_weight": [1.0],
        })
        market = self._make_market_data(price=10.0)
        result = sim.run(signals, market)

        # 买入 10000 股 @ 10元 = 100000
        # 佣金 = max(100000 * 0.001, 5) = 100
        # 过户费 = 100000 * 0.00002 = 2
        # 总费用 = 102
        # 剩余现金 = 0 - 102 = -102... 等等，现金不够了
        # 实际：目标金额 = 100000，但总成本 = 100000 + 102 = 100102 > 现金
        # 按比例缩减

        trades = result.trades_df
        assert len(trades) == 1
        trade = trades.iloc[0]
        # 买入金额应略小于 100000
        assert trade["amount"] < 100000
        assert trade["commission"] == pytest.approx(
            max(trade["amount"] * 0.001, 5.0) + trade["amount"] * 0.00002
        )

    def test_sell_order_before_buy(self):
        """先卖出释放现金，再买入。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "target_weight": [1.0, 0.0],
        })
        market = self._make_market_data(price=10.0)
        result = sim.run(signals, market)

        trades = result.trades_df
        # Day1: buy 10000 shares; Day2: sell 10000 shares
        assert len(trades) == 2
        assert trades.iloc[0]["direction"] == "buy"
        assert trades.iloc[1]["direction"] == "sell"

    def test_kcb_rounding(self):
        """科创板：200股起，1股递增。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["688001.SH"],
            "target_weight": [1.0],
        })
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["688001.SH"],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [10000],
            "limit_up": [110.0],
            "limit_down": [90.0],
        })
        result = sim.run(signals, market)

        trades = result.trades_df
        assert len(trades) == 1
        # 100000 / 100 = 1000 股，科创板 200起1股递增 → 1000 股
        assert trades.iloc[0]["shares"] == 1000

    def test_kcb_below_minimum(self):
        """科创板：目标股数 < 200 → 跳过不买。"""
        sim = DetailedSimulator(self._zero_fee_config(initial_cash=10000))

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["688001.SH"],
            "target_weight": [1.0],
        })
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["688001.SH"],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [10000],
            "limit_up": [110.0],
            "limit_down": [90.0],
        })
        result = sim.run(signals, market)

        # 10000/100 = 100 股 < 200 → 不买（trades_df 为 None）
        trades = result.trades_df
        assert trades is None or len(trades) == 0

    def test_limit_up_blocked_buy(self):
        """涨停日买入被阻塞。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "target_weight": [1.0],
        })
        # 涨停：open=limit_up, low=limit_up
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "open": [11.0],
            "high": [11.0],
            "low": [11.0],
            "close": [11.0],
            "volume": [10000],
            "limit_up": [11.0],
            "limit_down": [9.0],
        })
        result = sim.run(signals, market)

        trades = result.trades_df
        assert trades is None or len(trades) == 0  # 被涨停阻塞，无成交

    def test_limit_down_blocked_sell(self):
        """跌停日卖出被阻塞。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "target_weight": [1.0, 0.0],
        })
        # Day2 跌停：open = limit_down = 9.0
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "open": [10.0, 9.0],
            "high": [10.0, 9.0],
            "low": [10.0, 9.0],
            "close": [10.0, 9.0],
            "volume": [10000, 10000],
            "limit_up": [11.0, 9.9],
            "limit_down": [9.0, 9.0],  # Day2 limit_down = 9.0 = open
        })
        result = sim.run(signals, market)

        trades = result.trades_df
        # Day1 buy succeeds, Day2 sell blocked by limit_down
        assert len(trades) == 1
        assert trades.iloc[0]["direction"] == "buy"

    def test_short_and_cover(self):
        """做空开仓和平仓。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "target_weight": [-0.5, 0.0],
        })
        market = self._make_market_data(price=10.0)
        result = sim.run(signals, market)

        trades = result.trades_df
        assert len(trades) == 2
        assert trades.iloc[0]["direction"] == "short"
        assert trades.iloc[1]["direction"] == "cover"

    def test_c2c_mode(self):
        """c2c 模式：收盘价成交。"""
        sim = DetailedSimulator(self._zero_fee_config(price_type="c2c"))

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "target_weight": [1.0],
        })
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "open": [9.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.0],
            "volume": [10000],
            "limit_up": [11.0],
            "limit_down": [9.0],
        })
        result = sim.run(signals, market)

        trades = result.trades_df
        assert len(trades) == 1
        assert trades.iloc[0]["price"] == 10.0  # 收盘价成交

    def test_delisted_zeroed(self):
        """退市股票持仓清零。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "target_weight": [1.0, 1.0],
        })
        # market 需要覆盖两天，Day2 无数据 + 退市
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [10000, 10000],
            "limit_up": [11.0, 11.0],
            "limit_down": [9.0, 9.0],
        })
        result = sim.run(signals, market, delisted_symbols=["A"])

        # Day2 A 退市，持仓清零
        nav = result.nav_df["nav"].values
        assert nav[1] == pytest.approx(0.0, abs=1e-6)

    def test_backtest_result_summary(self):
        """验证回测结果 summary 计算。"""
        sim = DetailedSimulator(self._zero_fee_config())

        # 构造一个明确的涨跌序列
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["A", "A", "A"],
            "target_weight": [1.0, 1.0, 1.0],
        })
        market = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["A"] * 3,
            "open": [10.0, 11.0, 9.9],
            "high": [10.0, 11.0, 9.9],
            "low": [10.0, 11.0, 9.9],
            "close": [10.0, 11.0, 9.9],
            "volume": [10000] * 3,
            "limit_up": [11.0, 12.1, 10.89],
            "limit_down": [9.0, 9.9, 8.91],
        })
        result = sim.run(signals, market)

        summary = result.summary()
        # Day1 buy @ 10, Day2 close @ 11 (+10%), Day3 close @ 9.9 (-10% from 11)
        # NAV: 1.0 -> 1.1 -> 0.99
        assert summary["total_return"] == pytest.approx(-0.01, abs=1e-6)

    def test_metrics_computed(self):
        """验证每日 metrics 正确计算。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "target_weight": [1.0, 0.0],
        })
        market = self._make_market_data(price=10.0)
        result = sim.run(signals, market)

        assert result.metrics_df is not None
        assert len(result.metrics_df) == 3  # 3 trading days, non-rebalance days included

        # Day1: 买入 10000 股 @ 10 = 100000，turnover = 1.0
        m1 = result.metrics_df.iloc[0]
        assert m1["turnover"] == pytest.approx(1.0, abs=1e-6)
        assert m1["buy_turnover"] == pytest.approx(1.0, abs=1e-6)
        assert m1["sell_turnover"] == pytest.approx(0.0, abs=1e-6)
        assert m1["position_count"] == 1
        assert m1["long_count"] == 1
        assert m1["short_count"] == 0
        assert m1["cash_ratio"] == pytest.approx(0.0, abs=1e-6)
        assert m1["herfindahl"] == pytest.approx(1.0, abs=1e-6)

        # Day2: 卖出 10000 股 @ 10 = 100000，turnover = 1.0
        m2 = result.metrics_df.iloc[1]
        assert m2["turnover"] == pytest.approx(1.0, abs=1e-6)
        assert m2["sell_turnover"] == pytest.approx(1.0, abs=1e-6)
        assert m2["position_count"] == 0

        # Day3: 非调仓日，无交易，保持空仓
        m3 = result.metrics_df.iloc[2]
        assert m3["turnover"] == pytest.approx(0.0, abs=1e-6)
        assert m3["position_count"] == 0

    def test_save_outputs(self, tmp_path):
        """验证 save() 正确保存所有文件。"""
        sim = DetailedSimulator(self._zero_fee_config())

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "target_weight": [1.0],
        })
        market = self._make_market_data(price=10.0)
        result = sim.run(signals, market)

        output_dir = tmp_path / "backtest_result"
        metadata = {
            "strategy": {"name": "test"},
            "summary": result.summary(),
        }
        result.save(str(output_dir), metadata=metadata)

        assert (output_dir / "nav.parquet").exists()
        assert (output_dir / "positions.parquet").exists()
        assert (output_dir / "trades.parquet").exists()
        assert (output_dir / "metrics.parquet").exists()
        assert (output_dir / "metadata.json").exists()

        # 验证 metadata.json 内容
        import json
        with open(output_dir / "metadata.json") as f:
            saved_meta = json.load(f)
        assert saved_meta["strategy"]["name"] == "test"
        assert "summary" in saved_meta

    def test_summary_extended(self):
        """验证 summary 包含扩展字段。"""
        config = SimulationConfig(
            initial_cash=100000,
            commission_rate=0.001,
            min_commission=5.0,
            stamp_duty_rate=0.001,
            transfer_fee_rate=0.00002,
        )
        sim = DetailedSimulator(config)

        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "target_weight": [1.0, 0.0],
        })
        market = self._make_market_data(price=10.0)
        result = sim.run(signals, market)

        summary = result.summary()
        assert "total_commission" in summary
        assert "total_stamp_duty" in summary
        assert "total_trades" in summary
        assert "avg_daily_turnover" in summary
        assert "max_drawdown_start" in summary
        assert "max_drawdown_end" in summary
