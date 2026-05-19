from __future__ import annotations

from backtest.simulation.config import SimulationConfig
from backtest.simulation.models import BacktestResult, DecileBacktestResult, Trade, Position, DailySnapshot
from backtest.simulation.simple import SimpleSimulator
from backtest.simulation.decile import DecileSimulator, plot_decile_backtest
from backtest.simulation.detailed import DetailedSimulator
from backtest.simulation.executor import OrderExecutor
from backtest.simulation.dividends import DividendHandler
from backtest.simulation.utils import detect_board, round_lot, round_lot_for_symbol

__all__ = [
    "SimulationConfig",
    "BacktestResult",
    "DecileBacktestResult",
    "Trade",
    "Position",
    "DailySnapshot",
    "SimpleSimulator",
    "DecileSimulator",
    "plot_decile_backtest",
    "DetailedSimulator",
    "OrderExecutor",
    "DividendHandler",
    "detect_board",
    "round_lot",
    "round_lot_for_symbol",
]
