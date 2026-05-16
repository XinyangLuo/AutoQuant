"""Strategy module: factor → target weights.

Public API
----------
StrategyConfig
    Configuration dataclass with YAML/JSON loading.

SingleFactorStrategy
    Single-factor strategy: topK / long-short / decile selection.

MultiFactorStrategy
    Multi-factor combination strategy: zscore_equal / IC-weighted / ICIR-weighted.

UniverseFilter
    Daily universe filtering (ST, IPO, board, index members, liquidity).

WeightAllocator
    Portfolio weight allocation (equal, market-cap, factor-value).

Neutralizer
    Industry and market-cap neutralization.

format_signals(signals) -> pd.DataFrame
    Validate and normalize strategy output for engine consumption.
"""

from backtest.strategy.base import StrategyBase
from backtest.strategy.config import (
    BacktestConfig,
    FactorConfig,
    NeutralizeConfig,
    RiskConfig,
    SelectionConfig,
    StrategyConfig,
    UniverseConfig,
    WeightingConfig,
)
from backtest.strategy.neutralize import Neutralizer
from backtest.strategy.selection import build_signals
from backtest.strategy.signals import format_signals, group_by_date, normalize_weights
from backtest.strategy.strategies.multi_factor import MultiFactorStrategy
from backtest.strategy.strategies.single_factor import SingleFactorStrategy
from backtest.strategy.universe import UniverseFilter
from backtest.strategy.weight import WeightAllocator

__all__ = [
    "StrategyBase",
    "StrategyConfig",
    "UniverseConfig",
    "FactorConfig",
    "SelectionConfig",
    "WeightingConfig",
    "NeutralizeConfig",
    "RiskConfig",
    "BacktestConfig",
    "SingleFactorStrategy",
    "MultiFactorStrategy",
    "UniverseFilter",
    "WeightAllocator",
    "Neutralizer",
    "build_signals",
    "format_signals",
    "group_by_date",
    "normalize_weights",
]
