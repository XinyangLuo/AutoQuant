"""Strategy configuration: dataclasses, YAML/JSON loading, validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UniverseConfig:
    """Universe filtering parameters."""

    exclude_st: bool = True
    exclude_new_ipo_days: int = 252
    include_cyb: bool = True
    include_kcb: bool = False
    index_members: str | None = None
    min_market_cap: float | None = None
    min_avg_amount: float | None = None


@dataclass
class FactorConfig:
    """Single factor configuration within a strategy."""

    id: str
    direction: str = "desc"  # "desc" = higher is better, "asc" = lower is better
    weight: float = 1.0


@dataclass
class SelectionConfig:
    """Stock selection parameters."""

    method: str = "topk"  # "topk" / "long_short" / "decile"
    top_k: int = 20
    bottom_k: int = 20
    decile_group: int | None = None  # 0-9 for decile; None = return all groups


@dataclass
class WeightingConfig:
    """Portfolio weight allocation parameters."""

    method: str = "equal"  # "equal" / "market_cap" / "factor_value"


@dataclass
class NeutralizeConfig:
    """Neutralization parameters."""

    industry: bool = False
    industry_method: str = "group_rank"  # "group_rank" / "group_topk"
    market_cap: bool = False


@dataclass
class RiskConfig:
    """Risk control parameters (mostly reserved for future engine-level enforcement)."""

    max_industry_deviation: float | None = None
    max_single_stock_weight: float | None = None
    turnover_penalty: float = 0.0


@dataclass
class BacktestConfig:
    """Backtest date range and benchmark."""

    start_date: str = "20200101"
    end_date: str = "20241231"
    benchmark: str | None = "000300.SH"


@dataclass
class StrategyConfig:
    """Top-level strategy configuration."""

    name: str = "default"
    strategy_type: str = "single_factor_topk"
    rebalance_freq: str = "1W"  # "1D" / "1W" / "2W" / "1M" / "EOM"
    delay: int = 1

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    factors: list[FactorConfig] = field(default_factory=list)
    combine_method: str = "zscore_equal"  # "zscore_equal" / "ic_weighted" / "icir_weighted"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    neutralize: NeutralizeConfig = field(default_factory=NeutralizeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategyConfig:
        """Build StrategyConfig from a nested dict (YAML/JSON parsed)."""
        universe = UniverseConfig(**d.get("universe", {}))
        factors = [FactorConfig(**f) for f in d.get("factors", [])]
        selection = SelectionConfig(**d.get("selection", {}))
        weighting = WeightingConfig(**d.get("weighting", {}))
        neutralize = NeutralizeConfig(**d.get("neutralize", {}))
        risk = RiskConfig(**d.get("risk", {}))
        backtest = BacktestConfig(**d.get("backtest", {}))

        return cls(
            name=d.get("name", "default"),
            strategy_type=d.get("strategy", {}).get("type", "single_factor_topk"),
            rebalance_freq=d.get("strategy", {}).get("rebalance_freq", "1W"),
            delay=d.get("strategy", {}).get("delay", 1),
            universe=universe,
            factors=factors,
            combine_method=d.get("combine_method", "zscore_equal"),
            selection=selection,
            weighting=weighting,
            neutralize=neutralize,
            risk=risk,
            backtest=backtest,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategyConfig:
        """Load from YAML file. Falls back to JSON if PyYAML is unavailable."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            # Fallback: try JSON
            data = json.loads(text)

        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} did not parse to a dict")
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> StrategyConfig:
        """Load from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.factors:
            raise ValueError("At least one factor must be configured")

        for f in self.factors:
            if f.direction not in ("asc", "desc"):
                raise ValueError(f"Factor direction must be 'asc' or 'desc', got {f.direction}")

        if self.selection.method not in ("topk", "long_short", "decile"):
            raise ValueError(
                f"Selection method must be 'topk', 'long_short', or 'decile', "
                f"got {self.selection.method}"
            )

        if self.rebalance_freq not in ("1D", "1W", "2W", "1M", "EOM"):
            raise ValueError(
                f"Rebalance frequency must be '1D', '1W', '2W', '1M', or 'EOM', "
                f"got {self.rebalance_freq}"
            )

        if self.delay < 0:
            raise ValueError(f"Delay must be >= 0, got {self.delay}")

        if self.weighting.method not in ("equal", "market_cap", "factor_value"):
            raise ValueError(
                f"Weighting method must be 'equal', 'market_cap', or 'factor_value', "
                f"got {self.weighting.method}"
            )

        if self.combine_method not in ("zscore_equal", "ic_weighted", "icir_weighted", "risk_parity"):
            raise ValueError(
                f"Combine method must be 'zscore_equal', 'ic_weighted', 'icir_weighted', "
                f"or 'risk_parity', got {self.combine_method}"
            )
