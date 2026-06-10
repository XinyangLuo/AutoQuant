"""Strategy configuration: dataclasses, YAML/JSON loading, validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backtest.config_loader import get_section


_UNIVERSE_FALLBACKS: dict[str, Any] = {
    "exclude_st": True,
    "exclude_new_ipo_days": 252,
    "include_cyb": True,
    "include_kcb": False,
    "include_bse": False,
    "index_members": None,
    "min_market_cap": None,
    "min_avg_amount": None,
}


def _universe_default(key: str):
    """Read a universe default from ``config.yaml strategy.universe``."""
    def _factory():
        try:
            return get_section("strategy", "universe", key)
        except (KeyError, FileNotFoundError):
            return _UNIVERSE_FALLBACKS[key]
    return _factory


@dataclass
class UniverseConfig:
    """Universe filtering parameters."""

    exclude_st: bool = field(default_factory=_universe_default("exclude_st"))
    exclude_new_ipo_days: int = field(default_factory=_universe_default("exclude_new_ipo_days"))
    include_cyb: bool = field(default_factory=_universe_default("include_cyb"))
    include_kcb: bool = field(default_factory=_universe_default("include_kcb"))
    include_bse: bool = field(default_factory=_universe_default("include_bse"))
    index_members: str | None = field(default_factory=_universe_default("index_members"))
    min_market_cap: float | None = field(default_factory=_universe_default("min_market_cap"))
    min_avg_amount: float | None = field(default_factory=_universe_default("min_avg_amount"))


@dataclass
class FactorConfig:
    """Single factor configuration within a strategy.

    The strategy reads the factor's values at whichever neutralization
    variant the factor was registered with — variant is a property of the
    factor (recorded in registry), not the strategy.
    """

    id: str
    direction: str = "desc"  # "desc" = higher is better, "asc" = lower is better
    weight: float = 1.0


@dataclass
class SelectionConfig:
    """Stock selection parameters.

    ``top_k`` / ``bottom_k`` 与 ``top_pct`` / ``bottom_pct`` 互斥 —— 一边指
    绝对数量(``top_k=50`` 选前 50 只),一边指相对百分位(``top_pct=0.1``
    选前 10%)。每端必须恰好指定其中一个,validate 会强制。
    """

    method: str = "topk"  # "topk" / "long_short" / "decile"
    top_k: int | None = None
    bottom_k: int | None = None
    top_pct: float | None = None       # (0, 1],e.g. 0.1 表示前 10%
    bottom_pct: float | None = None    # (0, 1],e.g. 0.1 表示后 10%
    decile_group: int | None = None  # 0-9 for decile; None = return all groups


@dataclass
class WeightingConfig:
    """Portfolio weight allocation parameters."""

    method: str = "equal"  # "equal" / "market_cap" / "factor_value"


@dataclass
class RiskConfig:
    """Risk control parameters (mostly reserved for future engine-level enforcement)."""

    max_industry_deviation: float | None = None
    max_single_stock_weight: float | None = None
    turnover_penalty: float = 0.0


@dataclass
class BacktestConfig:
    """Backtest date range and benchmark."""

    start_date: str = "20160101"
    end_date: str = "20251231"
    benchmark: str | None = "000300.SH"


@dataclass
class StrategyConfig:
    """Top-level strategy configuration."""

    name: str = "default"
    strategy_type: str = "single_factor_topk"
    rebalance_freq: str = "1D"  # "1D" / "5D" / "1W" / "2W" / "1M" / "EOM"
    delay: int = 1

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    factors: list[FactorConfig] = field(default_factory=list)
    combine_method: str = "zscore_equal"  # "zscore_equal" / "ic_weighted" / "icir_weighted"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Decay: linear decay smoothing applied to factor values before signal
    # generation.  decay(x, n) weights recent values more heavily.
    # None = no decay (use raw factor values).
    decay: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict compatible with ``from_dict``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategyConfig:
        """Build StrategyConfig from YAML/JSON or ``to_dict`` output.

        Historical config files store strategy metadata under a nested
        ``strategy`` block, while persisted pipeline state stores dataclass
        fields directly (``strategy_type``, ``rebalance_freq``, ``delay``).
        Support both shapes so pipeline resume can restore full strategy state.
        """
        strategy_block = d.get("strategy", {})
        universe = UniverseConfig(**d.get("universe", {}))
        factors = [FactorConfig(**f) for f in d.get("factors", [])]
        selection = SelectionConfig(**d.get("selection", {}))
        weighting = WeightingConfig(**d.get("weighting", {}))
        risk = RiskConfig(**d.get("risk", {}))
        backtest = BacktestConfig(**d.get("backtest", {}))

        return cls(
            name=d.get("name", "default"),
            strategy_type=d.get(
                "strategy_type",
                strategy_block.get("type", "single_factor_topk"),
            ),
            rebalance_freq=d.get(
                "rebalance_freq",
                strategy_block.get("rebalance_freq", "1D"),
            ),
            delay=d.get("delay", strategy_block.get("delay", 1)),
            universe=universe,
            factors=factors,
            combine_method=d.get("combine_method", "zscore_equal"),
            selection=selection,
            weighting=weighting,
            risk=risk,
            backtest=backtest,
            decay=d.get("decay"),
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

        self._validate_selection_counts()

        if self.rebalance_freq not in ("1D", "5D", "1W", "2W", "1M", "EOM"):
            raise ValueError(
                f"Rebalance frequency must be '1D', '5D', '1W', '2W', '1M', or 'EOM', "
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

        if self.decay is not None and self.decay < 1:
            raise ValueError(f"Decay must be >= 1 or None, got {self.decay}")

    def _validate_selection_counts(self) -> None:
        """互斥校验:每端(long/short)恰好指定 k 或 pct 一个。"""
        s = self.selection
        if s.method == "topk":
            self._check_xor("top_k", s.top_k, "top_pct", s.top_pct)
        elif s.method == "long_short":
            self._check_xor("top_k", s.top_k, "top_pct", s.top_pct)
            self._check_xor("bottom_k", s.bottom_k, "bottom_pct", s.bottom_pct)
        # decile 不需要 top_k/top_pct,跳过

    @staticmethod
    def _check_xor(k_name: str, k_val, pct_name: str, pct_val) -> None:
        if (k_val is None) == (pct_val is None):
            raise ValueError(
                f"必须恰好指定 {k_name} 或 {pct_name} 之一,"
                f"得到 {k_name}={k_val}, {pct_name}={pct_val}"
            )
        if pct_val is not None and not (0 < pct_val <= 1):
            raise ValueError(f"{pct_name} 必须在 (0, 1] 区间,得到 {pct_val}")
        if k_val is not None and k_val < 1:
            raise ValueError(f"{k_name} 必须 >= 1,得到 {k_val}")
