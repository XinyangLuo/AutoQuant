"""Pipeline configuration: thresholds and knobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class StepThresholds:
    """Per-step admission thresholds."""

    # step1: coverage
    max_missing_rate_pv: float = 0.10
    max_missing_rate_fin: float = 0.30

    # step3: ICIR
    min_abs_ic: float = 0.01
    min_annual_icir: float = 1.0
    min_ic_tstat: float = 2.0
    min_ic_positive_ratio: float = 0.55

    # step4: monotonicity
    min_monotonicity: float = 0.7

    # step6: simple backtest
    min_sharpe_simple: float = 0.8
    min_annual_return_simple: float = 0.10
    max_max_drawdown: float = 0.30
    min_calmar_simple: float = 0.5
    max_annual_turnover: float = 20.0

    # step7: detailed backtest
    min_sharpe_detailed: float = 0.4
    min_annual_return_detailed: float = 0.08
    min_calmar_detailed: float = 0.5


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    factor_id: str
    start_date: str
    end_date: str
    frequency: Literal["D", "M"] = "D"

    # step3: ICIR check horizons
    eval_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])
    icir_check_horizons: list[int] = field(default_factory=lambda: [1, 5])

    # step2: neutralization verification
    max_corr_size: float = 0.05
    max_corr_industry: float = 0.05
    max_corr_existing: float = 0.5

    # step5: default strategy config
    default_top_pct: float = 0.1
    default_decay: int = 5
    default_rebalance: str = "1D"
    default_universe: str | None = None

    # retry
    max_retries: int = 3

    # thresholds (frequency-aware)
    thresholds: StepThresholds = field(default_factory=StepThresholds)

    # output
    results_root: str = "results"
    ret_type: str = "open"
    benchmark: str = "000300.SH"

    @classmethod
    def for_frequency(
        cls,
        frequency: Literal["D", "M"],
        **kwargs,
    ) -> PipelineConfig:
        """Factory with frequency-specific defaults."""
        th = StepThresholds()
        if frequency == "M":
            th.min_abs_ic = 0.03
            th.min_annual_icir = 0.8
            th.min_ic_tstat = 2.5
            th.min_ic_positive_ratio = 0.65
            th.min_sharpe_simple = 1.0
            th.min_sharpe_detailed = 0.6
        horizons = [20] if frequency == "M" else [1, 5]
        return cls(
            frequency=frequency,
            thresholds=th,
            icir_check_horizons=horizons,
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"YAML did not parse to dict: {path}")
        # Merge nested thresholds dict into StepThresholds
        th_dict = data.pop("thresholds", {})
        thresholds = StepThresholds(**th_dict)
        return cls(**data, thresholds=thresholds)

    def state_path(self) -> Path:
        return Path(self.results_root) / self.factor_id / "pipeline_state.json"

    def results_dir(self) -> Path:
        return Path(self.results_root) / self.factor_id
