"""Pipeline configuration: thresholds and knobs.

All defaults are read from the global ``config.yaml`` (single source of
truth).  Callers can still override any field at construction time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from backtest.config_loader import get_section


def _pipe_thresholds(section: str, key: str):
    """Helper: read a single threshold from ``config.yaml thresholds.pipeline``."""
    return lambda: get_section("thresholds", "pipeline", section, key)


def _pipe_config(key: str):
    """Helper: read a single value from ``config.yaml pipeline``."""
    return lambda: get_section("pipeline", key)


@dataclass
class StepThresholds:
    """Per-step admission thresholds — sourced from config.yaml."""

    # step1: coverage
    max_missing_rate_pv: float = field(default_factory=_pipe_thresholds("coverage", "max_missing_rate_pv"))
    max_missing_rate_fin: float = field(default_factory=_pipe_thresholds("coverage", "max_missing_rate_fin"))

    # step3: ICIR
    min_abs_ic: float = field(default_factory=_pipe_thresholds("icir", "min_abs_ic"))
    min_annual_icir: float = field(default_factory=_pipe_thresholds("icir", "min_annual_icir"))
    min_ic_tstat: float = field(default_factory=_pipe_thresholds("icir", "min_ic_tstat"))
    min_ic_positive_ratio: float = field(default_factory=_pipe_thresholds("icir", "min_ic_positive_ratio"))

    # step4: monotonicity
    min_monotonicity: float = field(default_factory=_pipe_thresholds("monotonicity", "min_monotonicity"))

    # step6: simple backtest
    min_sharpe_simple: float = field(default_factory=_pipe_thresholds("simple_backtest", "min_sharpe"))
    min_annual_return_simple: float = field(default_factory=_pipe_thresholds("simple_backtest", "min_annual_return"))
    max_max_drawdown: float = field(default_factory=_pipe_thresholds("simple_backtest", "max_max_drawdown"))
    min_calmar_simple: float = field(default_factory=_pipe_thresholds("simple_backtest", "min_calmar"))
    max_annual_turnover: float = field(default_factory=_pipe_thresholds("simple_backtest", "max_annual_turnover"))

    # step7: detailed backtest
    min_sharpe_detailed: float = field(default_factory=_pipe_thresholds("detailed_backtest", "min_sharpe"))
    min_annual_return_detailed: float = field(default_factory=_pipe_thresholds("detailed_backtest", "min_annual_return"))
    min_calmar_detailed: float = field(default_factory=_pipe_thresholds("detailed_backtest", "min_calmar"))


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    factor_id: str
    start_date: str
    end_date: str
    frequency: Literal["D", "M"] = "D"

    # step3: ICIR check horizons
    eval_horizons: list[int] = field(default_factory=_pipe_config("eval_horizons"))
    icir_check_horizons: list[int] = field(default_factory=_pipe_config("icir_check_horizons"))

    # step2: neutralization verification
    max_corr_size: float = field(default_factory=_pipe_config("max_corr_size"))
    max_corr_industry: float = field(default_factory=_pipe_config("max_corr_industry"))
    max_corr_existing: float = field(default_factory=_pipe_config("max_corr_existing"))

    # step5: default strategy config
    default_top_k: int | None = field(default_factory=_pipe_config("default_top_k"))
    default_top_pct: float | None = field(default_factory=_pipe_config("default_top_pct"))
    default_decay: int = field(default_factory=_pipe_config("default_decay"))
    default_rebalance: str = field(default_factory=_pipe_config("default_rebalance"))
    default_universe: str | None = None

    # retry
    max_retries: int = field(default_factory=_pipe_config("max_retries"))

    # thresholds (frequency-aware)
    thresholds: StepThresholds = field(default_factory=StepThresholds)

    # output
    results_root: str = "results"
    ret_type: str = field(default_factory=_pipe_config("ret_type"))
    benchmark: str = field(default_factory=_pipe_config("benchmark"))

    @classmethod
    def for_frequency(
        cls,
        frequency: Literal["D", "M"],
        **kwargs,
    ) -> PipelineConfig:
        """Factory with frequency-specific defaults.

        Monthly overrides are read from ``config.yaml pipeline.monthly``.
        """
        th = StepThresholds()
        if frequency == "M":
            try:
                monthly = get_section("pipeline", "monthly")
                monthly_th = monthly.get("thresholds", {})
                for field_name, val in monthly_th.items():
                    if hasattr(th, field_name):
                        setattr(th, field_name, val)
            except (KeyError, FileNotFoundError):
                # Safe monthly defaults when config.yaml is missing
                th.min_abs_ic = 0.03
                th.min_annual_icir = 0.8
                th.min_ic_tstat = 2.5
                th.min_ic_positive_ratio = 0.65
                th.min_sharpe_simple = 1.0
                th.min_sharpe_detailed = 0.6

        try:
            if frequency == "M":
                horizons = get_section("pipeline", "monthly", "icir_check_horizons")
            else:
                horizons = get_section("pipeline", "icir_check_horizons")
        except (KeyError, FileNotFoundError):
            horizons = [20] if frequency == "M" else [1, 5]

        return cls(
            frequency=frequency,
            thresholds=th,
            icir_check_horizons=horizons,
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
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
            raise ValueError(f"YAML did not parse to dict: {path}")

        # Merge nested thresholds dict into StepThresholds
        th_dict = data.pop("thresholds", {})
        # Flatten nested pipeline thresholds
        flat_th = {}
        for section, vals in th_dict.get("pipeline", {}).items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    # Map YAML keys to dataclass field names
                    field_name = f"min_{k}" if k.startswith("sharpe") or k.startswith("annual_return") or k.startswith("calmar") else k
                    if hasattr(StepThresholds, field_name):
                        flat_th[field_name] = v
                    else:
                        # Try direct mapping
                        flat_th[k] = v
            else:
                flat_th[section] = vals

        thresholds = StepThresholds(**flat_th)

        return cls(
            factor_id=data["factor_id"],
            start_date=data.get("start_date", "20160101"),
            end_date=data.get("end_date", "20251231"),
            frequency=data.get("frequency", "D"),
            eval_horizons=data.get("eval_horizons", [1, 5, 10, 20, 60]),
            icir_check_horizons=data.get("icir_check_horizons", [1, 5]),
            max_corr_size=data.get("max_corr_size", 0.05),
            max_corr_industry=data.get("max_corr_industry", 0.05),
            max_corr_existing=data.get("max_corr_existing", 0.5),
            default_top_k=data.get("default_top_k"),
            default_top_pct=data.get("default_top_pct", 0.1),
            default_decay=data.get("default_decay", 5),
            default_rebalance=data.get("default_rebalance", "1D"),
            default_universe=data.get("default_universe"),
            max_retries=data.get("max_retries", 3),
            thresholds=thresholds,
            results_root=data.get("results_root", "results"),
            ret_type=data.get("ret_type", "open"),
            benchmark=data.get("benchmark", "000300.SH"),
        )

    def state_path(self) -> Path:
        return Path(self.results_root) / self.factor_id / "pipeline_state.json"

    def results_dir(self) -> Path:
        return Path(self.results_root) / self.factor_id
