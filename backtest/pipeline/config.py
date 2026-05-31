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

from backtest.config_loader import get_section, get_section_or, load_yaml_file


# Hardcoded defaults used when no per-factor config is found.
_DEFAULT_PIPELINE = {
    "start_date": "20160101",
    "end_date": "20251231",
    "eval_horizons": [1, 5, 10, 20, 60],
    "icir_check_horizons": [1, 5],
    "max_corr_size": 0.05,
    "max_corr_industry": 0.05,
    "max_corr_existing": 0.5,
    "default_top_k": 100,
    "default_top_pct": None,
    "default_decay": 5,
    "default_rebalance": "1D",
    "ret_type": "open",
    "benchmark": "000300.SH",
}

_DEFAULT_STRATEGY = {
    "universe": {
        "exclude_st": True,
        "exclude_new_ipo_days": 252,
        "include_cyb": True,
        "include_kcb": False,
        "include_bse": False,
        "min_market_cap": 500_000_000,
        "min_avg_amount": 10_000_000,
    },
}

_DEFAULT_SIMULATION = {
    "initial_cash": 100_000_000,
    "commission_rate": 0.0003,
    "stamp_duty_rate": 0.001,
    "transfer_fee_rate": 0.00002,
    "allow_short": False,
}


def _find_factor_config_dir(factor_id: str) -> Path | None:
    """Resolve the directory containing the factor's per-factor ``config.yaml``.

    Scans ``alphas/`` following the conventional layout:
    ``alphas/{admitted,exp/user,exp/agent}/<factor_id>/``.
    """
    from backtest.config_loader import _find_project_root

    root = _find_project_root()
    alphas = root / "alphas"
    if not alphas.is_dir():
        return None

    for scope in ("admitted", "exp/user", "exp/agent"):
        candidate = alphas / scope / factor_id
        if candidate.is_dir():
            return candidate
    return None


def _pipe_thresholds(section: str, key: str):
    """Helper: read a single threshold from ``config.yaml thresholds.pipeline``."""
    return lambda: get_section("thresholds", "pipeline", section, key)


def _pipe_thresholds_or(default, section: str, key: str):
    """Helper: read a threshold with fallback default."""
    return lambda: get_section_or(default, "thresholds", "pipeline", section, key)


@dataclass
class StepThresholds:
    """Per-step admission thresholds — sourced from config.yaml."""

    # step1: coverage
    max_missing_rate_pv: float | None = field(default_factory=_pipe_thresholds("coverage", "max_missing_rate_pv"))
    max_missing_rate_fin: float | None = field(default_factory=_pipe_thresholds("coverage", "max_missing_rate_fin"))

    # step3: ICIR
    min_abs_ic: float | None = field(default_factory=_pipe_thresholds("icir", "min_abs_ic"))
    min_annual_icir: float | None = field(default_factory=_pipe_thresholds("icir", "min_annual_icir"))
    min_ic_tstat: float | None = field(default_factory=_pipe_thresholds("icir", "min_ic_tstat"))
    min_ic_positive_ratio: float | None = field(default_factory=_pipe_thresholds("icir", "min_ic_positive_ratio"))

    # step4: monotonicity
    min_monotonicity: float | None = field(default_factory=_pipe_thresholds("monotonicity", "min_monotonicity"))

    # step6: simple backtest — absolute
    min_sharpe_simple: float | None = field(default_factory=_pipe_thresholds("simple_backtest", "min_sharpe"))
    min_annual_return_simple: float | None = field(default_factory=_pipe_thresholds("simple_backtest", "min_annual_return"))
    max_max_drawdown: float | None = field(default_factory=_pipe_thresholds("simple_backtest", "max_max_drawdown"))
    min_calmar_simple: float | None = field(default_factory=_pipe_thresholds("simple_backtest", "min_calmar"))
    max_annual_turnover: float | None = field(default_factory=_pipe_thresholds("simple_backtest", "max_annual_turnover"))

    # step6: simple backtest — relative HS300
    min_excess_sharpe_simple_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_sharpe_hs300"))
    min_excess_annual_return_simple_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_annual_return_hs300"))
    max_excess_max_drawdown_simple_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "max_excess_max_drawdown_hs300"))
    min_excess_calmar_simple_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_calmar_hs300"))

    # step6: simple backtest — relative CSI500
    min_excess_sharpe_simple_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_sharpe_csi500"))
    min_excess_annual_return_simple_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_annual_return_csi500"))
    max_excess_max_drawdown_simple_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "max_excess_max_drawdown_csi500"))
    min_excess_calmar_simple_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_calmar_csi500"))

    # step6: simple backtest — relative CSI1000
    min_excess_sharpe_simple_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_sharpe_csi1000"))
    min_excess_annual_return_simple_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_annual_return_csi1000"))
    max_excess_max_drawdown_simple_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "max_excess_max_drawdown_csi1000"))
    min_excess_calmar_simple_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "simple_backtest", "min_excess_calmar_csi1000"))

    # step7: detailed backtest — absolute
    min_sharpe_detailed: float | None = field(default_factory=_pipe_thresholds("detailed_backtest", "min_sharpe"))
    min_annual_return_detailed: float | None = field(default_factory=_pipe_thresholds("detailed_backtest", "min_annual_return"))
    min_calmar_detailed: float | None = field(default_factory=_pipe_thresholds("detailed_backtest", "min_calmar"))
    max_max_drawdown_detailed: float | None = field(default_factory=_pipe_thresholds("detailed_backtest", "max_max_drawdown"))
    max_annual_turnover_detailed: float | None = field(default_factory=_pipe_thresholds("detailed_backtest", "max_annual_turnover"))

    # step7: detailed backtest — relative HS300
    min_excess_sharpe_detailed_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_sharpe_hs300"))
    min_excess_annual_return_detailed_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_annual_return_hs300"))
    max_excess_max_drawdown_detailed_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "max_excess_max_drawdown_hs300"))
    min_excess_calmar_detailed_hs300: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_calmar_hs300"))

    # step7: detailed backtest — relative CSI500
    min_excess_sharpe_detailed_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_sharpe_csi500"))
    min_excess_annual_return_detailed_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_annual_return_csi500"))
    max_excess_max_drawdown_detailed_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "max_excess_max_drawdown_csi500"))
    min_excess_calmar_detailed_csi500: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_calmar_csi500"))

    # step7: detailed backtest — relative CSI1000
    min_excess_sharpe_detailed_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_sharpe_csi1000"))
    min_excess_annual_return_detailed_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_annual_return_csi1000"))
    max_excess_max_drawdown_detailed_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "max_excess_max_drawdown_csi1000"))
    min_excess_calmar_detailed_csi1000: float | None = field(default_factory=_pipe_thresholds_or(None, "detailed_backtest", "min_excess_calmar_csi1000"))


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    Defaults come from the per-factor ``config.yaml`` when available,
    otherwise fall back to ``_DEFAULT_PIPELINE`` hardcoded values.
    Global thresholds remain in root ``config.yaml``.
    """

    factor_id: str
    start_date: str
    end_date: str
    frequency: Literal["D", "M"] = "D"

    # step3: ICIR check horizons
    eval_horizons: list[int] = field(default_factory=lambda: list(_DEFAULT_PIPELINE["eval_horizons"]))
    icir_check_horizons: list[int] = field(default_factory=lambda: list(_DEFAULT_PIPELINE["icir_check_horizons"]))

    # step2: neutralization verification
    max_corr_size: float = field(default_factory=lambda: _DEFAULT_PIPELINE["max_corr_size"])
    max_corr_industry: float = field(default_factory=lambda: _DEFAULT_PIPELINE["max_corr_industry"])
    max_corr_existing: float = field(default_factory=lambda: _DEFAULT_PIPELINE["max_corr_existing"])

    # step5: default strategy config
    default_top_k: int | None = field(default_factory=lambda: _DEFAULT_PIPELINE["default_top_k"])
    default_top_pct: float | None = field(default_factory=lambda: _DEFAULT_PIPELINE["default_top_pct"])
    default_decay: int = field(default_factory=lambda: _DEFAULT_PIPELINE["default_decay"])
    default_rebalance: str = field(default_factory=lambda: _DEFAULT_PIPELINE["default_rebalance"])
    default_universe: str | None = None

    # thresholds (frequency-aware)
    thresholds: StepThresholds = field(default_factory=StepThresholds)

    # output
    results_root: str = "results"
    ret_type: str = field(default_factory=lambda: _DEFAULT_PIPELINE["ret_type"])
    benchmark: str = field(default_factory=lambda: _DEFAULT_PIPELINE["benchmark"])

    # Per-factor strategy & simulation overrides (populated by from_factor_config).
    strategy_overrides: dict | None = None
    simulation_overrides: dict | None = None

    @classmethod
    def for_frequency(
        cls,
        frequency: Literal["D", "M"],
        **kwargs,
    ) -> PipelineConfig:
        """Factory with frequency-specific defaults.

        Monthly overrides are hardcoded; per-factor config can further
        adjust them via ``from_factor_config``.
        """
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
            icir_check_horizons=list(horizons),
            strategy_overrides=kwargs.pop("strategy_overrides", None),
            simulation_overrides=kwargs.pop("simulation_overrides", None),
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
            default_top_pct=data.get("default_top_pct"),
            default_decay=data.get("default_decay", 5),
            default_rebalance=data.get("default_rebalance", "1D"),
            default_universe=data.get("default_universe"),
            thresholds=thresholds,
            results_root=data.get("results_root", "results"),
            ret_type=data.get("ret_type", "open"),
            benchmark=data.get("benchmark", "000300.SH"),
        )

    @classmethod
    def from_factor_config(
        cls,
        factor_id: str,
        *,
        frequency: Literal["D", "M"] = "D",
        **overrides,
    ) -> PipelineConfig:
        """Build config by merging per-factor ``config.yaml`` with hardcoded defaults.

        Looks for ``<factor_dir>/config.yaml``.  The file can contain
        ``pipeline``, ``strategy``, and ``simulation`` sections (all optional).
        ``**overrides`` take highest priority.
        """
        cfg: dict[str, Any] = {}
        config_dir = _find_factor_config_dir(factor_id)
        if config_dir is not None:
            config_file = config_dir / "config.yaml"
            if config_file.exists():
                cfg = load_yaml_file(config_file)

        pipe = {**_DEFAULT_PIPELINE, **cfg.get("pipeline", {})}
        strat = {**_DEFAULT_STRATEGY, **cfg.get("strategy", {})}
        sim = {**_DEFAULT_SIMULATION, **cfg.get("simulation", {})}

        th = StepThresholds()
        if frequency == "M":
            th.min_abs_ic = 0.03
            th.min_annual_icir = 0.8
            th.min_ic_tstat = 2.5
            th.min_ic_positive_ratio = 0.65
            th.min_sharpe_simple = 1.0
            th.min_sharpe_detailed = 0.6

        horizons = [20] if frequency == "M" else pipe["icir_check_horizons"]

        # Pop keys that are set explicitly so they don't collide with **overrides.
        results_root_val = str(overrides.pop("results_root", "results"))

        return cls(
            factor_id=factor_id,
            start_date=str(overrides.pop("start_date", pipe["start_date"])),
            end_date=str(overrides.pop("end_date", pipe["end_date"])),
            frequency=frequency,
            eval_horizons=list(pipe["eval_horizons"]),
            icir_check_horizons=list(horizons) if frequency == "M" else list(pipe["icir_check_horizons"]),
            max_corr_size=pipe["max_corr_size"],
            max_corr_industry=pipe["max_corr_industry"],
            max_corr_existing=pipe["max_corr_existing"],
            default_top_k=pipe["default_top_k"],
            default_top_pct=pipe["default_top_pct"],
            default_decay=pipe["default_decay"],
            default_rebalance=pipe["default_rebalance"],
            thresholds=th,
            results_root=results_root_val,
            ret_type=str(overrides.pop("ret_type", pipe["ret_type"])),
            benchmark=str(overrides.pop("benchmark", pipe["benchmark"])),
            strategy_overrides=strat,
            simulation_overrides=sim,
            **overrides,
        )

    def state_path(self) -> Path:
        return Path(self.results_root) / self.factor_id / "pipeline_state.json"

    def results_dir(self) -> Path:
        return Path(self.results_root) / self.factor_id
