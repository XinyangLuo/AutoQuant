"""Agent configuration — sourced from the global ``config.yaml``.

All thresholds are read via ``backtest.config_loader`` so they stay in sync
with the pipeline / admission configs.  Callers can override any field at
construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backtest.config_loader import get_section


# Fallback defaults when config.yaml is missing or key not found.
_ADM_FALLBACKS: dict[str, Any] = {
    "min_rankicir": 0.25,
    "min_ic_positive_ratio": 0.52,
    "max_turnover": 0.5,
    "max_corr": 0.85,
    "primary_horizon": 20,
    "ret_type": "open",
    "exclude_limit_up": True,
}

_PIPELINE_FALLBACKS: dict[str, Any] = {
    "simple_backtest": {
        "min_sharpe": 0.8,
        "min_annual_return": 0.10,
        "max_max_drawdown": 0.40,
        "min_calmar": 0.5,
        "max_annual_turnover": 20.0,
    },
    "detailed_backtest": {
        "min_sharpe": 0.4,
        "min_annual_return": 0.08,
        "min_calmar": 0.5,
    },
}

_AGENT_FALLBACKS: dict[str, Any] = {
    "high_bar_sharpe": 1.0,
    "start_date": "20160101",
    "end_date": "20251231",
}


def _adm(key: str):
    def _factory():
        try:
            return get_section("thresholds", "admission", key)
        except (KeyError, FileNotFoundError):
            return _ADM_FALLBACKS[key]
    return _factory


def _pipe(section: str, key: str):
    def _factory():
        try:
            return get_section("thresholds", "pipeline", section, key)
        except (KeyError, FileNotFoundError):
            try:
                return _PIPELINE_FALLBACKS[section][key]
            except KeyError:
                raise KeyError(
                    f"Config key thresholds.pipeline.{section}.{key} not found "
                    f"and no fallback defined"
                )
    return _factory


def _agent_root(key: str):
    def _factory():
        try:
            return get_section("agent", key)
        except (KeyError, FileNotFoundError):
            return _AGENT_FALLBACKS.get(key)
    return _factory


@dataclass
class AgentConfig:
    """Top-level agent configuration.

    Candidate thresholds  -> ``thresholds.admission``.
    Backtest thresholds   -> ``thresholds.pipeline``.
    Agent-only knobs      -> root ``agent`` key.
    """

    # ---- candidate thresholds (from thresholds.admission) -----------------
    min_rankicir: float = field(default_factory=_adm("min_rankicir"))
    min_ic_positive_ratio: float = field(default_factory=_adm("min_ic_positive_ratio"))
    max_turnover: float = field(default_factory=_adm("max_turnover"))
    max_corr: float = field(default_factory=_adm("max_corr"))
    primary_horizon: int = field(default_factory=_adm("primary_horizon"))
    ret_type: str = field(default_factory=_adm("ret_type"))
    exclude_limit_up: bool = field(default_factory=_adm("exclude_limit_up"))

    # ---- backtest thresholds (from thresholds.pipeline) -------------------
    min_sharpe_simple: float = field(default_factory=_pipe("simple_backtest", "min_sharpe"))
    min_sharpe_detailed: float = field(default_factory=_pipe("detailed_backtest", "min_sharpe"))
    min_annual_return_detailed: float = field(
        default_factory=_pipe("detailed_backtest", "min_annual_return")
    )
    max_max_drawdown: float = field(
        default_factory=_pipe("simple_backtest", "max_max_drawdown")
    )
    min_calmar_simple: float = field(default_factory=_pipe("simple_backtest", "min_calmar"))
    max_annual_turnover: float = field(
        default_factory=_pipe("simple_backtest", "max_annual_turnover")
    )

    # ---- agent-specific ---------------------------------------------------
    #: High-bar early-stop threshold (from root ``agent:`` key).
    high_bar_sharpe: float = field(default_factory=_agent_root("high_bar_sharpe"))

    #: Backtest date range (from root ``agent:`` key).
    start_date: str = field(default_factory=_agent_root("start_date"))
    end_date: str = field(default_factory=_agent_root("end_date"))

    #: Frequency-aware factory — see ``PipelineConfig.for_frequency``.
    frequency: str = "D"
