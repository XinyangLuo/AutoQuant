"""Agent configuration — sourced from the global ``config.yaml``.

All thresholds are read via ``backtest.config_loader`` so they stay in sync
with the pipeline / admission configs.  Callers can override any field at
construction time.

**Candidate thresholds** (RankICIR / IC+ / turnover / max-corr / horizon /
ret-type) are drawn from ``thresholds.admission`` — the agent's candidate bar
is the same as the admission reference bar, so they are not duplicated in
``thresholds.agent``.

**Backtest thresholds** (Sharpe / drawdown / Calmar / turnover) are drawn from
``thresholds.agent`` where they align with the pipeline gates.
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
    "ret_type": "close",
    "exclude_limit_up": True,
}

_AGENT_FALLBACKS: dict[str, Any] = {
    "min_sharpe_simple": 0.5,
    "min_sharpe_detailed": 0.5,
    "min_annual_return_detailed": 0.05,
    "max_max_drawdown": 0.25,
    "min_calmar_simple": 0.5,
    "max_annual_turnover": 10.0,
    "high_bar_sharpe": 1.0,
}


def _adm(key: str):
    """Helper: read from ``config.yaml thresholds.admission``."""
    def _factory():
        try:
            return get_section("thresholds", "admission", key)
        except (KeyError, FileNotFoundError):
            return _ADM_FALLBACKS[key]
    return _factory


def _agent_th(key: str):
    """Helper: read from ``config.yaml thresholds.agent``."""
    def _factory():
        try:
            return get_section("thresholds", "agent", key)
        except (KeyError, FileNotFoundError):
            return _AGENT_FALLBACKS[key]
    return _factory


def _agent_root(key: str):
    """Helper: read from ``config.yaml agent`` (root level)."""
    def _factory():
        try:
            return get_section("agent", key)
        except (KeyError, FileNotFoundError):
            return None
    return _factory


@dataclass
class AgentConfig:
    """Top-level agent configuration.

    Candidate thresholds come from ``thresholds.admission`` (single source).
    Backtest / agent-specific thresholds come from ``thresholds.agent``.
    """

    # ---- candidate thresholds (from thresholds.admission) -----------------
    min_rankicir: float = field(default_factory=_adm("min_rankicir"))
    min_ic_positive_ratio: float = field(default_factory=_adm("min_ic_positive_ratio"))
    max_turnover: float = field(default_factory=_adm("max_turnover"))
    max_corr: float = field(default_factory=_adm("max_corr"))
    primary_horizon: int = field(default_factory=_adm("primary_horizon"))
    ret_type: str = field(default_factory=_adm("ret_type"))
    exclude_limit_up: bool = field(default_factory=_adm("exclude_limit_up"))

    # ---- backtest thresholds (from thresholds.agent) ----------------------
    min_sharpe_simple: float = field(default_factory=_agent_th("min_sharpe_simple"))
    min_sharpe_detailed: float = field(default_factory=_agent_th("min_sharpe_detailed"))
    min_annual_return_detailed: float = field(default_factory=_agent_th("min_annual_return_detailed"))
    max_max_drawdown: float = field(default_factory=_agent_th("max_max_drawdown"))
    min_calmar_simple: float = field(default_factory=_agent_th("min_calmar_simple"))
    max_annual_turnover: float = field(default_factory=_agent_th("max_annual_turnover"))

    # ---- agent-specific ---------------------------------------------------
    #: Candidate threshold for simple Sharpe (alias, same as pipeline step6).
    min_simple_sharpe: float = field(default_factory=_agent_th("min_sharpe_simple"))

    #: High-bar early-stop threshold.
    high_bar_sharpe: float = field(default_factory=_agent_th("high_bar_sharpe"))

    #: Frequency-aware factory — see ``PipelineConfig.for_frequency``.
    frequency: str = "D"

    @classmethod
    def from_pipeline_config(cls, pipeline_config: Any) -> "AgentConfig":
        """Build from an existing ``PipelineConfig`` instance."""
        th = pipeline_config.thresholds

        def _safe_adm(key: str):
            try:
                return get_section("thresholds", "admission", key)
            except (KeyError, FileNotFoundError):
                return _ADM_FALLBACKS[key]

        return cls(
            min_rankicir=_safe_adm("min_rankicir"),
            min_ic_positive_ratio=_safe_adm("min_ic_positive_ratio"),
            max_turnover=_safe_adm("max_turnover"),
            max_corr=pipeline_config.max_corr_existing,
            primary_horizon=_safe_adm("primary_horizon"),
            ret_type=pipeline_config.ret_type,
            min_sharpe_simple=th.min_sharpe_simple,
            min_simple_sharpe=th.min_sharpe_simple,
            max_max_drawdown=th.max_max_drawdown,
            min_calmar_simple=th.min_calmar_simple,
            max_annual_turnover=th.max_annual_turnover,
            min_sharpe_detailed=th.min_sharpe_detailed,
            min_annual_return_detailed=th.min_annual_return_detailed,
            frequency=pipeline_config.frequency,
        )
