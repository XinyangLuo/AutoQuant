"""Agent configuration — sourced from the global ``config.yaml``.

All thresholds are now read via ``PipelineConfig`` / ``StepThresholds`` from
``backtest.pipeline.config`` (single source of truth).  This module only
provides agent-specific knobs and date range defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backtest.config_loader import get_section


def _agent(key: str, default: Any = None):
    return lambda: _read_agent(key, default)


def _read_agent(key: str, default: Any = None) -> Any:
    try:
        return get_section("agent", key)
    except (KeyError, FileNotFoundError):
        pass
    try:
        return get_section("pipeline", key)
    except (KeyError, FileNotFoundError):
        return default


_FALLBACKS: dict[str, Any] = {
    "start_date": "20160101",
    "end_date": "20251231",
}


@dataclass
class AgentConfig:
    """Top-level agent configuration.

    Pipeline thresholds and strategy defaults are read from
    ``PipelineConfig`` / ``StepThresholds`` — not duplicated here.
    """

    start_date: str = field(default_factory=_agent("start_date", _FALLBACKS["start_date"]))
    end_date: str = field(default_factory=_agent("end_date", _FALLBACKS["end_date"]))
