"""Agent configuration — sourced from the global ``config.yaml``.

All thresholds are now read via ``PipelineConfig`` / ``StepThresholds`` from
``backtest.pipeline.config`` (single source of truth).  Date range is read
from ``pipeline.start_date`` / ``pipeline.end_date`` — same keys as the
manual pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backtest.config_loader import get_section_or


@dataclass
class AgentConfig:
    """Top-level agent configuration.

    Pipeline thresholds and strategy defaults are read from
    ``PipelineConfig`` / ``StepThresholds`` — not duplicated here.
    """

    start_date: str = field(
        default_factory=lambda: get_section_or("20160101", "pipeline", "start_date"),
    )
    end_date: str = field(
        default_factory=lambda: get_section_or("20251231", "pipeline", "end_date"),
    )
