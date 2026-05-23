"""Experiment ABC — encapsulates one execution unit."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class Experiment(ABC):
    """Abstract base class for an experiment.

    In AutoQuant an experiment is a single factor: its generated code,
    execution results (factor evaluation + backtest metrics), and status.
    """

    # Identifiers
    experiment_id: str = ""

    # Source code (for factor experiments this is Python code with @register)
    source_code: str = ""
    source_file_path: Path | None = None

    # Execution results
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: Literal["pending", "running", "passed", "rejected", "candidate"] = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "source_code": self.source_code,
            "source_file_path": str(self.source_file_path) if self.source_file_path else None,
            "raw_metrics": self.raw_metrics,
            "extra_metrics": self.extra_metrics,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        return cls(
            experiment_id=data.get("experiment_id", ""),
            source_code=data.get("source_code", ""),
            source_file_path=Path(data["source_file_path"]) if data.get("source_file_path") else None,
            raw_metrics=data.get("raw_metrics", {}),
            extra_metrics=data.get("extra_metrics", {}),
            status=data.get("status", "pending"),
        )
