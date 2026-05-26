"""AutoQuant factor experiment — tracks one generated factor through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class AutoQuantFactorExperiment:
    """A factor experiment: generated code + execution results + status."""

    # Factor identity
    factor_id: str = ""
    factor_code: str = ""
    factor_file_path: Path | None = None

    # Factor evaluation (from backtest.factor.evaluate)
    eval_result: dict[str, Any] = field(default_factory=dict)

    # Simple backtest metrics
    simple_bt_metrics: dict[str, Any] | None = None
    simple_bt_dir: Path | None = None

    # Detailed backtest metrics
    detailed_bt_metrics: dict[str, Any] | None = None
    detailed_bt_dir: Path | None = None

    # Hypothesis metadata (for knowledge base / trace)
    category: str = ""
    keywords: list[str] = field(default_factory=list)

    # Pipeline / runner state
    error: str | None = None

    # Status tracking
    status: Literal["pending", "running", "passed", "rejected", "candidate"] = "pending"

    def __post_init__(self) -> None:
        # Legacy compat: alias for code that uses experiment_id / source_code
        object.__setattr__(self, "experiment_id", self.factor_id)
        object.__setattr__(self, "source_code", self.factor_code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "factor_code": self.factor_code,
            "factor_file_path": str(self.factor_file_path) if self.factor_file_path else None,
            "eval_result": self.eval_result,
            "simple_bt_metrics": self.simple_bt_metrics,
            "simple_bt_dir": str(self.simple_bt_dir) if self.simple_bt_dir else None,
            "detailed_bt_metrics": self.detailed_bt_metrics,
            "detailed_bt_dir": str(self.detailed_bt_dir) if self.detailed_bt_dir else None,
            "category": self.category,
            "keywords": self.keywords,
            "error": self.error,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoQuantFactorExperiment":
        return cls(
            factor_id=data.get("factor_id", ""),
            factor_code=data.get("factor_code", ""),
            factor_file_path=Path(data["factor_file_path"]) if data.get("factor_file_path") else None,
            eval_result=data.get("eval_result", {}),
            simple_bt_metrics=data.get("simple_bt_metrics"),
            simple_bt_dir=Path(data["simple_bt_dir"]) if data.get("simple_bt_dir") else None,
            detailed_bt_metrics=data.get("detailed_bt_metrics"),
            detailed_bt_dir=Path(data["detailed_bt_dir"]) if data.get("detailed_bt_dir") else None,
            category=data.get("category", ""),
            keywords=data.get("keywords", []),
            error=data.get("error"),
            status=data.get("status", "pending"),
        )
