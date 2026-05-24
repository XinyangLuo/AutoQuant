"""AutoQuant-specific Experiment implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backtest.factor.evaluation import EvaluationResult

from .core.experiment import Experiment


@dataclass
class AutoQuantFactorExperiment(Experiment):
    """A factor experiment in the AutoQuant system.

    Extends the base :class:`Experiment` with AutoQuant-specific fields:
    factor_id, generated Python code, evaluation results, and backtest metrics.
    """

    # Factor identity
    factor_id: str = ""
    factor_code: str = ""  # Python code with @register decorator
    factor_file_path: Path | None = None

    # Factor evaluation (from backtest.factor.evaluate)
    eval_result: dict[str, Any] = field(default_factory=dict)

    # Simple backtest metrics (from backtest.evaluation.evaluate)
    simple_bt_metrics: dict[str, Any] | None = None
    simple_bt_dir: Path | None = None

    # Detailed backtest metrics (from backtest.evaluation.evaluate)
    detailed_bt_metrics: dict[str, Any] | None = None
    detailed_bt_dir: Path | None = None

    # Hypothesis metadata (for knowledge base retrieval)
    category: str = ""
    keywords: list[str] = field(default_factory=list)

    # Pipeline / runner state
    error: str | None = None

    def __post_init__(self) -> None:
        # Sync with base class fields
        if self.factor_id and not self.experiment_id:
            self.experiment_id = self.factor_id
        if self.factor_code and not self.source_code:
            self.source_code = self.factor_code
        if self.factor_file_path and not self.source_file_path:
            self.source_file_path = self.factor_file_path

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoQuantFactorExperiment":
        base = super().from_dict(data)
        obj = cls(
            experiment_id=base.experiment_id,
            source_code=base.source_code,
            source_file_path=base.source_file_path,
            raw_metrics=base.raw_metrics,
            extra_metrics=base.extra_metrics,
            status=base.status,
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
        )
        obj.__post_init__()
        return obj
