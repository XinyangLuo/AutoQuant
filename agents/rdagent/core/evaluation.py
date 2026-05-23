"""Evaluation layer — Evaluator ABC and Feedback dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .experiment import Experiment


@dataclass
class Feedback:
    """Structured feedback from an evaluator.

    This is consumed by the next hypothesis generation round to improve
    the agent's suggestions.
    """

    # Core decision
    decision: bool = False  # True = meets candidate threshold

    # Natural-language summary for the LLM
    observation: str = ""  # What happened (metrics, pass/fail reasons)
    suggestion: str = ""  # What to try next

    # Optional structured fields that subclasses may fill
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "observation": self.observation,
            "suggestion": self.suggestion,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Feedback":
        return cls(
            decision=data.get("decision", False),
            observation=data.get("observation", ""),
            suggestion=data.get("suggestion", ""),
            metrics=data.get("metrics", {}),
        )


class Evaluator(ABC):
    """Abstract base class for experiment evaluation."""

    @abstractmethod
    def evaluate(self, experiment: "Experiment") -> Feedback:
        """Evaluate an experiment and return structured feedback.

        Parameters
        ----------
        experiment : Experiment

        Returns
        -------
        Feedback
        """
        raise NotImplementedError
