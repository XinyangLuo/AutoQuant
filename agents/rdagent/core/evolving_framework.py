"""Evolving framework — Trace and EvolvingStrategy ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evaluation import Feedback
    from .experiment import Experiment


@dataclass
class Trace:
    """History of experiments and their feedback.

    The Trace is passed to ``HypothesisGen.gen()`` so the agent can
    learn from past successes and failures.
    """

    hist: list[tuple["Experiment", "Feedback"]] = field(default_factory=list)

    def add(self, experiment: "Experiment", feedback: "Feedback") -> None:
        self.hist.append((experiment, feedback))

    def last_n(self, n: int) -> list[tuple["Experiment", "Feedback"]]:
        return self.hist[-n:]

    def successes(self) -> list[tuple["Experiment", "Feedback"]]:
        return [(e, f) for e, f in self.hist if f.decision]

    def failures(self) -> list[tuple["Experiment", "Feedback"]]:
        return [(e, f) for e, f in self.hist if not f.decision]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hist": [
                {"experiment": e.to_dict(), "feedback": f.to_dict()}
                for e, f in self.hist
            ]
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        experiment_factory=None,
        feedback_factory=None,
    ) -> "Trace":
        from .evaluation import Feedback
        from .experiment import Experiment

        exp_factory = experiment_factory or Experiment.from_dict
        fb_factory = feedback_factory or Feedback.from_dict

        trace = cls()
        for item in data.get("hist", []):
            exp = exp_factory(item["experiment"])
            fb = fb_factory(item["feedback"])
            trace.add(exp, fb)
        return trace


class EvolvingStrategy(ABC):
    """Abstract base class for an evolving / iterative strategy.

    In practice the main loop in ``run.py`` acts as the evolving strategy.
    This ABC is reserved for future extensions (e.g. multi-objective
    evolution, genetic algorithms, etc.).
    """

    @abstractmethod
    def evolve(self, trace: Trace) -> "Experiment":
        """Produce the next experiment given the current trace.

        Parameters
        ----------
        trace : Trace

        Returns
        -------
        Experiment
        """
        raise NotImplementedError
