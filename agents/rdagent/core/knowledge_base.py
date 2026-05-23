"""KnowledgeBase ABC — domain knowledge accumulation and retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evaluation import Feedback
    from .experiment import Experiment
    from .proposal import Hypothesis


class KnowledgeBase(ABC):
    """Abstract base class for a domain knowledge base.

    A KnowledgeBase accumulates experience from past experiments and
    supports retrieval of similar cases to guide hypothesis generation.
    """

    @abstractmethod
    def add_experience(
        self,
        experiment: "Experiment",
        feedback: "Feedback",
    ) -> None:
        """Record the outcome of one experiment.

        Parameters
        ----------
        experiment : Experiment
        feedback : Feedback
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_similar(
        self,
        hypothesis: "Hypothesis",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve past experiences similar to the given hypothesis.

        Parameters
        ----------
        hypothesis : Hypothesis
        top_k : int

        Returns
        -------
        list[dict]
        """
        raise NotImplementedError

    @abstractmethod
    def get_sota(self) -> dict[str, Any]:
        """Return the current best-known performance (state-of-the-art).

        Returns
        -------
        dict
            Best metrics seen so far.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        """Persist the knowledge base to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        """Load the knowledge base from disk."""
        raise NotImplementedError
