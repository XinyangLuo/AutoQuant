"""Scenario ABC — describes a quantitative research domain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Scenario(ABC):
    """Abstract base class for a research scenario / domain.

    A Scenario provides the context an Agent needs to generate meaningful
    hypotheses: data schema, trading rules, evaluation criteria, available
    operators, etc.
    """

    #: Directory containing markdown prompt templates.
    _prompt_dir: Path = Path(".")

    @abstractmethod
    def get_data_schema(self) -> dict[str, Any]:
        """Return the data schema available to factors.

        Returns
        -------
        dict
            Mapping of table names to column descriptions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trading_rules(self) -> dict[str, Any]:
        """Return trading rules specific to this market.

        Returns
        -------
        dict
            Rules such as T+1, price limits, ST handling, IPO exclusions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_evaluation_criteria(self) -> dict[str, Any]:
        """Return evaluation thresholds and criteria.

        Returns
        -------
        dict
            Metric names and their target thresholds.
        """
        raise NotImplementedError

    @abstractmethod
    def get_factor_categories(self) -> list[str]:
        """Return the taxonomy of factor categories."""
        raise NotImplementedError

    @abstractmethod
    def get_available_operators(self) -> list[str]:
        """Return the list of transforms / operators available to factors."""
        raise NotImplementedError

    @abstractmethod
    def get_neutralization_options(self) -> list[str]:
        """Return available neutralization / variant pipelines."""
        raise NotImplementedError
