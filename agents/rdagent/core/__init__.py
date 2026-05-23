"""RD-Agent core abstractions — copied from Microsoft RD-Agent.

This module provides the minimal ABC layer needed by the AutoQuant factor
research agent.  No external dependency on the full rdagent package.
"""

from __future__ import annotations

from .evaluation import Evaluator, Feedback
from .evolving_framework import EvolvingStrategy, Trace
from .experiment import Experiment
from .knowledge_base import KnowledgeBase
from .proposal import Hypothesis, Hypothesis2Experiment, HypothesisGen
from .scenario import Scenario

__all__ = [
    "Scenario",
    "Hypothesis",
    "HypothesisGen",
    "Hypothesis2Experiment",
    "Experiment",
    "Evaluator",
    "Feedback",
    "EvolvingStrategy",
    "Trace",
    "KnowledgeBase",
]
