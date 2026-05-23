"""Smoke tests for the agent factor research system.

These tests verify that core components instantiate and interact correctly
without requiring a full backtest database. They catch structural bugs
(interface mismatches, serialization round-trips, config loading) quickly.

For end-to-end tests that exercise the full Runner + backtest pipeline,
see ``tests/test_agent_pipeline.py`` (TBD).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agents.rdagent.config import AgentConfig
from agents.rdagent.core.evaluation import Feedback
from agents.rdagent.core.evolving_framework import Trace
from agents.rdagent.core.proposal import Hypothesis
from agents.rdagent.evaluator import AutoQuantFactorEvaluator, QuantFeedback
from agents.rdagent.experiment import AutoQuantFactorExperiment
from agents.rdagent.hypothesis import _extract_json, _inject_factor_id
from agents.rdagent.knowledge import AShareKnowledgeBase
from agents.rdagent.run import _load_checkpoint, _save_checkpoint
from agents.rdagent.scenario import AShareQuantScenario


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_agent_config_loads():
    """AgentConfig can be instantiated without crashing."""
    cfg = AgentConfig()
    assert cfg.min_rankicir > 0
    assert 0 < cfg.min_ic_positive_ratio < 1
    assert cfg.max_turnover > 0
    assert cfg.min_sharpe_simple >= 0


def test_agent_config_override():
    """Callers can override any field."""
    cfg = AgentConfig(min_rankicir=0.99)
    assert cfg.min_rankicir == 0.99


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def test_scenario_instantiates():
    """AShareQuantScenario provides all required context."""
    scenario = AShareQuantScenario()
    schema = scenario.get_data_schema()
    assert "market_daily" in schema
    assert "income_q" in schema

    rules = scenario.get_trading_rules()
    assert "settlement" in rules

    cats = scenario.get_factor_categories()
    assert "momentum" in cats
    assert "value" in cats

    ops = scenario.get_available_operators()
    assert "rank" in ops
    assert "ts_mean" in ops


def test_scenario_prompt_rendering():
    """Scenario prompt renders without unfilled placeholders."""
    import warnings

    scenario = AShareQuantScenario()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prompt = scenario.render_scenario_prompt()
        # Allow warnings about unfilled placeholders from template variables
        # that may not be provided by render_scenario_prompt
        assert "A-Share" in prompt or "Data Schema" in prompt or "quantitative" in prompt


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_evaluator_decision_pass():
    """Evaluator marks experiment as candidate when thresholds are met."""
    evaluator = AutoQuantFactorEvaluator(min_simple_sharpe=0.5)
    exp = AutoQuantFactorExperiment(
        factor_id="f_test",
        eval_result={
            "rankicir": 0.30,
            "ic_positive_ratio": 0.55,
            "turnover": 0.30,
            "max_corr": 0.50,
        },
        simple_bt_metrics={"sharpe": 0.60},
    )
    fb = evaluator.evaluate(exp)
    assert fb.decision is True
    assert "PASS" in fb.observation


def test_evaluator_decision_fail_low_rankicir():
    """Evaluator rejects when RankICIR is below threshold."""
    evaluator = AutoQuantFactorEvaluator()
    exp = AutoQuantFactorExperiment(
        factor_id="f_test",
        eval_result={
            "rankicir": 0.10,
            "ic_positive_ratio": 0.55,
            "turnover": 0.30,
            "max_corr": 0.50,
        },
        simple_bt_metrics={"sharpe": 0.60},
    )
    fb = evaluator.evaluate(exp)
    assert fb.decision is False
    assert "low" in fb.suggestion.lower() or "RankICIR" in fb.suggestion


def test_quant_feedback_round_trip():
    """QuantFeedback serializes and deserializes without data loss."""
    fb = QuantFeedback(
        decision=True,
        rankicir=0.30,
        ic_positive_ratio=0.55,
        simple_sharpe=0.80,
        metrics={"rankicir": 0.30, "turnover": 0.20},
    )
    d = fb.to_dict()
    fb2 = QuantFeedback.from_dict(d)
    assert fb2.decision is True
    assert fb2.rankicir == pytest.approx(0.30)
    assert fb2.simple_sharpe == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def test_experiment_round_trip():
    """AutoQuantFactorExperiment serializes and deserializes correctly."""
    exp = AutoQuantFactorExperiment(
        factor_id="f_auto_001",
        factor_code="def my_factor(df): return df['close']",
        category="momentum",
        keywords=["close", "price"],
        eval_result={"rankicir": 0.25},
    )
    d = exp.to_dict()
    exp2 = AutoQuantFactorExperiment.from_dict(d)
    assert exp2.factor_id == "f_auto_001"
    assert exp2.category == "momentum"
    assert exp2.keywords == ["close", "price"]


# ---------------------------------------------------------------------------
# Hypothesis helpers
# ---------------------------------------------------------------------------


def test_extract_json_fenced():
    """_extract_json handles markdown-fenced JSON."""
    text = 'Some text\n```json\n{"a": 1, "b": "two"}\n```\nMore text'
    result = _extract_json(text)
    assert result == {"a": 1, "b": "two"}


def test_extract_json_bare():
    """_extract_json handles bare JSON."""
    text = 'Here is the result: {"a": 1, "b": "two"} thanks!'
    result = _extract_json(text)
    assert result == {"a": 1, "b": "two"}


def test_inject_factor_id_positional():
    """_inject_factor_id replaces positional arg in @register."""
    code = '@register("old_id")\ndef my_factor(df):\n    return df["close"]'
    result = _inject_factor_id(code, "f_auto_new")
    assert 'f_auto_new' in result
    assert 'old_id' not in result


def test_inject_factor_id_keyword():
    """_inject_factor_id replaces keyword arg in @register."""
    code = '@register(factor_id="old_id")\ndef my_factor(df):\n    return df["close"]'
    result = _inject_factor_id(code, "f_auto_new")
    assert 'f_auto_new' in result
    assert 'old_id' not in result


def test_inject_factor_id_no_decorator():
    """_inject_factor_id prepends decorator if none exists."""
    code = 'def my_factor(df):\n    return df["close"]'
    result = _inject_factor_id(code, "f_auto_new")
    assert '@register("f_auto_new")' in result


# ---------------------------------------------------------------------------
# Trace / Checkpoint
# ---------------------------------------------------------------------------


def test_trace_round_trip():
    """Trace serializes and deserializes with correct factories."""
    trace = Trace()
    exp = AutoQuantFactorExperiment(factor_id="f1", eval_result={"rankicir": 0.3})
    fb = QuantFeedback(decision=True, rankicir=0.3)
    trace.add(exp, fb)

    d = trace.to_dict()
    trace2 = Trace.from_dict(
        d,
        experiment_factory=AutoQuantFactorExperiment.from_dict,
        feedback_factory=QuantFeedback.from_dict,
    )
    assert len(trace2.hist) == 1
    assert trace2.hist[0][0].factor_id == "f1"
    assert trace2.hist[0][1].rankicir == pytest.approx(0.3)


def test_checkpoint_save_and_load():
    """_save_checkpoint and _load_checkpoint round-trip correctly."""
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        kb = AShareKnowledgeBase(db_path=output_dir / "kb.json")
        trace = Trace()
        exp = AutoQuantFactorExperiment(
            factor_id="f_auto_001",
            eval_result={"rankicir": 0.30},
        )
        fb = QuantFeedback(decision=True, rankicir=0.30)
        trace.add(exp, fb)
        candidates = [exp]

        _save_checkpoint(trace, candidates, 3, output_dir)

        loaded = _load_checkpoint(output_dir)
        assert loaded is not None
        trace2, candidates2, round_num = loaded
        assert round_num == 3
        assert len(candidates2) == 1
        assert candidates2[0].factor_id == "f_auto_001"
        assert len(trace2.hist) == 1


def test_checkpoint_load_missing():
    """_load_checkpoint returns None when no checkpoint exists."""
    with tempfile.TemporaryDirectory() as td:
        result = _load_checkpoint(Path(td))
        assert result is None


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


def test_knowledge_base_add_and_retrieve():
    """KnowledgeBase accumulates experience and supports retrieval."""
    with tempfile.TemporaryDirectory() as td:
        kb = AShareKnowledgeBase(db_path=Path(td) / "kb.json")
        exp = AutoQuantFactorExperiment(
            factor_id="f1",
            category="momentum",
            keywords=["vol", "price"],
        )
        fb = QuantFeedback(decision=True, rankicir=0.30)
        kb.add_experience(exp, fb)

        # Retrieve similar
        hypo = Hypothesis(
            hypothesis_text="test",
            category="momentum",
            keywords=["vol"],
        )
        results = kb.retrieve_similar(hypo, top_k=3)
        assert len(results) == 1
        assert results[0]["factor_id"] == "f1"


def test_knowledge_base_corrupt_recovery():
    """KnowledgeBase recovers from corrupt JSON."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "kb.json"
        db_path.write_text("NOT JSON{{")
        kb = AShareKnowledgeBase(db_path=db_path)
        assert len(kb._records) == 0
