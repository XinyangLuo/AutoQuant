"""Tests for agent automation modules: trace, kb_update, evaluator layering."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agents.evaluator import (
    AutoQuantFactorEvaluator,
    EvaluationFeedback,
    ExecutionFeedback,
    HypothesisFeedback,
    QuantFeedback,
)
from agents.experiment import AutoQuantFactorExperiment
from agents.kb_update import KbUpdater
from agents.trace import TraceManager, TraceRecord


# ---------------------------------------------------------------------------
# Trace tests
# ---------------------------------------------------------------------------


class TestTraceManager:
    def test_append_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tm = TraceManager(td)
            r1 = TraceRecord(
                round=1,
                factor_id="f_auto_001",
                category="volume_reversal",
                data_sources=["market_daily"],
                status="fail",
                failure_type="icir_fail",
                error_signature=None,
                diagnosis="ICIR too low",
                fix_strategy="increase window",
                fix_level="factor",
                factor_change="params",
                factor_params={"window": 20},
                strategy_params={},
                code_summary="rank(ts_std(turnover_rate, 20))",
                tried_params={"window": 20},
                recommend_abandon=False,
                metrics={"annual_icir": 0.5},
                same_direction=True,
                new_hypothesis=None,
            )
            tm.append(r1)

            records = tm.read_all()
            assert len(records) == 1
            assert records[0]["round"] == 1
            assert records[0]["factor_id"] == "f_auto_001"
            assert records[0]["branch_id"] == "main"
            assert records[0]["parent_round_id"] is None

    def test_round_increment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tm = TraceManager(td)
            for i in range(1, 4):
                tm.append(
                    TraceRecord(
                        round=i,
                        factor_id="f_auto_001",
                        category="test",
                        data_sources=[],
                        status="fail",
                        failure_type=None,
                        error_signature=None,
                        diagnosis="",
                        fix_strategy="",
                        fix_level="",
                        factor_change=None,
                        factor_params={},
                        strategy_params={},
                        code_summary="",
                        tried_params={},
                        recommend_abandon=False,
                        metrics={},
                        same_direction=True,
                        new_hypothesis=None,
                    )
                )
            assert tm.get_max_round() == 3
            assert tm.get_next_round() == 4
            assert tm.get_default_parent_round() == 3

    def test_branch_dag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tm = TraceManager(td)
            # Round 1 (root)
            tm.append(
                TraceRecord(
                    round=1,
                    factor_id="f_auto_001",
                    category="test",
                    data_sources=[],
                    status="fail",
                    failure_type="backtest_fail",
                    error_signature=None,
                    diagnosis="",
                    fix_strategy="",
                    fix_level="",
                    factor_change=None,
                    factor_params={},
                    strategy_params={},
                    code_summary="",
                    tried_params={},
                    recommend_abandon=False,
                    metrics={},
                    same_direction=True,
                    new_hypothesis=None,
                    parent_round_id=None,
                    branch_id="main",
                )
            )
            # Round 2 (branch from round 1)
            tm.append(
                TraceRecord(
                    round=2,
                    factor_id="f_auto_001_b",
                    category="test",
                    data_sources=[],
                    status="fail",
                    failure_type="backtest_fail",
                    error_signature=None,
                    diagnosis="",
                    fix_strategy="",
                    fix_level="",
                    factor_change=None,
                    factor_params={},
                    strategy_params={},
                    code_summary="",
                    tried_params={},
                    recommend_abandon=False,
                    metrics={},
                    same_direction=True,
                    new_hypothesis=None,
                    parent_round_id=1,
                    branch_id="explore_low_turnover",
                    fork_reason="try lower turnover variant",
                )
            )

            records = tm.read_all()
            assert records[1]["parent_round_id"] == 1
            assert records[1]["branch_id"] == "explore_low_turnover"
            assert records[1]["fork_reason"] == "try lower turnover variant"

    def test_from_result_json_metrics_mapping(self) -> None:
        result: dict[str, Any] = {
            "factor_id": "f_auto_001",
            "status": "fail",
            "failure_type": "ridge_fail",
            "error": None,
            "metrics": {
                "annual_icir": 2.5,
                "simple_sharpe": 0.9,
                "max_corr": 0.35,
                "residual_annual_icir": 1.2,
            },
            "experiment": {
                "factor_id": "f_auto_001",
                "factor_code": "",
                "step_results": {
                    "step8": {
                        "passed": False,
                        "metrics": {"r2": 0.85, "tier": "C"},
                    }
                },
                "category": "momentum_reversal",
                "data_sources": ["market_daily"],
            },
        }
        record = TraceRecord.from_result_json(result, round_num=3, category="momentum_reversal")
        assert record.round == 3
        assert record.factor_id == "f_auto_001"
        assert record.metrics["annual_icir"] == 2.5
        assert record.metrics["simple_sharpe"] == 0.9
        assert record.metrics["r2"] == 0.85
        assert record.metrics["max_existing_corr"] == 0.35
        assert record.metrics["residual_icir"] == 1.2
        assert record.category == "momentum_reversal"

    def test_from_result_json_rc_output(self) -> None:
        result: dict[str, Any] = {
            "factor_id": "f_auto_001",
            "status": "fail",
            "failure_type": "icir_fail",
            "error": None,
            "metrics": {},
            "experiment": {"factor_id": "f_auto_001", "factor_code": "", "step_results": {}},
        }
        rc_output = {
            "diagnosis": "ICIR decayed",
            "fix_strategy": "shorter window",
            "fix_level": "factor",
            "factor_change": "params",
            "factor_params": {"window": 10},
            "strategy_params": {"decay": 5},
            "recommend_abandon": False,
            "same_direction": True,
            "new_hypothesis": None,
        }
        record = TraceRecord.from_result_json(result, rc_output=rc_output)
        assert record.diagnosis == "ICIR decayed"
        assert record.factor_params == {"window": 10}
        assert record.strategy_params == {"decay": 5}
        assert record.tried_params == {"window": 10}  # inherited from factor_params

    def test_from_result_json_error_signature(self) -> None:
        long_error = "NameError: name 'foo' is not defined" + " and more text" * 20
        result: dict[str, Any] = {
            "factor_id": "f_auto_001",
            "status": "error",
            "failure_type": "code_error",
            "error": long_error,
            "metrics": {},
            "experiment": {"factor_id": "f_auto_001", "factor_code": "", "step_results": {}},
        }
        record = TraceRecord.from_result_json(result)
        assert record.error_signature is not None
        assert len(record.error_signature) <= 120
        assert "NameError" in record.error_signature


# ---------------------------------------------------------------------------
# QuantFeedback layer tests
# ---------------------------------------------------------------------------


class TestQuantFeedbackLayers:
    def test_layer_mapping(self) -> None:
        fb = QuantFeedback(
            decision=False,
            failed_step="step3",
            execution=ExecutionFeedback(coverage_ratio=0.95),
            evaluation=EvaluationFeedback(annual_icir=1.2, monotonicity=0.8),
        )

        # code_error → execution layer
        r = fb.get_relevant_layer("code_error")
        assert r["layer"] == "execution"
        assert "coverage_ratio" in r["execution"]

        # icir_fail → evaluation layer
        r = fb.get_relevant_layer("icir_fail")
        assert r["layer"] == "evaluation"
        assert "annual_icir" in r["evaluation"]

        # backtest_fail → evaluation layer
        r = fb.get_relevant_layer("backtest_fail")
        assert r["layer"] == "evaluation"

        # config_error → execution layer
        r = fb.get_relevant_layer("config_error")
        assert r["layer"] == "execution"

    def test_flat_backward_compat(self) -> None:
        fb = QuantFeedback(
            decision=True,
            passed_steps=["step1", "step2", "step3"],
            evaluation=EvaluationFeedback(
                annual_icir=2.5,
                pos_ratio=0.62,
                simple_sharpe=1.1,
            ),
        )
        flat = fb.to_flat_dict()
        assert flat["decision"] is True
        assert flat["annual_icir"] == 2.5
        assert flat["pos_ratio"] == 0.62
        assert flat["simple_sharpe"] == 1.1
        assert "metrics" in flat

    def test_layered_format(self) -> None:
        fb = QuantFeedback(
            decision=False,
            execution=ExecutionFeedback(error="SyntaxError"),
            evaluation=EvaluationFeedback(annual_icir=1.5),
            hypothesis=HypothesisFeedback(category="value"),
        )
        layered = fb.to_layered_dict()
        assert "execution" in layered
        assert "evaluation" in layered
        assert "hypothesis" in layered
        assert layered["execution"]["error"] == "SyntaxError"
        assert layered["evaluation"]["annual_icir"] == 1.5
        assert layered["hypothesis"]["category"] == "value"

    def test_metrics_property(self) -> None:
        fb = QuantFeedback(
            evaluation=EvaluationFeedback(annual_icir=2.0, ridge_r2=0.5),
            execution=ExecutionFeedback(coverage_ratio=0.9),
        )
        m = fb.metrics
        assert m["annual_icir"] == 2.0
        assert m["ridge_r2"] == 0.5
        # metrics property only includes evaluation-layer keys for backward compat
        assert "coverage_ratio" not in m

    def test_from_dict_flat_format(self) -> None:
        data = {
            "decision": False,
            "annual_icir": 1.8,
            "simple_sharpe": 0.7,
            "metrics": {"annual_icir": 1.8, "simple_sharpe": 0.7},
        }
        fb = QuantFeedback.from_dict(data)
        assert fb.decision is False
        assert fb.evaluation.annual_icir == 1.8
        assert fb.evaluation.simple_sharpe == 0.7

    def test_from_dict_layered_format(self) -> None:
        data = {
            "decision": True,
            "execution": {"code_valid": True},
            "evaluation": {"annual_icir": 3.0},
            "hypothesis": {"category": "momentum"},
        }
        fb = QuantFeedback.from_dict(data)
        assert fb.execution.code_valid is True
        assert fb.evaluation.annual_icir == 3.0
        assert fb.hypothesis.category == "momentum"


class TestAutoQuantFactorEvaluatorLayers:
    def test_evaluate_populates_execution(self) -> None:
        exp = AutoQuantFactorExperiment(
            factor_id="f_test",
            step_results={
                "step1": {"passed": False, "metrics": {"coverage_ratio": 0.3}, "reason": "too many NaN"},
            },
        )
        ev = AutoQuantFactorEvaluator()
        fb = ev.evaluate(exp)
        assert fb.decision is False
        assert fb.execution.coverage_ratio == 0.3
        assert fb.execution.failed_step is None  # step1 failure goes to evaluation layer mapping, but execution obj has coverage

    def test_evaluate_populates_evaluation(self) -> None:
        exp = AutoQuantFactorExperiment(
            factor_id="f_test",
            step_results={
                "step1": {"passed": True, "metrics": {}},
                "step2": {"passed": True, "metrics": {"max_existing_corr": 0.15}},
                "step3": {"passed": True, "metrics": {"annual_icir": 2.5, "pos_ratio": 0.6}},
                "step4": {"passed": True, "metrics": {"spearman": 0.85}},
                "step5": {"passed": True, "metrics": {}},
                "step6": {"passed": True, "metrics": {}},
                "step7": {"passed": True, "metrics": {"annual_turnover": 2.5}},
                "step8": {"passed": True, "metrics": {"tier": "A", "r2": 0.1}},
                "step9": {"passed": True, "metrics": {"annual_icirs": {"f_001": 1.8}}},
            },
            simple_bt_metrics={"sharpe": 1.2, "annual_return": 0.15},
            detailed_bt_metrics={"sharpe": 1.1, "annual_return": 0.13, "annual_turnover": 2.5},
        )
        ev = AutoQuantFactorEvaluator()
        fb = ev.evaluate(exp)
        assert fb.decision is True
        assert fb.evaluation.annual_icir == 2.5
        assert fb.evaluation.turnover == 2.5
        assert fb.evaluation.ridge_tier == "A"
        assert fb.evaluation.ridge_r2 == 0.1
        assert fb.evaluation.cost_drag == pytest.approx(0.02)

    def test_evaluate_failed_step_in_evaluation(self) -> None:
        exp = AutoQuantFactorExperiment(
            factor_id="f_test",
            step_results={
                "step1": {"passed": True},
                "step2": {"passed": True},
                "step3": {"passed": False, "reason": "ICIR < threshold", "metrics": {"annual_icir": 0.5}},
            },
        )
        ev = AutoQuantFactorEvaluator()
        fb = ev.evaluate(exp)
        assert fb.failed_step == "step3"
        assert fb.evaluation.failed_step == "step3"
        assert fb.evaluation.failure_reason == "ICIR < threshold"


# ---------------------------------------------------------------------------
# KbUpdater tests
# ---------------------------------------------------------------------------


class TestKbUpdater:
    def test_hypothesis_index_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            exp = AutoQuantFactorExperiment(
                factor_id="f_auto_001",
                category="volume_reversal",
                eval_result={"annual_icir": 2.0},
                simple_bt_metrics={"sharpe": 0.8},
                status="rejected",
            )
            # First insert
            updater._update_hypothesis_index(exp)
            records = updater._load_jsonl(Path(td) / "hypothesis_index.jsonl")
            assert len(records) == 1
            assert records[0]["factor_id"] == "f_auto_001"
            assert records[0]["best_icir"] == 2.0
            assert records[0]["status"] == "rejected"

            # Second update with higher ICIR
            exp2 = AutoQuantFactorExperiment(
                factor_id="f_auto_001",
                category="volume_reversal",
                eval_result={"annual_icir": 2.5},
                simple_bt_metrics={"sharpe": 1.0},
                status="candidate",
            )
            updater._update_hypothesis_index(exp2)
            records = updater._load_jsonl(Path(td) / "hypothesis_index.jsonl")
            assert len(records) == 1  # still 1 (upsert)
            assert records[0]["best_icir"] == 2.5  # max
            assert records[0]["best_sharpe"] == 1.0  # updated
            assert records[0]["status"] == "candidate"

    def test_hypothesis_index_new_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            exp = AutoQuantFactorExperiment(
                factor_id="f_auto_002",
                category="momentum",
                eval_result={"annual_icir": 1.5},
                status="rejected",
            )
            updater._update_hypothesis_index(exp)
            records = updater._load_jsonl(Path(td) / "hypothesis_index.jsonl")
            assert len(records) == 1
            assert records[0]["factor_id"] == "f_auto_002"
            assert records[0]["category"] == "momentum"

    def test_anti_pattern_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            rc_output = {
                "failure_type": "backtest_fail",
                "new_anti_pattern": {
                    "pattern": "high_turnover_trap",
                    "category": "price_pattern",
                    "signature": "turnover>100% and cost_drag>10%",
                    "fix": "abandon immediately",
                },
            }
            # First insert
            updater._update_anti_patterns(rc_output)
            data = updater._load_json(Path(td) / "anti_patterns.json")
            assert "backtest_fail" in data
            assert len(data["backtest_fail"]) == 1
            assert data["backtest_fail"][0]["count"] == 1

            # Second insert with same signature
            updater._update_anti_patterns(rc_output)
            data = updater._load_json(Path(td) / "anti_patterns.json")
            assert len(data["backtest_fail"]) == 1
            assert data["backtest_fail"][0]["count"] == 2
            assert "last_seen" in data["backtest_fail"][0]

    def test_anti_pattern_null_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            rc_output = {"failure_type": "icir_fail", "new_anti_pattern": None}
            result = updater._update_anti_patterns(rc_output)
            assert result is False
            assert not (Path(td) / "anti_patterns.json").exists()

    def test_successful_patterns_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            exp = AutoQuantFactorExperiment(
                factor_id="f_auto_003",
                category="value",
                eval_result={"annual_icir": 2.0},
                simple_bt_metrics={"sharpe": 1.0},
                status="candidate",
            )
            updater._update_successful_patterns(exp)
            data = updater._load_json(Path(td) / "successful_patterns.json")
            assert "value" in data
            assert len(data["value"]) == 1
            assert data["value"][0]["factor_id"] == "f_auto_003"

            # Update same factor_id
            exp2 = AutoQuantFactorExperiment(
                factor_id="f_auto_003",
                category="value",
                eval_result={"annual_icir": 2.5},
                simple_bt_metrics={"sharpe": 1.2},
                status="candidate",
            )
            updater._update_successful_patterns(exp2)
            data = updater._load_json(Path(td) / "successful_patterns.json")
            assert len(data["value"]) == 1
            assert data["value"][0]["key_metrics"]["annual_icir"] == 2.5

    def test_failed_attempts_append(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            exp = AutoQuantFactorExperiment(
                factor_id="f_auto_004",
                category="momentum",
                eval_result={"annual_icir": 1.0},
                simple_bt_metrics={"sharpe": 0.3},
                status="rejected",
            )
            trace_record = {
                "code_summary": "ts_mean(pct_chg, 20)",
                "diagnosis": "ICIR too low",
                "failure_type": "icir_fail",
            }
            updater._append_failed_attempts(exp, trace_record, status="fail")
            records = updater._load_jsonl(Path(td) / "failed_attempts.jsonl")
            assert len(records) == 1
            assert records[0]["factor_id"] == "f_auto_004"
            assert records[0]["status"] == "fail"
            assert records[0]["code_summary"] == "ts_mean(pct_chg, 20)"
            assert records[0]["why_failed"] == "ICIR too low"
            assert records[0]["failure_type"] == "icir_fail"

    def test_update_on_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            exp = AutoQuantFactorExperiment(
                factor_id="f_auto_005",
                category="quality",
                eval_result={"annual_icir": 3.0},
                simple_bt_metrics={"sharpe": 1.5},
                status="candidate",
            )
            summary = updater.update_on_pass(exp)
            assert summary.hypothesis_index_updated is True
            assert summary.successful_patterns_updated is True
            # update_on_pass no longer writes to failed_attempts (only update_on_fail does)
            assert summary.failed_attempts_appended is False

            # Verify hypothesis_index
            records = updater._load_jsonl(Path(td) / "hypothesis_index.jsonl")
            assert len(records) == 1
            assert records[0]["status"] == "candidate"

    def test_update_on_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            updater = KbUpdater(td)
            exp = AutoQuantFactorExperiment(
                factor_id="f_auto_006",
                category="technical",
                eval_result={"annual_icir": 0.8},
                simple_bt_metrics={"sharpe": 0.2},
                status="rejected",
            )
            rc_output = {
                "failure_type": "backtest_fail",
                "new_anti_pattern": {
                    "pattern": "drawdown_trap",
                    "category": "technical",
                    "signature": "MDD>40% invariant",
                    "fix": "abandon",
                },
            }
            summary = updater.update_on_fail(exp, rc_output=rc_output)
            assert summary.anti_pattern_updated is True
            assert summary.hypothesis_index_updated is True
            assert summary.failed_attempts_appended is True

    def test_extract_formula_fingerprint(self) -> None:
        updater = KbUpdater()
        exp = AutoQuantFactorExperiment(
            factor_id="f_test",
            factor_code='\n"""Volume reversal factor"""\ndef factor(panel):\n    return -rank(ts_std(turnover_rate, 20))\n',
        )
        fp = updater._extract_formula_fingerprint(exp)
        assert "Volume reversal" in fp

    def test_extract_formula_fingerprint_fallback(self) -> None:
        updater = KbUpdater()
        exp = AutoQuantFactorExperiment(factor_id="f_test", factor_code="")
        fp = updater._extract_formula_fingerprint(exp)
        assert fp == ""

    def test_infer_data_sources(self) -> None:
        updater = KbUpdater()
        exp = AutoQuantFactorExperiment(
            factor_id="f_test",
            factor_code="def factor(panel): return panel['close']",
        )
        assert updater._infer_data_sources(exp) == ["market_daily"]

        exp2 = AutoQuantFactorExperiment(
            factor_id="f_test",
            factor_code="def factor(panel): return panel['inc_revenue'] / panel['bs_equity']",
        )
        ds = updater._infer_data_sources(exp2)
        assert "income_q" in ds
        assert "balancesheet_q" in ds
