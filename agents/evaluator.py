"""Factor evaluator — converts pipeline step results into structured feedback.

QuantFeedback is split into three layers:
  * ExecutionFeedback — code errors, schema errors, coverage, config
  * EvaluationFeedback — ICIR, monotonicity, backtest metrics, ridge, residual
  * HypothesisFeedback — category, data sources, construct validity

The layered design lets the Result Critic subagent receive only the
relevant layer for a given failure_type, reducing token usage and
improving focus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .experiment import AutoQuantFactorExperiment


# ---------------------------------------------------------------------------
# Layered feedback dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExecutionFeedback:
    """Execution-layer feedback: code / schema / config / coverage failures."""

    error: str | None = None
    traceback: str | None = None
    code_valid: bool = False
    imports_valid: bool = False
    coverage_ratio: float | None = None
    failed_step: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.error is not None:
            d["error"] = self.error
        if self.traceback is not None:
            d["traceback"] = self.traceback
        if self.code_valid:
            d["code_valid"] = self.code_valid
        if self.imports_valid:
            d["imports_valid"] = self.imports_valid
        if self.coverage_ratio is not None:
            d["coverage_ratio"] = self.coverage_ratio
        if self.failed_step is not None:
            d["failed_step"] = self.failed_step
        if self.failure_reason is not None:
            d["failure_reason"] = self.failure_reason
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionFeedback":
        return cls(
            error=data.get("error"),
            traceback=data.get("traceback"),
            code_valid=data.get("code_valid", False),
            imports_valid=data.get("imports_valid", False),
            coverage_ratio=data.get("coverage_ratio"),
            failed_step=data.get("failed_step"),
            failure_reason=data.get("failure_reason"),
        )


@dataclass
class EvaluationFeedback:
    """Evaluation-layer feedback: predictive power, backtest, style tests."""

    annual_icir: float = float("-inf")
    pos_ratio: float = 0.0
    turnover: float = float("inf")
    max_corr: float = 0.0
    max_existing_factor: str | None = None
    monotonicity: float | None = None
    simple_sharpe: float | None = None
    simple_mdd: float | None = None
    simple_annual_return: float | None = None
    simple_calmar: float | None = None
    detailed_sharpe: float | None = None
    detailed_annual_return: float | None = None
    cost_drag: float | None = None
    ridge_tier: str | None = None
    ridge_r2: float | None = None
    residual_annual_icir: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.annual_icir != float("-inf"):
            d["annual_icir"] = self.annual_icir
        if self.pos_ratio != 0.0:
            d["pos_ratio"] = self.pos_ratio
        if self.turnover != float("inf"):
            d["turnover"] = self.turnover
        if self.max_corr != 0.0:
            d["max_corr"] = self.max_corr
        if self.max_existing_factor is not None:
            d["max_existing_factor"] = self.max_existing_factor
        if self.monotonicity is not None:
            d["monotonicity"] = self.monotonicity
        if self.simple_sharpe is not None:
            d["simple_sharpe"] = self.simple_sharpe
        if self.simple_mdd is not None:
            d["simple_mdd"] = self.simple_mdd
        if self.simple_annual_return is not None:
            d["simple_annual_return"] = self.simple_annual_return
        if self.simple_calmar is not None:
            d["simple_calmar"] = self.simple_calmar
        if self.detailed_sharpe is not None:
            d["detailed_sharpe"] = self.detailed_sharpe
        if self.detailed_annual_return is not None:
            d["detailed_annual_return"] = self.detailed_annual_return
        if self.cost_drag is not None:
            d["cost_drag"] = self.cost_drag
        if self.ridge_tier is not None:
            d["ridge_tier"] = self.ridge_tier
        if self.ridge_r2 is not None:
            d["ridge_r2"] = self.ridge_r2
        if self.residual_annual_icir is not None:
            d["residual_annual_icir"] = self.residual_annual_icir
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationFeedback":
        return cls(
            annual_icir=data.get("annual_icir", float("-inf")),
            pos_ratio=data.get("pos_ratio", 0.0),
            turnover=data.get("turnover", float("inf")),
            max_corr=data.get("max_corr", 0.0),
            max_existing_factor=data.get("max_existing_factor"),
            monotonicity=data.get("monotonicity"),
            simple_sharpe=data.get("simple_sharpe"),
            simple_mdd=data.get("simple_mdd"),
            simple_annual_return=data.get("simple_annual_return"),
            simple_calmar=data.get("simple_calmar"),
            detailed_sharpe=data.get("detailed_sharpe"),
            detailed_annual_return=data.get("detailed_annual_return"),
            cost_drag=data.get("cost_drag"),
            ridge_tier=data.get("ridge_tier"),
            ridge_r2=data.get("ridge_r2"),
            residual_annual_icir=data.get("residual_annual_icir"),
        )


@dataclass
class HypothesisFeedback:
    """Hypothesis-layer feedback: direction, construct validity, data consistency."""

    category: str = ""
    data_sources: list[str] = field(default_factory=list)
    construct_valid: bool = True
    direction_consistent: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.category:
            d["category"] = self.category
        if self.data_sources:
            d["data_sources"] = self.data_sources
        if not self.construct_valid:
            d["construct_valid"] = self.construct_valid
        if not self.direction_consistent:
            d["direction_consistent"] = self.direction_consistent
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisFeedback":
        return cls(
            category=data.get("category", ""),
            data_sources=data.get("data_sources", []),
            construct_valid=data.get("construct_valid", True),
            direction_consistent=data.get("direction_consistent", True),
        )


# ---------------------------------------------------------------------------
# QuantFeedback (container)
# ---------------------------------------------------------------------------

@dataclass
class QuantFeedback:
    """Structured feedback for a factor experiment run.

    Builds the decision from per-step results rather than re-computing
    thresholds — the canonical thresholds live in ``PipelineConfig`` /
    ``StepThresholds`` and are enforced by the pipeline step functions.

    The feedback is split into three layers so that the Result Critic
    subagent can receive only the relevant layer for a given failure_type.
    """

    # Top-level decision fields
    decision: bool = False
    observation: str = ""
    suggestion: str = ""
    passed_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None
    failure_reason: str | None = None

    # Three layers
    execution: ExecutionFeedback = field(default_factory=ExecutionFeedback)
    evaluation: EvaluationFeedback = field(default_factory=EvaluationFeedback)
    hypothesis: HypothesisFeedback = field(default_factory=HypothesisFeedback)

    # Backward-compat: flat metrics property
    @property
    def metrics(self) -> dict[str, Any]:
        """Flatten evaluation metrics into a single dict (backward compatible).

        Only evaluation-layer numeric metrics are included, matching the
        legacy ``metrics`` field schema.
        """
        return self.evaluation.to_dict()

    # Layer mapping for selective injection
    _LAYER_MAP: dict[str, str] = field(
        default_factory=lambda: {
            "code_error": "execution",
            "schema_error": "execution",
            "execution_error": "execution",
            "coverage_fail": "execution",
            "config_error": "execution",
            "neutralization_fail": "evaluation",
            "icir_fail": "evaluation",
            "monotonicity_fail": "evaluation",
            "backtest_fail": "evaluation",
            "ridge_fail": "evaluation",
            "residual_fail": "evaluation",
            "metrics_fail": "evaluation",
        }
    )

    def get_relevant_layer(self, failure_type: str | None) -> dict[str, Any]:
        """Return the layer relevant to *failure_type* plus top-level decision fields.

        This is the preferred format for injecting into the Result Critic
        prompt — only the relevant layer is sent, reducing token usage.
        """
        # Default to "hypothesis" for passes (failure_type is None) so that
        # downstream consumers get metadata rather than backtest metrics.
        layer_name = self._LAYER_MAP.get(failure_type or "", "hypothesis")
        layer_obj = getattr(self, layer_name)
        result: dict[str, Any] = {
            "decision": self.decision,
            "observation": self.observation,
            "suggestion": self.suggestion,
            "passed_steps": self.passed_steps,
            "failed_step": self.failed_step,
            "failure_reason": self.failure_reason,
            "layer": layer_name,
        }
        result[layer_name] = layer_obj.to_dict()
        return result

    def to_flat_dict(self) -> dict[str, Any]:
        """Backward-compatible flat dict — same schema as the legacy format."""
        base: dict[str, Any] = {
            "decision": self.decision,
            "observation": self.observation,
            "suggestion": self.suggestion,
            "metrics": self.metrics,
            "passed_steps": self.passed_steps,
            "failed_step": self.failed_step,
            "failure_reason": self.failure_reason,
        }
        # Flatten evaluation metrics for direct access
        ev = self.evaluation
        extras: dict[str, Any] = {
            "annual_icir": ev.annual_icir,
            "pos_ratio": ev.pos_ratio,
            "turnover": ev.turnover,
            "max_corr": ev.max_corr,
            "simple_sharpe": ev.simple_sharpe,
            "simple_mdd": ev.simple_mdd,
            "simple_annual_return": ev.simple_annual_return,
            "simple_calmar": ev.simple_calmar,
            "detailed_sharpe": ev.detailed_sharpe,
            "detailed_annual_return": ev.detailed_annual_return,
            "cost_drag": ev.cost_drag,
            "monotonicity": ev.monotonicity,
            "ridge_tier": ev.ridge_tier,
            "residual_annual_icir": ev.residual_annual_icir,
        }
        for k, v in extras.items():
            if v is None:
                continue
            if isinstance(v, float):
                if v == float("-inf") or v == float("inf") or math.isnan(v):
                    continue
            base[k] = v
        return base

    def to_layered_dict(self) -> dict[str, Any]:
        """New layered format for consumers that understand the three-layer model."""
        return {
            "decision": self.decision,
            "observation": self.observation,
            "suggestion": self.suggestion,
            "passed_steps": self.passed_steps,
            "failed_step": self.failed_step,
            "failure_reason": self.failure_reason,
            "execution": self.execution.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "hypothesis": self.hypothesis.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantFeedback":
        """Restore from a dict (accepts both flat and layered formats)."""
        # Try layered first
        if "execution" in data or "evaluation" in data:
            return cls(
                decision=data.get("decision", False),
                observation=data.get("observation", ""),
                suggestion=data.get("suggestion", ""),
                passed_steps=data.get("passed_steps", []),
                failed_step=data.get("failed_step"),
                failure_reason=data.get("failure_reason"),
                execution=ExecutionFeedback.from_dict(data.get("execution", {})),
                evaluation=EvaluationFeedback.from_dict(data.get("evaluation", {})),
                hypothesis=HypothesisFeedback.from_dict(data.get("hypothesis", {})),
            )

        # Flat format fallback (legacy)
        metrics = data.get("metrics", {})
        return cls(
            decision=data.get("decision", False),
            observation=data.get("observation", ""),
            suggestion=data.get("suggestion", ""),
            passed_steps=data.get("passed_steps", []),
            failed_step=data.get("failed_step"),
            failure_reason=data.get("failure_reason"),
            evaluation=EvaluationFeedback(
                annual_icir=data.get("annual_icir", metrics.get("annual_icir", float("-inf"))),
                pos_ratio=data.get("pos_ratio", metrics.get("pos_ratio", 0.0)),
                turnover=data.get("turnover", metrics.get("turnover", float("inf"))),
                max_corr=data.get("max_corr", metrics.get("max_corr", 0.0)),
                max_existing_factor=metrics.get("max_existing_factor"),
                simple_sharpe=data.get("simple_sharpe", metrics.get("simple_sharpe")),
                simple_mdd=data.get("simple_mdd", metrics.get("simple_mdd")),
                simple_annual_return=data.get(
                    "simple_annual_return", metrics.get("simple_annual_return")
                ),
                simple_calmar=data.get("simple_calmar", metrics.get("simple_calmar")),
                detailed_sharpe=data.get("detailed_sharpe", metrics.get("detailed_sharpe")),
                detailed_annual_return=data.get(
                    "detailed_annual_return", metrics.get("detailed_annual_return")
                ),
                cost_drag=data.get("cost_drag", metrics.get("cost_drag")),
                monotonicity=data.get("monotonicity", metrics.get("monotonicity")),
                ridge_tier=data.get("ridge_tier", metrics.get("ridge_tier")),
                ridge_r2=metrics.get("ridge_r2"),
                residual_annual_icir=data.get(
                    "residual_annual_icir", metrics.get("residual_annual_icir")
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class AutoQuantFactorEvaluator:
    """Evaluate a factor experiment from pipeline step results.

    The decision is derived from the per-step pass/fail status recorded
    by the canonical pipeline — no thresholds are re-computed here.
    """

    def evaluate(self, experiment: AutoQuantFactorExperiment) -> QuantFeedback:
        """Evaluate a completed experiment and return structured feedback."""
        step_results = experiment.step_results or {}

        passed_steps = [name for name, sr in step_results.items() if sr.get("passed", False)]
        failed_steps = [name for name, sr in step_results.items() if not sr.get("passed", True)]

        failed_step = failed_steps[0] if failed_steps else None
        failure_reason = None
        if failed_step:
            failure_reason = step_results[failed_step].get("reason")

        # Gate steps are step1-step9; step10 is a report step that always passes
        gate_step_names = {f"step{i}" for i in range(1, 10)}
        gate_passed = [s for s in passed_steps if s in gate_step_names]
        decision = len(failed_steps) == 0 and len(gate_passed) == len(gate_step_names)

        # ------------------------------------------------------------------
        # Build ExecutionFeedback
        # ------------------------------------------------------------------
        execution = ExecutionFeedback()
        if experiment.error:
            execution.error = experiment.error
        sr1 = step_results.get("step1", {})
        if sr1.get("metrics"):
            execution.coverage_ratio = sr1["metrics"].get("coverage_ratio")
        sr5 = step_results.get("step5", {})
        if not sr5.get("passed", True):
            execution.failed_step = "step5"
            execution.failure_reason = sr5.get("reason")

        # ------------------------------------------------------------------
        # Build EvaluationFeedback
        # ------------------------------------------------------------------
        sr2 = step_results.get("step2", {}).get("metrics", {})
        sr3 = step_results.get("step3", {}).get("metrics", {})
        sr4 = step_results.get("step4", {}).get("metrics", {})
        sr6 = experiment.simple_bt_metrics or {}
        sr7 = experiment.detailed_bt_metrics or {}
        sr8 = step_results.get("step8", {}).get("metrics", {})
        sr9 = step_results.get("step9", {}).get("metrics", {})

        annual_icir = sr3.get("annual_icir", float("-inf"))
        ic_pos = sr3.get("pos_ratio", 0.0)
        turnover = sr7.get("annual_turnover", float("inf"))
        max_corr = sr2.get("max_existing_corr", 0.0)
        monotonicity = sr4.get("spearman")
        simple_sharpe = sr6.get("sharpe")
        simple_mdd = sr6.get("max_drawdown")
        simple_ann_ret = sr6.get("annual_return")
        simple_calmar = sr6.get("calmar")
        detailed_sharpe = sr7.get("sharpe")
        detailed_ann_ret = sr7.get("annual_return")
        ridge_tier = sr8.get("tier")
        annual_icirs = sr9.get("annual_icirs", {})
        residual_annual_icir = None
        if annual_icirs:
            valid_icirs = [
                v for v in annual_icirs.values()
                if v is not None and not math.isnan(v)
            ]
            if valid_icirs:
                residual_annual_icir = max(valid_icirs)

        cost_drag = None
        if simple_ann_ret is not None and detailed_ann_ret is not None:
            cost_drag = simple_ann_ret - detailed_ann_ret

        evaluation = EvaluationFeedback(
            annual_icir=annual_icir,
            pos_ratio=ic_pos,
            turnover=turnover,
            max_corr=max_corr,
            max_existing_factor=sr2.get("max_existing_factor"),
            monotonicity=monotonicity,
            simple_sharpe=simple_sharpe,
            simple_mdd=simple_mdd,
            simple_annual_return=simple_ann_ret,
            simple_calmar=simple_calmar,
            detailed_sharpe=detailed_sharpe,
            detailed_annual_return=detailed_ann_ret,
            cost_drag=cost_drag,
            ridge_tier=ridge_tier,
            ridge_r2=sr8.get("r2"),
            residual_annual_icir=residual_annual_icir,
        )

        # Set evaluation-level failed_step if the failure is in evaluation layer
        if failed_step and failed_step in {"step2", "step3", "step4", "step6", "step7", "step8", "step9"}:
            evaluation.failed_step = failed_step
            evaluation.failure_reason = failure_reason

        # ------------------------------------------------------------------
        # Build HypothesisFeedback
        # ------------------------------------------------------------------
        hypothesis = HypothesisFeedback(
            category=experiment.category or "",
            data_sources=experiment.keywords or [],
        )

        # ------------------------------------------------------------------
        # Assemble QuantFeedback
        # ------------------------------------------------------------------
        observation = self._format_observation(
            passed_steps,
            failed_step,
            failure_reason,
            evaluation,
        )
        suggestion = self._generate_suggestion(
            decision,
            failed_step,
            evaluation,
        )

        return QuantFeedback(
            decision=decision,
            observation=observation,
            suggestion=suggestion,
            passed_steps=passed_steps,
            failed_step=failed_step,
            failure_reason=failure_reason,
            execution=execution,
            evaluation=evaluation,
            hypothesis=hypothesis,
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_observation(
        self,
        passed_steps: list[str],
        failed_step: str | None,
        failure_reason: str | None,
        ev: EvaluationFeedback,
    ) -> str:
        parts: list[str] = []

        if failed_step:
            parts.append(f"Failed at {failed_step}: {failure_reason or 'unknown'}")
        parts.append(f"Passed steps: {', '.join(passed_steps) if passed_steps else 'none'}")

        if ev.annual_icir != float("-inf"):
            parts.append(f"Annual ICIR = {ev.annual_icir:.3f}")
        parts.append(f"IC+ ratio = {ev.pos_ratio:.1%}")
        if ev.monotonicity is not None:
            parts.append(f"Monotonicity = {ev.monotonicity:.3f}")
        if ev.simple_sharpe is not None:
            parts.append(f"Simple Sharpe = {ev.simple_sharpe:.3f}")
        if ev.simple_mdd is not None:
            parts.append(f"Simple MDD = {ev.simple_mdd:.3%}")
        if ev.simple_annual_return is not None:
            parts.append(f"Simple Annual Return = {ev.simple_annual_return:.2%}")
        if ev.detailed_sharpe is not None:
            parts.append(f"Detailed Sharpe = {ev.detailed_sharpe:.3f}")
        if ev.detailed_annual_return is not None:
            parts.append(f"Detailed Annual Return = {ev.detailed_annual_return:.2%}")
        if ev.ridge_tier:
            parts.append(f"Ridge tier = {ev.ridge_tier}")
        if ev.residual_annual_icir is not None:
            parts.append(f"Residual annual ICIR = {ev.residual_annual_icir:.3f}")
        return "\n".join(parts)

    def _generate_suggestion(
        self,
        decision: bool,
        failed_step: str | None,
        ev: EvaluationFeedback,
    ) -> str:
        if decision:
            return (
                "All pipeline steps passed. The factor is a candidate for admission. "
                "Review the pipeline report in results/candidates/ before admitting."
            )

        if failed_step is None:
            return "Pipeline did not complete. Check the error log for details."

        step_suggestions = {
            "step1": "Coverage check failed. The factor has too many missing values. "
            "Check data source availability or widen the universe.",
            "step2": "Neutralization failed — factor is too correlated with size or industry. "
            "The Barra neutralization pipeline may need review.",
            "step3": f"ICIR check failed (Annual ICIR={ev.annual_icir:.3f}). "
            "Try a longer lookback window, add a volume filter, or change the construction.",
            "step4": "Monotonicity check failed — decile returns are not monotonic. "
            "The factor may only work at extremes. Consider adding a secondary filter.",
            "step5": "Strategy config error. Check top_k/top_pct/decay settings in config.yaml.",
            "step6": f"Simple backtest failed (Sharpe={ev.simple_sharpe or 'N/A'}). "
            "Try adjusting decay, universe, or top_k via config.yaml pipeline defaults.",
            "step7": "Detailed backtest failed. Costs or market frictions may be eating the alpha. "
            "Review turnover and fee impact.",
            "step8": "Ridge R² check failed — factor is a style clone of existing Barra factors. "
            "The factor does not add independent information.",
            "step9": "Residual ICIR check failed — factor has no incremental predictive power "
            "beyond already-admitted factors.",
            "step10": "Report generation failed. Check results directory permissions.",
        }
        return step_suggestions.get(failed_step, f"Pipeline stopped at {failed_step}.")
