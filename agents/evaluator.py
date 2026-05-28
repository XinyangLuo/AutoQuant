"""Factor evaluator — converts pipeline step results into structured feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .experiment import AutoQuantFactorExperiment


@dataclass
class QuantFeedback:
    """Structured feedback for a factor experiment run.

    Builds the decision from per-step results rather than re-computing
    thresholds — the canonical thresholds live in ``PipelineConfig`` /
    ``StepThresholds`` and are enforced by the pipeline step functions.
    """

    decision: bool = False
    observation: str = ""
    suggestion: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    # Step-level results summary
    passed_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None
    failure_reason: str | None = None

    # Factor evaluation metrics (from step3)
    annual_icir: float = float("-inf")
    pos_ratio: float = 0.0
    turnover: float = float("inf")
    max_corr: float = 0.0

    # Simple backtest metrics (from step6)
    simple_sharpe: float | None = None
    simple_mdd: float | None = None
    simple_annual_return: float | None = None
    simple_calmar: float | None = None

    # Detailed backtest metrics (from step7)
    detailed_sharpe: float | None = None
    detailed_annual_return: float | None = None
    cost_drag: float | None = None

    # Pipeline gate metrics
    monotonicity: float | None = None
    ridge_tier: str | None = None
    residual_annual_icir: float | None = None

    def to_dict(self) -> dict[str, Any]:
        import math

        base = {
            "decision": self.decision,
            "observation": self.observation,
            "suggestion": self.suggestion,
            "metrics": self.metrics,
            "passed_steps": self.passed_steps,
            "failed_step": self.failed_step,
            "failure_reason": self.failure_reason,
        }
        extras = {
            "annual_icir": self.annual_icir,
            "pos_ratio": self.pos_ratio,
            "turnover": self.turnover,
            "max_corr": self.max_corr,
            "simple_sharpe": self.simple_sharpe,
            "simple_mdd": self.simple_mdd,
            "simple_calmar": self.simple_calmar,
            "detailed_sharpe": self.detailed_sharpe,
            "detailed_annual_return": self.detailed_annual_return,
            "cost_drag": self.cost_drag,
            "monotonicity": self.monotonicity,
            "ridge_tier": self.ridge_tier,
            "residual_annual_icir": self.residual_annual_icir,
        }
        for k, v in extras.items():
            if v is None:
                continue
            if isinstance(v, float):
                if v == float("-inf") or v == float("inf") or math.isnan(v):
                    continue
            base[k] = v
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantFeedback":
        metrics = data.get("metrics", {})
        return cls(
            decision=data.get("decision", False),
            observation=data.get("observation", ""),
            suggestion=data.get("suggestion", ""),
            metrics=metrics,
            passed_steps=data.get("passed_steps", []),
            failed_step=data.get("failed_step"),
            failure_reason=data.get("failure_reason"),
            annual_icir=data.get("annual_icir", metrics.get("annual_icir", float("-inf"))),
            pos_ratio=data.get("pos_ratio", metrics.get("pos_ratio", 0.0)),
            turnover=data.get("turnover", metrics.get("turnover", float("inf"))),
            max_corr=data.get("max_corr", metrics.get("max_corr", 0.0)),
            simple_sharpe=data.get("simple_sharpe", metrics.get("simple_sharpe")),
            simple_mdd=data.get("simple_mdd", metrics.get("simple_mdd")),
            simple_annual_return=data.get("simple_annual_return", metrics.get("simple_annual_return")),
            simple_calmar=data.get("simple_calmar", metrics.get("simple_calmar")),
            detailed_sharpe=data.get("detailed_sharpe", metrics.get("detailed_sharpe")),
            detailed_annual_return=data.get("detailed_annual_return", metrics.get("detailed_annual_return")),
            cost_drag=data.get("cost_drag", metrics.get("cost_drag")),
            monotonicity=data.get("monotonicity", metrics.get("monotonicity")),
            ridge_tier=data.get("ridge_tier", metrics.get("ridge_tier")),
            residual_annual_icir=data.get("residual_annual_icir"),
        )


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

        # Extract metrics from step results
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
        residual_annual_icir = (
            max(v for v in annual_icirs.values() if v is not None and not math.isnan(v))
            if annual_icirs else None
        )

        cost_drag = None
        if simple_ann_ret is not None and detailed_ann_ret is not None:
            cost_drag = simple_ann_ret - detailed_ann_ret

        observation = self._format_observation(
            passed_steps, failed_step, failure_reason,
            annual_icir, ic_pos, turnover, max_corr, monotonicity,
            simple_sharpe, simple_mdd, simple_ann_ret,
            detailed_sharpe, detailed_ann_ret,
            ridge_tier, residual_annual_icir,
        )
        suggestion = self._generate_suggestion(
            decision, failed_step,
            annual_icir, ic_pos, turnover, max_corr, simple_sharpe,
        )

        return QuantFeedback(
            decision=decision,
            observation=observation,
            suggestion=suggestion,
            passed_steps=passed_steps,
            failed_step=failed_step,
            failure_reason=failure_reason,
            metrics={
                "annual_icir": annual_icir,
                "pos_ratio": ic_pos,
                "turnover": turnover,
                "max_corr": max_corr,
                "simple_sharpe": simple_sharpe,
                "simple_mdd": simple_mdd,
                "simple_annual_return": simple_ann_ret,
                "detailed_sharpe": detailed_sharpe,
                "detailed_annual_return": detailed_ann_ret,
                "cost_drag": cost_drag,
                "monotonicity": monotonicity,
                "ridge_tier": ridge_tier,
                "residual_annual_icir": residual_annual_icir,
            },
            annual_icir=annual_icir,
            pos_ratio=ic_pos,
            turnover=turnover,
            max_corr=max_corr,
            simple_sharpe=simple_sharpe,
            simple_mdd=simple_mdd,
            simple_annual_return=simple_ann_ret,
            simple_calmar=simple_calmar,
            detailed_sharpe=detailed_sharpe,
            detailed_annual_return=detailed_ann_ret,
            cost_drag=cost_drag,
            monotonicity=monotonicity,
            ridge_tier=ridge_tier,
            residual_annual_icir=residual_annual_icir,
        )

    def _format_observation(
        self,
        passed_steps: list[str],
        failed_step: str | None,
        failure_reason: str | None,
        annual_icir: float, ic_pos: float, turnover: float, max_corr: float,
        monotonicity: float | None,
        simple_sharpe: float | None, simple_mdd: float | None,
        simple_ann_ret: float | None,
        detailed_sharpe: float | None, detailed_ann_ret: float | None,
        ridge_tier: str | None,
        residual_annual_icir: float | None,
    ) -> str:
        parts: list[str] = []

        if failed_step:
            parts.append(f"Failed at {failed_step}: {failure_reason or 'unknown'}")
        parts.append(f"Passed steps: {', '.join(passed_steps) if passed_steps else 'none'}")

        if annual_icir != float("-inf"):
            parts.append(f"Annual ICIR = {annual_icir:.3f}")
        parts.append(f"IC+ ratio = {ic_pos:.1%}")
        if monotonicity is not None:
            parts.append(f"Monotonicity = {monotonicity:.3f}")
        if simple_sharpe is not None:
            parts.append(f"Simple Sharpe = {simple_sharpe:.3f}")
        if simple_mdd is not None:
            parts.append(f"Simple MDD = {simple_mdd:.3%}")
        if simple_ann_ret is not None:
            parts.append(f"Simple Annual Return = {simple_ann_ret:.2%}")
        if detailed_sharpe is not None:
            parts.append(f"Detailed Sharpe = {detailed_sharpe:.3f}")
        if detailed_ann_ret is not None:
            parts.append(f"Detailed Annual Return = {detailed_ann_ret:.2%}")
        if ridge_tier:
            parts.append(f"Ridge tier = {ridge_tier}")
        if residual_annual_icir is not None:
            parts.append(f"Residual annual ICIR = {residual_annual_icir:.3f}")
        return "\n".join(parts)

    def _generate_suggestion(
        self,
        decision: bool,
        failed_step: str | None,
        annual_icir: float, ic_pos: float, turnover: float,
        max_corr: float, simple_sharpe: float | None,
    ) -> str:
        if decision:
            return (
                "All pipeline steps passed. The factor is a candidate for admission. "
                "Review the pipeline report in results/agent/candidates/ before admitting."
            )

        if failed_step is None:
            return "Pipeline did not complete. Check the error log for details."

        step_suggestions = {
            "step1": "Coverage check failed. The factor has too many missing values. "
                      "Check data source availability or widen the universe.",
            "step2": "Neutralization failed — factor is too correlated with size or industry. "
                     "The Barra neutralization pipeline may need review.",
            "step3": f"ICIR check failed (Annual ICIR={annual_icir:.3f}). "
                     "Try a longer lookback window, add a volume filter, or change the construction.",
            "step4": "Monotonicity check failed — decile returns are not monotonic. "
                     "The factor may only work at extremes. Consider adding a secondary filter.",
            "step5": "Strategy config error. Check top_k/top_pct/decay settings in config.yaml.",
            "step6": f"Simple backtest failed (Sharpe={simple_sharpe or 'N/A'}). "
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
