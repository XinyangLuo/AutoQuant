"""AutoQuant factor evaluator — converts backtest metrics into structured feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backtest.factor.admission import RECOMMENDED_THRESHOLDS as _ADM_THRESHOLDS
from backtest.pipeline.config import StepThresholds as _PipeThresh

from .core.evaluation import Evaluator, Feedback
from .experiment import AutoQuantFactorExperiment


@dataclass
class QuantFeedback(Feedback):
    """Structured feedback for a quantitative factor experiment.

    Extends the base :class:`Feedback` with quantitative metrics that the
    hypothesis generator uses to steer the next iteration.
    """

    # Factor evaluation metrics
    rankicir: float = float("-inf")
    ic_positive_ratio: float = 0.0
    turnover: float = float("inf")
    max_corr: float = 0.0

    # Simple backtest metrics
    simple_sharpe: float | None = None
    simple_mdd: float | None = None
    simple_annual_return: float | None = None
    simple_calmar: float | None = None

    # Detailed backtest metrics (optional)
    detailed_sharpe: float | None = None
    detailed_annual_return: float | None = None
    cost_drag: float | None = None  # simple vs detailed annual_return diff

    # Pipeline gate metrics
    monotonicity: float | None = None
    ridge_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        # Add scalar fields that are NOT already in metrics to avoid duplication
        metrics_keys = set(base.get("metrics", {}).keys())
        extras = {
            "simple_mdd": self.simple_mdd,
            "simple_calmar": self.simple_calmar,
            "detailed_sharpe": self.detailed_sharpe,
            "detailed_annual_return": self.detailed_annual_return,
            "cost_drag": self.cost_drag,
            "monotonicity": self.monotonicity,
            "ridge_tier": self.ridge_tier,
        }
        for k, v in extras.items():
            if k not in metrics_keys and v is not None:
                base[k] = v
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantFeedback":
        base = Feedback.from_dict(data)
        metrics = data.get("metrics", {})
        return cls(
            decision=base.decision,
            observation=base.observation,
            suggestion=base.suggestion,
            metrics=base.metrics,
            rankicir=data.get("rankicir", metrics.get("rankicir", float("-inf"))),
            ic_positive_ratio=data.get("ic_positive_ratio", metrics.get("ic_positive_ratio", 0.0)),
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
        )


class AutoQuantFactorEvaluator(Evaluator):
    """Evaluate an AutoQuant factor experiment and produce QuantFeedback.

    Thresholds (tunable via constructor)
    ------------------------------------
    - rankicir >= 0.25
    - ic_positive_ratio >= 0.52
    - turnover < 0.5
    - simple_sharpe >= 0.5
    """

    def __init__(
        self,
        *,
        min_rankicir: float | None = None,
        min_ic_positive_ratio: float | None = None,
        max_turnover: float | None = None,
        max_corr: float | None = None,
        min_simple_sharpe: float | None = None,
    ):
        self.min_rankicir = min_rankicir if min_rankicir is not None else _ADM_THRESHOLDS["min_rankicir"]
        self.min_ic_positive_ratio = min_ic_positive_ratio if min_ic_positive_ratio is not None else _ADM_THRESHOLDS["min_ic_positive_ratio"]
        self.max_turnover = max_turnover if max_turnover is not None else _ADM_THRESHOLDS["max_turnover"]
        self.max_corr = max_corr if max_corr is not None else _ADM_THRESHOLDS["max_corr"]
        self.min_simple_sharpe = min_simple_sharpe if min_simple_sharpe is not None else _PipeThresh().min_sharpe_simple

    def evaluate(self, experiment: AutoQuantFactorExperiment) -> QuantFeedback:
        """Evaluate an experiment and return structured feedback.

        Parameters
        ----------
        experiment : AutoQuantFactorExperiment
            Must have completed the runner pipeline (eval_result + bt metrics).

        Returns
        -------
        QuantFeedback
        """
        er = experiment.eval_result or {}
        sm = experiment.simple_bt_metrics or {}
        dm = experiment.detailed_bt_metrics

        # Extract metrics
        rankicir = er.get("rankicir", float("-inf"))
        ic_pos = er.get("ic_positive_ratio", 0.0)
        turnover = er.get("turnover", float("inf"))
        max_corr = er.get("max_corr", 0.0)

        simple_sharpe = sm.get("sharpe")
        simple_mdd = sm.get("max_drawdown")
        simple_ann_ret = sm.get("annual_return")
        simple_calmar = sm.get("calmar")

        detailed_sharpe = None
        detailed_ann_ret = None
        if dm:
            detailed_sharpe = dm.get("sharpe")
            detailed_ann_ret = dm.get("annual_return")

        # Cost drag: simple - detailed annual return
        cost_drag = None
        if simple_ann_ret is not None and detailed_ann_ret is not None:
            cost_drag = simple_ann_ret - detailed_ann_ret

        # Decision: candidate threshold check
        decision = (
            rankicir >= self.min_rankicir
            and ic_pos >= self.min_ic_positive_ratio
            and turnover < self.max_turnover
            and max_corr < self.max_corr
            and (simple_sharpe is None or simple_sharpe >= self.min_simple_sharpe)
        )

        # Observation (natural language summary)
        observation = self._format_observation(
            rankicir, ic_pos, turnover, max_corr,
            simple_sharpe, simple_mdd, simple_ann_ret,
            detailed_sharpe, detailed_ann_ret,
        )

        # Suggestion (improvement direction)
        suggestion = self._generate_suggestion(
            rankicir, ic_pos, turnover, max_corr, simple_sharpe,
        )

        return QuantFeedback(
            decision=decision,
            observation=observation,
            suggestion=suggestion,
            metrics={
                "rankicir": rankicir,
                "ic_positive_ratio": ic_pos,
                "turnover": turnover,
                "max_corr": max_corr,
                "simple_sharpe": simple_sharpe,
                "simple_mdd": simple_mdd,
                "simple_annual_return": simple_ann_ret,
                "detailed_sharpe": detailed_sharpe,
                "detailed_annual_return": detailed_ann_ret,
                "cost_drag": cost_drag,
            },
            rankicir=rankicir,
            ic_positive_ratio=ic_pos,
            turnover=turnover,
            max_corr=max_corr,
            simple_sharpe=simple_sharpe,
            simple_mdd=simple_mdd,
            simple_annual_return=simple_ann_ret,
            simple_calmar=simple_calmar,
            detailed_sharpe=detailed_sharpe,
            detailed_annual_return=detailed_ann_ret,
            cost_drag=cost_drag,
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_observation(
        self,
        rankicir: float,
        ic_pos: float,
        turnover: float,
        max_corr: float,
        simple_sharpe: float | None,
        simple_mdd: float | None,
        simple_ann_ret: float | None,
        detailed_sharpe: float | None,
        detailed_ann_ret: float | None,
    ) -> str:
        """Build a natural-language summary of the experiment results."""
        parts: list[str] = []

        def _status(val: float | None, thresh: float, higher_is_better: bool = True) -> str:
            if val is None:
                return "N/A"
            passed = (val >= thresh) if higher_is_better else (val < thresh)
            return f"{val:.3f} {'PASS' if passed else 'FAIL'}"

        parts.append(f"RankICIR = {_status(rankicir, self.min_rankicir)} (threshold: {self.min_rankicir})")
        parts.append(f"IC+ ratio = {_status(ic_pos, self.min_ic_positive_ratio)} (threshold: {self.min_ic_positive_ratio})")
        parts.append(f"Turnover = {_status(turnover, self.max_turnover, higher_is_better=False)} (threshold: <{self.max_turnover})")
        parts.append(f"Max corr with existing = {_status(max_corr, self.max_corr, higher_is_better=False)} (threshold: <{self.max_corr})")

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

        return "\n".join(parts)

    def _generate_suggestion(
        self,
        rankicir: float,
        ic_pos: float,
        turnover: float,
        max_corr: float,
        simple_sharpe: float | None,
    ) -> str:
        """Generate an improvement suggestion based on failed metrics."""
        suggestions: list[str] = []

        if rankicir < self.min_rankicir:
            suggestions.append(
                f"RankICIR is low ({rankicir:.3f} < {self.min_rankicir}). "
                "Try a longer lookback window, add a volume filter, or combine "
                "with a secondary signal to improve persistence."
            )

        if ic_pos < self.min_ic_positive_ratio:
            suggestions.append(
                f"IC+ ratio is weak ({ic_pos:.1%} < {self.min_ic_positive_ratio:.0%}). "
                "The factor direction may be unstable. Consider inverting the signal "
                "or adding a regime filter."
            )

        if turnover >= self.max_turnover:
            suggestions.append(
                f"Turnover is too high ({turnover:.3f} >= {self.max_turnover}). "
                "Use slower-moving inputs, increase smoothing (e.g. ts_mean), "
                "or apply a delay/decay to reduce churn."
            )

        if max_corr >= self.max_corr:
            suggestions.append(
                f"Max correlation with existing factors is high ({max_corr:.3f} >= {self.max_corr}). "
                "The factor may be a style clone. Try a different construction "
                "or orthogonalize against known style factors."
            )

        if simple_sharpe is not None and simple_sharpe < self.min_simple_sharpe:
            suggestions.append(
                f"Simple Sharpe is weak ({simple_sharpe:.3f} < {self.min_simple_sharpe}). "
                "Review the signal strength and consider tighter universe filtering."
            )

        if not suggestions:
            return "All candidate thresholds are met. Consider pushing for the high bar (Sharpe >= 1.0) or running on a longer history."

        return " ".join(suggestions)
