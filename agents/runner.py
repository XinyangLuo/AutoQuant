"""Agent adapter for generated-factor pipeline execution.

The shared generated-factor execution path lives in
``backtest.pipeline.runner``.  This module keeps the historical
``AutoQuantFactorRunner`` API and maps ``PipelineState`` back onto the agent
experiment dataclass.
"""

from __future__ import annotations

from backtest.pipeline.runner import GeneratedFactorPipelineRunner

from .experiment import AutoQuantFactorExperiment


class AutoQuantFactorRunner(GeneratedFactorPipelineRunner):
    """Run the complete AutoQuant pipeline for a single factor experiment."""

    def run(
        self,
        experiment: AutoQuantFactorExperiment,
        from_step: int = 1,
        to_step: int | None = None,
        top_k: int | None = None,
        top_pct: float | None = None,
        decay: int | None = None,
        universe: str | None = None,
        rebalance: str | None = None,
        skip_report: bool | None = None,
    ) -> AutoQuantFactorExperiment:
        experiment.status = "running"

        try:
            run = self.run_factor_code(
                factor_id=experiment.factor_id,
                factor_code=experiment.factor_code,
                from_step=from_step,
                to_step=to_step,
                top_k=top_k,
                top_pct=top_pct,
                decay=decay,
                universe=universe,
                rebalance=rebalance,
                skip_report=bool(skip_report) if skip_report is not None else False,
                skip_mark_rejected=True,
            )
            state = run.state
            experiment.factor_file_path = run.factor_file_path

            experiment.step_results = {
                name: {"passed": sr.passed, "reason": sr.reason, "metrics": sr.metrics}
                for name, sr in state.step_results.items()
            }

            sr3 = state.step_results.get("step3")
            if sr3 and sr3.metrics:
                experiment.eval_result = sr3.metrics

            if state.simple_bt_metrics:
                experiment.simple_bt_metrics = state.simple_bt_metrics
            if state.detailed_bt_metrics:
                experiment.detailed_bt_metrics = state.detailed_bt_metrics

            sr8 = state.step_results.get("step8")
            if sr8 and sr8.metrics:
                experiment.ridge_result = sr8.metrics

            sr9 = state.step_results.get("step9")
            if sr9 and sr9.metrics:
                experiment.residual_icir_result = sr9.metrics

            experiment.report_path = state.artifacts.get("report", "")

            if state.status == "ready_for_review":
                experiment.status = "candidate"
            elif state.status == "quick_pass":
                experiment.status = "quick_pass"
            elif state.is_rejected():
                experiment.status = "rejected"
                last_step = state.last_step()
                if last_step and last_step in state.step_results:
                    sr = state.step_results[last_step]
                    if sr.reason:
                        experiment.error = f"[{last_step}] {sr.reason}"
        except Exception as exc:
            experiment.status = "rejected"
            experiment.error = f"{type(exc).__name__}: {exc}"
            raise

        return experiment
