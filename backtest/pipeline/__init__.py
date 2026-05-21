"""Factor mining pipeline: step1~step9 CLI-driven pipeline.

Usage::

    python -m backtest.pipeline init f_001 --start 20160101 --end 20251231
    python -m backtest.pipeline step1 f_001
    python -m backtest.pipeline step2 f_001
    ...
    python -m backtest.pipeline step9 f_001

Or run all steps at once::

    python -m backtest.pipeline run-all f_001 --start 20160101 --end 20251231
"""

from backtest.pipeline.config import PipelineConfig, StepThresholds
from backtest.pipeline.state import PipelineState, StepResult
from backtest.pipeline.steps import (
    step1_coverage_check,
    step2_neutralization_check,
    step3_icir_check,
    step4_monotonicity_check,
    step5_build_strategy,
    step6_simple_backtest,
    step7_detailed_backtest,
    step8_ridge_r2,
    step9_report_and_admit,
)

__all__ = [
    "PipelineConfig",
    "StepThresholds",
    "PipelineState",
    "StepResult",
    "step1_coverage_check",
    "step2_neutralization_check",
    "step3_icir_check",
    "step4_monotonicity_check",
    "step5_build_strategy",
    "step6_simple_backtest",
    "step7_detailed_backtest",
    "step8_ridge_r2",
    "step9_report_and_admit",
]
