"""Pipeline state: serializable dataclass shared across CLI steps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from backtest.pipeline.config import PipelineConfig, StepThresholds


StepName = Literal[
    "step1", "step2", "step3", "step4",
    "step5", "step6", "step7", "step8", "step9", "step10",
]


@dataclass
class StepResult:
    passed: bool
    reason: str | None = None
    metrics: dict = field(default_factory=dict)


@dataclass
class PipelineState:
    factor_id: str
    config: PipelineConfig
    status: Literal["running", "passed", "rejected", "admitted", "ready_for_review"] = "running"
    current_step: StepName | None = None
    step_results: dict[str, StepResult] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    # Shared data populated during pipeline execution (not serialised)
    strategy_config: "StrategyConfig | None" = None
    signals: "pd.DataFrame | None" = None
    simple_bt_metrics: dict | None = None
    detailed_bt_metrics: dict | None = None
    ridge_result: "RidgeCheckResult | None" = None
    residual_icir_result: "ResidualICIRResult | None" = None
    eval_result: "EvaluationResult | None" = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = self.config.state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls, path: Path | None = None) -> PipelineState:
        if path is None:
            raise ValueError("path is required for load")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineState:
        config = _config_from_dict(data.pop("config", {}))
        step_results_raw = data.pop("step_results", {})
        step_results = {
            k: StepResult(**v) for k, v in step_results_raw.items()
        }
        # Discard legacy retry fields for backward compatibility
        data.pop("retry_count", None)
        data.pop("retry_params", None)
        return cls(
            config=config,
            step_results=step_results,
            **data,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_rejected(self) -> bool:
        return self.status == "rejected"

    def is_admitted(self) -> bool:
        return self.status == "admitted"

    def last_step(self) -> str | None:
        return self.current_step

    def get_result(self, step: str) -> StepResult | None:
        return self.step_results.get(step)

    def can_proceed_to(self, step: StepName) -> bool:
        """Check if all prerequisite steps have passed."""
        if self.is_rejected():
            return False
        order = ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10"]
        try:
            idx = order.index(step)
        except ValueError:
            return False
        for prev in order[:idx]:
            r = self.step_results.get(prev)
            if r is None or not r.passed:
                return False
        return True

    def record(self, step: StepName, result: StepResult) -> None:
        self.step_results[step] = result
        self.current_step = step
        if not result.passed:
            self.status = "rejected"
        elif self.status == "rejected":
            # Reset rejection so re-running a previously-failed step
            # doesn't permanently block downstream steps from proceeding.
            self.status = "running"


def _config_from_dict(data: dict) -> PipelineConfig:
    """Rebuild PipelineConfig from JSON, refreshing strategy defaults from config.yaml."""
    from backtest.config_loader import get_section

    # Refresh fields that users edit in config.yaml between runs.
    # CLI overrides (start_date, end_date, frequency, etc.) stay as-serialized.
    refresh_keys = {
        "default_top_k": ("pipeline", "default_top_k"),
        "default_top_pct": ("pipeline", "default_top_pct"),
        "default_decay": ("pipeline", "default_decay"),
        "default_rebalance": ("pipeline", "default_rebalance"),
        "default_universe": ("pipeline", "default_universe"),
    }
    for field_name, section_keys in refresh_keys.items():
        try:
            data[field_name] = get_section(*section_keys)
        except (KeyError, FileNotFoundError, ValueError):
            pass  # keep serialized value

    # Refresh thresholds from config.yaml as well.
    # Reuse the same mapping logic as PipelineConfig.from_yaml().
    th_dict = data.pop("thresholds", {})
    try:
        from backtest.config_loader import get_section as _gs
        from backtest.pipeline.config import StepThresholds as _ST

        pipeline_th = _gs("thresholds", "pipeline")
        for section, vals in pipeline_th.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    field_name = (
                        f"min_{k}"
                        if k.startswith("sharpe")
                           or k.startswith("annual_return")
                           or k.startswith("calmar")
                        else k
                    )
                    if hasattr(_ST, field_name):
                        th_dict[field_name] = v
            elif hasattr(_ST, section):
                th_dict[section] = vals
    except (KeyError, FileNotFoundError, ValueError):
        pass  # keep serialized thresholds
    thresholds = StepThresholds(**th_dict)
    # Discard legacy retry field for backward compatibility
    data.pop("max_retries", None)
    return PipelineConfig(**data, thresholds=thresholds)
