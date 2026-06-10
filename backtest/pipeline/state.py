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
    status: Literal["running", "passed", "rejected", "admitted", "ready_for_review", "quick_pass"] = "running"
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
        d = asdict(self)
        # Drop non-serializable transient fields to avoid bloating the JSON
        for key in (
            "signals", "simple_bt_metrics", "detailed_bt_metrics",
            "ridge_result", "residual_icir_result", "eval_result",
        ):
            d.pop(key, None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PipelineState:
        # Use .get() rather than .pop() to avoid mutating the caller's dict.
        config = _config_from_dict(data.get("config", {}))
        step_results_raw = data.get("step_results", {})
        step_results = {
            k: StepResult(**v) for k, v in step_results_raw.items()
        }
        # Restore eval_result if present (EvaluationResult serialises via
        # to_dict(); discard on deserialisation error so pipeline can fall
        # back to re-running evaluate() in step4).
        eval_result = None
        er_raw = data.get("eval_result")
        if isinstance(er_raw, dict):
            try:
                from backtest.factor.evaluation import EvaluationResult
                eval_result = EvaluationResult.from_dict(er_raw)
            except Exception:
                pass  # fallback to None; step4 will re-run evaluate if needed

        # Discard legacy retry fields for backward compatibility
        # (not present in current serialized state; harmless if absent).
        return cls(
            config=config,
            step_results=step_results,
            factor_id=data.get("factor_id", ""),
            status=data.get("status", "running"),
            current_step=data.get("current_step"),
            artifacts=data.get("artifacts", {}),
            eval_result=eval_result,
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

    def clear_from_step(self, step: StepName) -> None:
        """Discard step_results and artifacts from ``step`` onward.

        Used when resuming a pipeline with new config/strategy overrides so
        that downstream results are recomputed instead of silently reused.
        """
        order = [
            "step1", "step2", "step3", "step4", "step5",
            "step6", "step7", "step8", "step9", "step10",
        ]
        if step not in order:
            return
        start_idx = order.index(step)
        for s in order[start_idx:]:
            self.step_results.pop(s, None)
        # Artifacts keyed by step tag or report path are also stale once
        # the corresponding steps are re-run.  Remove only the artifact
        # entries that depend on downstream steps; leave eval_result etc.
        stale_artifact_keys = {
            "strategy_config", "simple_bt", "detailed_bt",
            "ridge", "residual_icir", "report",
        }
        for key in stale_artifact_keys:
            self.artifacts.pop(key, None)
        # Non-serialized cached objects that depend on cleared steps must
        # also be wiped so that downstream steps don't reuse stale data.
        if start_idx < 3:
            self.eval_result = None
        if start_idx < 4:
            self.strategy_config = None
            self.signals = None
        if start_idx < 6:
            self.simple_bt_metrics = None
        if start_idx < 7:
            self.detailed_bt_metrics = None
        if start_idx < 8:
            self.ridge_result = None
        if start_idx < 9:
            self.residual_icir_result = None
        # Reset status so downstream steps can proceed.
        if self.status in ("rejected", "ready_for_review", "quick_pass"):
            self.status = "running"


def _config_from_dict(data: dict) -> PipelineConfig:
    """Rebuild PipelineConfig from JSON, refreshing strategy defaults from config.yaml.

    .. note::

       Strategy defaults (top_k, decay, rebalance, etc.) and thresholds are
       **intentionally refreshed from the current config.yaml** when
       deserializing.  This means a reloaded state may show different
       threshold values than when the run originally executed — the trade-off
       is that resumed runs always use the latest settings.
    """
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
    # Work on a copy to avoid mutating the caller's dict.
    data = dict(data)
    th_dict = data.pop("thresholds", {})
    try:
        from backtest.pipeline.config import _collect_threshold_overrides

        pipeline_th = get_section("thresholds", "pipeline")
        config_overrides = _collect_threshold_overrides(pipeline_th)
        th_dict.update(config_overrides)
    except (KeyError, FileNotFoundError, ValueError):
        pass  # keep serialized thresholds
    thresholds = StepThresholds(**th_dict)
    return PipelineConfig(**data, thresholds=thresholds)
