"""Factor runner — executes the full backtest pipeline for a generated factor.

Delegates to ``backtest.pipeline.steps.run_pipeline()`` — the same shared
function used by ``python -m backtest.pipeline run-all``.  Both paths get
identical pipeline behavior, state persistence, and artifacts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from backtest.config_loader import get_section_or
from backtest.data.storage import MarketStorage
from backtest.factor.compute import apply_variant_pipeline, compute_factor
from backtest.factor.registry import get_factor_meta, sync_registry, unregister
from backtest.factor.storage import FactorStorage
from backtest.pipeline import PipelineState
from backtest.pipeline.steps import run_pipeline

from .experiment import AutoQuantFactorExperiment


class AutoQuantFactorRunner:
    """Run the complete AutoQuant pipeline for a single factor experiment.

    Steps
    -----
    1. Write code to disk -> import to trigger ``@register``
    2. Backfill: ``compute_factor()`` + neutralization -> work DB
    3-10. Canonical pipeline steps (delegated to ``backtest.pipeline.steps``)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        *,
        results_root: Path | str = "results/agent",
        generated_dir: Path | str | None = None,
        market_storage: MarketStorage | None = None,
        factor_storage: FactorStorage | None = None,
        benchmark: str = "000300.SH",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.results_root = Path(results_root)
        self.generated_dir = Path(generated_dir) if generated_dir else Path("alphas/exp/agent")
        self.market_storage = market_storage
        self.factor_storage = factor_storage
        self.benchmark = benchmark

        self._market_storage_owned = market_storage is None
        self._factor_storage_owned = factor_storage is None
        if self.market_storage is None:
            self.market_storage = MarketStorage(read_only=True)
        if self.factor_storage is None:
            self.factor_storage = FactorStorage()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._close_storages()
        except Exception:
            pass
        return False

    def _close_storages(self) -> None:
        if self._market_storage_owned and self.market_storage is not None:
            self.market_storage.close()
            self.market_storage = None
        if self._factor_storage_owned and self.factor_storage is not None:
            self.factor_storage.close()
            self.factor_storage = None

    def cleanup_work_db(self, factor_id: str) -> None:
        if self.factor_storage is None:
            return
        if not self._factor_storage_owned:
            return
        try:
            self.factor_storage.delete_factor(factor_id)
        except Exception as exc:
            print(f"  [cleanup] WARN: failed to drop {factor_id} from work DB: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        experiment: AutoQuantFactorExperiment,
    ) -> AutoQuantFactorExperiment:
        experiment.status = "running"

        try:
            # Phase A: register + backfill (agent-owned, not in pipeline steps)
            self._register_factor(experiment)
            self._backfill_factor(experiment)

            # Phase B: canonical step1~step10 (shared with manual CLI)
            ret_type = get_section_or("open", "pipeline", "ret_type")
            state = run_pipeline(
                factor_id=experiment.factor_id,
                frequency="D",
                start_date=self.start_date,
                end_date=self.end_date,
                results_root=str(self.results_root),
                ret_type=ret_type,
                benchmark=self.benchmark,
                skip_mark_rejected=True,
            )

            # Map pipeline state back to experiment
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

            if state.status == "ready_for_review":
                experiment.status = "candidate"
            elif state.is_rejected():
                experiment.status = "rejected"
                last_step = state.last_step()
                if last_step and last_step in state.step_results:
                    sr = state.step_results[last_step]
                    if sr.reason:
                        experiment.error = f"[{last_step}] {sr.reason}"
        except Exception as e:
            experiment.status = "rejected"
            experiment.error = f"{type(e).__name__}: {e}"
            raise

        return experiment

    # ------------------------------------------------------------------
    # Phase A: Registration + backfill
    # ------------------------------------------------------------------

    def _register_factor(self, experiment: AutoQuantFactorExperiment) -> None:
        if not experiment.factor_code:
            raise ValueError("experiment.factor_code is empty")
        if not experiment.factor_id:
            raise ValueError("experiment.factor_id is empty")

        gen_dir = self.generated_dir
        gen_dir.mkdir(parents=True, exist_ok=True)
        init_file = gen_dir / "__init__.py"
        if not init_file.exists():
            try:
                init_file.write_text("# Auto-generated factor modules\n", encoding="utf-8")
            except (PermissionError, OSError) as e:
                raise RuntimeError(f"Cannot write {init_file}: {e}. Ensure the directory is writable.")

        file_path = gen_dir / f"{experiment.factor_id}.py"
        mod_prefix = str(gen_dir).replace("/", ".").replace("\\", ".")
        mod_name = f"{mod_prefix}.{experiment.factor_id}"
        sys.modules.pop(mod_name, None)

        file_path.write_text(experiment.factor_code, encoding="utf-8")
        experiment.factor_file_path = file_path

        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            unregister(experiment.factor_id)
            raise

        sync_registry()

    def _backfill_factor(self, experiment: AutoQuantFactorExperiment) -> None:
        meta = get_factor_meta(experiment.factor_id)
        raw_df = compute_factor(
            experiment.factor_id,
            self.start_date,
            self.end_date,
            market_storage=self.market_storage,
        )

        variant = meta.get("variant", "barra_ind_size")
        if variant != "none":
            try:
                df = apply_variant_pipeline(
                    raw_df, experiment.factor_id,
                    market_storage=self.market_storage,
                    factor_storage=self.factor_storage,
                )
            except RuntimeError as e:
                msg = str(e).lower()
                if "admitted into" in msg:
                    df = raw_df
                else:
                    raise
        else:
            df = raw_df

        if df.empty:
            raise RuntimeError(
                f"Factor {experiment.factor_id} produced empty DataFrame after compute. "
                "Check that the factor function returns non-NaN values."
            )
        self.factor_storage.insert_factors(df)
