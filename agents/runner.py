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
        results_root: Path | str = "results",
        results_subdir: str | None = None,
        state_subdir: str | None = None,
        generated_dir: Path | str | None = None,
        market_storage: MarketStorage | None = None,
        factor_storage: FactorStorage | None = None,
        factor_storage_read_only: bool = False,
        benchmark: str = "000300.SH",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.results_root = Path(results_root)
        self.results_subdir = results_subdir
        self.state_subdir = state_subdir
        self.generated_dir = Path(generated_dir) if generated_dir else Path("alphas/exp/agent")
        self.market_storage = market_storage
        self.factor_storage = factor_storage
        self.benchmark = benchmark

        self._market_storage_owned = market_storage is None
        self._factor_storage_owned = factor_storage is None
        if self.market_storage is None:
            self.market_storage = MarketStorage(read_only=True)
        if self.factor_storage is None:
            self.factor_storage = FactorStorage(read_only=factor_storage_read_only)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._close_storages()
        except Exception as exc:
            print(f"[runner] warning: failed to close storages: {exc}", file=sys.stderr)
        return False

    def _close_storages(self) -> None:
        if self._market_storage_owned and self.market_storage is not None:
            self.market_storage.close()
            self.market_storage = None
        if self._factor_storage_owned and self.factor_storage is not None:
            self.factor_storage.close()
            self.factor_storage = None

    def cleanup_work_db(self, factor_id: str) -> None:
        if not self._factor_storage_owned:
            return
        try:
            if self.factor_storage is None:
                with FactorStorage() as fs:
                    fs.delete_factor(factor_id)
            else:
                self.factor_storage.delete_factor(factor_id)
        except Exception as exc:
            print(f"  [cleanup] WARN: failed to drop {factor_id} from work DB: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            # Phase A: register + backfill (agent-owned, not in pipeline steps).
            # Skip when re-running from a later step (e.g. strategy-only param changes).
            if from_step == 1:
                self._register_factor(experiment)
                self._backfill_factor(experiment)
            elif from_step > 1:
                if not self._factor_data_exists(experiment.factor_id):
                    raise RuntimeError(
                        f"Factor {experiment.factor_id} has no data in the work DB. "
                        f"Cannot resume from step {from_step} — run with --from-step 1 first "
                        f"to register and backfill the factor."
                    )
                # Guard against stale code: if the user changed the factor formula
                # but kept the same factor_id, old values in the work DB would be
                # silently reused.  Compare the on-disk code with the submitted code.
                if experiment.factor_code:
                    expected_path = self.generated_dir / experiment.factor_id / "factor.py"
                    if expected_path.exists():
                        disk_code = expected_path.read_text(encoding="utf-8")
                        if disk_code.strip() != experiment.factor_code.strip():
                            raise RuntimeError(
                                f"Factor code on disk ({expected_path}) does not match "
                                f"the submitted code. Run with --from-step 1 to re-register "
                                f"the factor."
                            )

            # Release the agent-owned factor DB handle before the pipeline opens
            # its own read/write or read-only handles. DuckDB rejects mixed
            # connection configurations to the same file inside one process.
            if self._factor_storage_owned and self.factor_storage is not None:
                self.factor_storage.close()
                self.factor_storage = None

            # Phase B: canonical step1~step10 (shared with manual CLI)
            ret_type = get_section_or("open", "pipeline", "ret_type")
            state = run_pipeline(
                factor_id=experiment.factor_id,
                frequency="D",
                start_date=self.start_date,
                end_date=self.end_date,
                results_root=str(self.results_root),
                results_subdir=self.results_subdir,
                state_subdir=self.state_subdir,
                ret_type=ret_type,
                benchmark=self.benchmark,
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

            # Capture pipeline diagnostic report path (always generated by run_pipeline)
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

        factor_dir = gen_dir / experiment.factor_id
        factor_dir.mkdir(parents=True, exist_ok=True)
        file_path = factor_dir / "factor.py"
        mod_prefix = str(gen_dir).replace("/", ".").replace("\\", ".")
        mod_name = f"{mod_prefix}.{experiment.factor_id}.factor"
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
            # Clean up gently — if unregister() also raises we must not
            # mask the original exec_module exception.
            try:
                sys.modules.pop(mod_name, None)
                unregister(experiment.factor_id)
            except Exception:
                pass
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

        from backtest.factor.variants import DEFAULT_VARIANT

        variant = meta.get("variant", DEFAULT_VARIANT)
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

    def _factor_data_exists(self, factor_id: str) -> bool:
        """Check whether factor values exist in the work DB."""
        try:
            return factor_id in self.factor_storage.get_existing_factor_ids()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            # First run: table not created yet → treat as "no data".
            if any(
                phrase in msg
                for phrase in ("does not exist", "not found", "no such table")
            ):
                return False
            # Real errors (locked DB, permission denied, corrupted file) →
            # surface them so the user sees the actual problem.
            raise RuntimeError(
                f"Cannot check factor data for {factor_id}: {exc}"
            ) from exc
