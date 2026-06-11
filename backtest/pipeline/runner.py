"""Generated-factor execution for the canonical pipeline.

This module owns the shared path used when a caller has factor source code,
not just an already-backfilled factor id:

1. write/import the factor code so ``@register`` runs
2. compute and backfill factor values into the work DB
3. delegate all gates/backtests to ``run_pipeline()``
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from backtest.config_loader import get_section_or
from backtest.data.storage import MarketStorage
from backtest.factor.compute import apply_variant_pipeline, compute_factor
from backtest.factor.registry import get_factor_meta, sync_registry, unregister
from backtest.factor.storage import FactorStorage
from backtest.pipeline.state import PipelineState
from backtest.pipeline.steps import run_pipeline


@dataclass
class GeneratedFactorRun:
    """Result of running a generated factor through the pipeline."""

    state: PipelineState
    factor_file_path: Path | None = None


class GeneratedFactorPipelineRunner:
    """Run factor source code through backfill and the canonical pipeline."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        *,
        results_root: Path | str = "results",
        results_subdir: str | None = None,
        state_subdir: str | None = None,
        frequency: str = "D",
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
        self.frequency = frequency
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

    def run_factor_code(
        self,
        *,
        factor_id: str,
        factor_code: str,
        from_step: int = 1,
        to_step: int | None = None,
        top_k: int | None = None,
        top_pct: float | None = None,
        decay: int | None = None,
        universe: str | None = None,
        rebalance: str | None = None,
        skip_report: bool = False,
        skip_mark_rejected: bool = True,
    ) -> GeneratedFactorRun:
        """Register/backfill factor code, then run the shared pipeline."""

        factor_file_path: Path | None = None
        if from_step == 1:
            factor_file_path = self._register_factor(factor_id, factor_code)
            self._backfill_factor(factor_id)
        elif from_step > 1:
            if not self._factor_data_exists(factor_id):
                raise RuntimeError(
                    f"Factor {factor_id} has no data in the work DB. "
                    f"Cannot resume from step {from_step} -- run with --from-step 1 first "
                    f"to register and backfill the factor."
                )
            expected_path = self.generated_dir / factor_id / "factor.py"
            if expected_path.exists():
                factor_file_path = expected_path
                disk_code = expected_path.read_text(encoding="utf-8")
                if factor_code and disk_code.strip() != factor_code.strip():
                    raise RuntimeError(
                        f"Factor code on disk ({expected_path}) does not match "
                        f"the submitted code. Run with --from-step 1 to re-register "
                        f"the factor."
                    )

        # Release the factor DB handle before run_pipeline opens its own
        # handles. DuckDB rejects mixed read/write modes in one process.
        if self._factor_storage_owned and self.factor_storage is not None:
            self.factor_storage.close()
            self.factor_storage = None

        ret_type = get_section_or("open", "pipeline", "ret_type")
        state = run_pipeline(
            factor_id=factor_id,
            frequency=self.frequency,
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
            skip_report=skip_report,
            skip_mark_rejected=skip_mark_rejected,
        )
        return GeneratedFactorRun(state=state, factor_file_path=factor_file_path)

    def _register_factor(self, factor_id: str, factor_code: str) -> Path:
        if not factor_code:
            raise ValueError("factor_code is empty")
        if not factor_id:
            raise ValueError("factor_id is empty")

        gen_dir = self.generated_dir
        gen_dir.mkdir(parents=True, exist_ok=True)
        init_file = gen_dir / "__init__.py"
        if not init_file.exists():
            try:
                init_file.write_text("# Auto-generated factor modules\n", encoding="utf-8")
            except (PermissionError, OSError) as exc:
                raise RuntimeError(
                    f"Cannot write {init_file}: {exc}. Ensure the directory is writable."
                ) from exc

        factor_dir = gen_dir / factor_id
        factor_dir.mkdir(parents=True, exist_ok=True)
        file_path = factor_dir / "factor.py"
        mod_prefix = str(gen_dir).replace("/", ".").replace("\\", ".")
        mod_name = f"{mod_prefix}.{factor_id}.factor"
        sys.modules.pop(mod_name, None)

        file_path.write_text(factor_code, encoding="utf-8")

        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            try:
                sys.modules.pop(mod_name, None)
                unregister(factor_id)
            except Exception:
                pass
            raise

        sync_registry()
        return file_path

    def _backfill_factor(self, factor_id: str) -> None:
        meta = get_factor_meta(factor_id)
        raw_df = compute_factor(
            factor_id,
            self.start_date,
            self.end_date,
            market_storage=self.market_storage,
        )

        from backtest.factor.variants import DEFAULT_VARIANT

        variant = meta.get("variant", DEFAULT_VARIANT)
        if variant != "none":
            try:
                df = apply_variant_pipeline(
                    raw_df,
                    factor_id,
                    market_storage=self.market_storage,
                    factor_storage=self.factor_storage,
                )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "admitted into" in msg:
                    df = raw_df
                else:
                    raise
        else:
            df = raw_df

        if df.empty:
            raise RuntimeError(
                f"Factor {factor_id} produced empty DataFrame after compute. "
                "Check that the factor function returns non-NaN values."
            )
        self.factor_storage.insert_factors(df)

    def _factor_data_exists(self, factor_id: str) -> bool:
        """Check whether factor values exist in the work DB."""
        try:
            return factor_id in self.factor_storage.get_existing_factor_ids()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if any(
                phrase in msg
                for phrase in ("does not exist", "not found", "no such table")
            ):
                return False
            raise RuntimeError(
                f"Cannot check factor data for {factor_id}: {exc}"
            ) from exc
