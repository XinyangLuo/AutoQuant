"""Parallel strategy sweep for generated factors.

Used by ``agents.claude_cli sweep`` to explore a grid of strategy parameters
(top_k, decay, rebalance) without recomputing the factor each time.  Each
parameter combination gets its own cloned factor ID so that pipeline state
and artifacts are isolated and the sweep can run in parallel.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

from agents.config import AgentConfig
from agents.evaluator import AutoQuantFactorEvaluator
from agents.experiment import AutoQuantFactorExperiment
from agents.helpers import force_register_factor_id, validate_python_code, validate_transforms_imports
from agents.runner import AutoQuantFactorRunner
from backtest.factor.registry import sync_registry, unregister
from backtest.factor.storage import FactorStorage


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    # numpy scalar types (float32, float64, etc.) are not JSON serializable
    try:
        import numpy as np
        if isinstance(value, np.floating):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
    except ImportError:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp{os.getpid()}")
    tmp.write_text(
        json.dumps(_clean_json(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


def _build_clone_id(factor_id: str, top_k: int, decay: int, rebalance: str) -> str:
    """Stable ID suffix for a strategy-parameter combo."""
    rebalance_safe = rebalance.lower().replace(" ", "")
    return f"{factor_id}_sw_tk{top_k}_d{decay}_{rebalance_safe}"


def _clone_factor_code(code: str, clone_id: str, generated_dir: Path) -> Path:
    """Write factor code rewritten for *clone_id* to disk and return the path."""
    clone_code = force_register_factor_id(code, clone_id)
    clone_dir = generated_dir / clone_id
    clone_dir.mkdir(parents=True, exist_ok=True)
    file_path = clone_dir / "factor.py"
    file_path.write_text(clone_code, encoding="utf-8")
    return file_path


def _copy_factor_values(source_id: str, clone_id: str) -> None:
    """Copy already-backfilled factor values from *source_id* to *clone_id*."""
    with FactorStorage() as fs:
        if clone_id in fs.get_existing_factor_ids():
            return
        df = fs.get_factor(source_id)
        if df.empty:
            raise RuntimeError(f"Source factor {source_id} has no data in work DB")
        # insert_factors expects long format: date, symbol, factor_id, value
        df["factor_id"] = clone_id
        fs.insert_factors(df)


def _delete_clone_factors(clone_ids: list[str]) -> None:
    """Drop temporary sweep columns from the work DB."""
    with FactorStorage() as fs:
        fs.delete_factors(clone_ids)


def _register_clone_module(clone_dir: Path, clone_id: str, generated_dir: Path, *, sync: bool = True) -> None:
    """Import the cloned factor module so the registry knows about it.

    *sync=True* (main process): import and persist to ``registry.json``.
    *sync=False* (worker process): import into in-memory cache only;
    avoids concurrent writes to ``registry.json`` from multiple workers.
    """
    mod_prefix = ".".join(p for p in generated_dir.parts if p)
    mod_name = f"{mod_prefix}.{clone_id}.factor"
    sys.modules.pop(mod_name, None)
    file_path = clone_dir / "factor.py"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if sync:
        sync_registry()


def _run_sweep_combo_worker(
    clone_id: str,
    generated_dir: str,
    results_root: str,
    top_k: int,
    decay: int,
    rebalance: str,
    to_step: int | None,
) -> dict[str, Any]:
    """Worker entry point: run step5+ for a cloned factor ID in isolation.

    Must be a top-level function because it is executed in a subprocess.
    """
    generated_dir_p = Path(generated_dir)
    clone_dir = generated_dir_p / clone_id
    file_path = clone_dir / "factor.py"

    if not file_path.exists():
        return {"clone_id": clone_id, "error": f"factor file not found: {file_path}"}

    code = file_path.read_text(encoding="utf-8")
    experiment = AutoQuantFactorExperiment(factor_id=clone_id, factor_code=code)
    experiment.factor_file_path = file_path

    cfg = AgentConfig()
    feedback = None
    error = None
    tb = None
    try:
        _register_clone_module(clone_dir, clone_id, generated_dir_p, sync=False)
        evaluator = AutoQuantFactorEvaluator()
        with AutoQuantFactorRunner(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            results_root=Path(results_root),
            generated_dir=generated_dir_p,
            factor_storage=FactorStorage(read_only=True),
        ) as runner:
            try:
                experiment = runner.run(
                    experiment,
                    from_step=5,
                    to_step=to_step,
                    top_k=top_k,
                    decay=decay,
                    rebalance=rebalance,
                )
                feedback = evaluator.evaluate(experiment)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                experiment.error = error
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        experiment.error = error

    # Map status to a simple payload that can cross the process boundary.
    status: str
    if error:
        status = "error"
    elif experiment.status == "candidate":
        status = "pass"
    elif experiment.status == "quick_pass":
        status = "quick_pass"
    else:
        status = "fail"

    result_path = Path(results_root) / clone_id / "result.json"
    if experiment.report_path:
        result_path = Path(experiment.report_path).parent / "result.json"

    metrics: dict[str, Any] = feedback.metrics if feedback else {}
    payload = {
        "clone_id": clone_id,
        "status": status,
        "error": error,
        "traceback": tb,
        "params": {"top_k": top_k, "decay": decay, "rebalance": rebalance},
        "metrics": _clean_json(metrics),
        "result_path": str(result_path),
        "report_path": experiment.report_path,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(result_path, payload)
    return payload


def run_sweep(
    factor_id: str,
    factor_file: Path,
    generated_dir: Path,
    results_root: Path,
    top_ks: list[int],
    decays: list[int],
    rebalances: list[str],
    *,
    to_step: int | None = 6,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Run a parallel strategy sweep over the Cartesian product of params.

    Steps
    -----
    1. Ensure the base factor is registered and backfilled.
    2. For each param combo, clone the factor code and copy factor values
       into the work DB under a unique clone ID.
    3. Run step5+ for each clone in parallel.
    4. Return a list of per-combo result payloads; the caller prints the
       summary table.
    """
    cfg = AgentConfig()
    code = factor_file.read_text(encoding="utf-8")
    validate_python_code(code)
    validate_transforms_imports(code)
    code = force_register_factor_id(code, factor_id)

    # Phase A: backfill the base factor once.
    experiment = AutoQuantFactorExperiment(factor_id=factor_id, factor_code=code)
    with AutoQuantFactorRunner(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        results_root=results_root,
        generated_dir=generated_dir,
    ) as runner:
        experiment = runner.run(experiment, from_step=1, to_step=4)
        if experiment.status != "quick_pass" and experiment.status != "candidate":
            # step1-4 must succeed before we can sweep strategy params.
            raise RuntimeError(
                f"Base factor failed at step1-4 (status={experiment.status}). "
                f"Cannot start strategy sweep."
            )

    combos = list(product(top_ks, decays, rebalances))
    clone_ids: list[str] = []
    clone_files: list[Path] = []

    results: list[dict[str, Any]] = []
    try:
        for top_k, decay, rebalance in combos:
            clone_id = _build_clone_id(factor_id, top_k, decay, rebalance)
            clone_ids.append(clone_id)
            clone_file = _clone_factor_code(code, clone_id, generated_dir)
            clone_files.append(clone_file)
            _copy_factor_values(factor_id, clone_id)

        max_workers = min(workers, len(combos)) if combos else 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_sweep_combo_worker,
                    clone_id,
                    str(generated_dir),
                    str(results_root),
                    top_k,
                    decay,
                    rebalance,
                    to_step,
                ): (top_k, decay, rebalance)
                for (top_k, decay, rebalance), clone_id in zip(combos, clone_ids)
            }
            for future in as_completed(futures):
                try:
                    payload = future.result()
                except Exception as exc:
                    top_k, decay, rebalance = futures[future]
                    payload = {
                        "clone_id": _build_clone_id(factor_id, top_k, decay, rebalance),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "params": {"top_k": top_k, "decay": decay, "rebalance": rebalance},
                        "metrics": {},
                        "result_path": None,
                        "report_path": None,
                    }
                results.append(payload)
    finally:
        # Clean up temporary sweep columns from work DB.
        if clone_ids:
            try:
                _delete_clone_factors(clone_ids)
            except Exception as exc:
                print(f"[sweep] warning: failed to clean up clone factors: {exc}")
        # Remove generated factor modules so they don't pollute alphas/exp/agent.
        import shutil

        for clone_id in clone_ids:
            clone_dir = generated_dir / clone_id
            if clone_dir.exists():
                try:
                    shutil.rmtree(clone_dir)
                except Exception as exc:
                    print(f"[sweep] warning: failed to remove clone dir {clone_dir}: {exc}")
        # Unregister cloned modules to keep the parent registry tidy.
        for clone_id in clone_ids:
            try:
                unregister(clone_id)
            except Exception as exc:
                print(f"[sweep] warning: failed to unregister {clone_id}: {exc}")

    return results
    