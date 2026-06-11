"""Multi-universe parallel strategy sweep for generated factors.

Used by ``agents.claude_cli sweep`` to explore strategy parameters
(top_pct=10%, decay, rebalance) across the four main宽基指数 universes.

One factor hypothesis → one factor implementation.  Each
(universe × decay × rebalance) combination is a strategy variant.
Universes run **serially** (to avoid DB contention); strategy combos
within a universe run in **parallel** via ProcessPoolExecutor.

Directory layout::

    results/{factor_id}/
      hs300/
        factor_eval/          # per-universe IC / decile
        top10pct_1D_d5/       # strategy variant
          simple/
          detailed/
          plots/
          pipeline_report.md
      csi500/
        ...
      cross_universe.json     # final comparison + best per universe
"""

from __future__ import annotations

import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from agents.config import AgentConfig
from agents.evaluator import AutoQuantFactorEvaluator
from agents.experiment import AutoQuantFactorExperiment
from agents.helpers import force_register_factor_id, validate_python_code, validate_transforms_imports
from agents.runner import AutoQuantFactorRunner
from backtest.factor.registry import get_factor_meta
from backtest.factor.storage import FactorStorage
from backtest.pipeline.config import PipelineConfig
from backtest.pipeline.state import PipelineState
from backtest.pipeline.steps import (
    step1_coverage_check,
    step2_neutralization_check,
    step3_icir_check,
    step4_monotonicity_check,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNIVERSES: dict[str, str] = {
    "hs300": "000300.SH",
    "csi500": "000905.SH",
    "csi1000": "000852.SH",
    "csi2000": "932000.CSI",
}

# Selection: always top 10 %.
TOP_PCT = 0.1

# Strategy parameter grids by factor type.
_PRICE_VOLUME_COMBOS = list(product(
    [5, 10, 15],          # decay
    ["1D", "5D"],         # rebalance
))
_FUNDAMENTAL_COMBOS = list(product(
    [5],                  # decay
    ["1M", "3M"],         # rebalance
))

# ---------------------------------------------------------------------------
# Factor-type detection
# ---------------------------------------------------------------------------

_FINANCIAL_SOURCES = frozenset({"income_q", "balancesheet_q", "cashflow_q"})


def _is_fundamental(factor_id: str) -> bool:
    """Return True if the factor uses financial-statement data sources."""
    try:
        meta = get_factor_meta(factor_id)
    except Exception:
        return False
    sources = meta.get("data_sources", [])
    return bool(_FINANCIAL_SOURCES & set(sources))


def _get_combos(factor_id: str) -> list[tuple[int, str]]:
    """Return (decay, rebalance) combos appropriate for this factor type."""
    if _is_fundamental(factor_id):
        return _FUNDAMENTAL_COMBOS
    return _PRICE_VOLUME_COMBOS


# ---------------------------------------------------------------------------
# Helpers (shared with original sweep)
# ---------------------------------------------------------------------------

from backtest.evaluation.benchmark import _INDEX_TO_BENCHMARK_ALIAS as _BENCH_INDEX_MAP


def _index_to_alias(code: str) -> str:
    """Map an index ts_code to its benchmark alias, falling back to 'hs300'."""
    return _BENCH_INDEX_MAP.get(code, "hs300")


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np
        if isinstance(value, np.floating):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return str(value)
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


def _combo_tag(decay: int, rebalance: str) -> str:
    """Stable strategy-combo tag.  Always uses top_pct=0.1 (top10pct)."""
    return f"top10pct_{rebalance.lower()}_d{decay}"


def _print_progress(message: str) -> None:
    print(f"[sweep] {message}", flush=True)


def _fmt_progress_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _progress_line(
    prefix: str, done: int, total: int, result: dict[str, Any],
) -> str:
    params = result.get("params", {}) or {}
    metrics = result.get("metrics", {}) or {}
    tag = result.get("combo_tag") or _combo_tag(
        int(params.get("decay", 0)),
        str(params.get("rebalance", "")),
    )
    line = (
        f"{prefix} {done}/{total} done {tag} "
        f"status={result.get('status', 'unknown')} "
        f"sharpe={_fmt_progress_metric(metrics.get('simple_sharpe'))} "
        f"ann_ret={_fmt_progress_metric(metrics.get('simple_annual_return'))} "
        f"mdd={_fmt_progress_metric(metrics.get('simple_mdd'))} "
        f"calmar={_fmt_progress_metric(metrics.get('simple_calmar'))}"
    )
    error = result.get("error")
    if error:
        line += f" error={error}"
    return line


# ---------------------------------------------------------------------------
# State seeding
# ---------------------------------------------------------------------------


def _seed_combo_state(
    *,
    base_state: PipelineState,
    factor_id: str,
    results_root: Path,
    results_subdir: str,
    state_subdir: str,
) -> Path:
    """Seed a per-combo pipeline state with base step1~step4 results.

    Workers start at step5.  Each worker has its own ``results_root`` so
    they do not contend on ``pipeline_state.json``.
    """
    combo_config = replace(
        base_state.config,
        results_root=str(results_root),
        results_subdir=results_subdir,
        state_subdir=state_subdir,
    )
    combo_state = PipelineState(
        factor_id=factor_id,
        config=combo_config,
        status="running",
        current_step="step4",
        step_results={
            step: result
            for step, result in base_state.step_results.items()
            if step.startswith("step") and int(step[4:]) <= 4
        },
        artifacts={
            key: value
            for key, value in base_state.artifacts.items()
            if key in {"eval_result"}
        },
    )
    state_path = combo_config.state_path()
    combo_state.save(state_path)
    return state_path


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_combo_worker(
    factor_id: str,
    factor_code: str,
    generated_dir: str,
    results_root: str,
    results_subdir: str,
    state_subdir: str,
    universe_code: str,
    universe_name: str,
    decay: int,
    rebalance: str,
    from_step: int,
    to_step: int | None,
    factor_cache: str = "",
    market_cache: str = "",
) -> dict[str, Any]:
    """Worker: run one (universe × strategy) combo in isolation."""
    # Set cache paths in this worker's own environment so
    # StrategyBase.run() picks them up regardless of the
    # multiprocessing start method (fork vs spawn).
    old_factor_cache = os.environ.get("AQ_FACTOR_CACHE")
    old_market_cache = os.environ.get("AQ_MARKET_CACHE")
    try:
        if factor_cache:
            os.environ["AQ_FACTOR_CACHE"] = factor_cache
        if market_cache:
            os.environ["AQ_MARKET_CACHE"] = market_cache

        tag = _combo_tag(decay, rebalance)
        generated_dir_p = Path(generated_dir)
        results_root_p = Path(results_root)
        combo_dir = results_root_p / state_subdir

        experiment = AutoQuantFactorExperiment(factor_id=factor_id, factor_code=factor_code)
        experiment.factor_file_path = generated_dir_p / factor_id / "factor.py"

        cfg = AgentConfig()
        feedback = None
        error = None
        tb = None
        try:
            evaluator = AutoQuantFactorEvaluator()
            with AutoQuantFactorRunner(
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                results_root=results_root_p,
                results_subdir=results_subdir,
                state_subdir=state_subdir,
                generated_dir=generated_dir_p,
                factor_storage_read_only=True,
                benchmark=universe_code,
            ) as runner:
                try:
                    kwargs: dict[str, Any] = {"from_step": from_step, "to_step": to_step}
                    if from_step <= 5:
                        kwargs.update(
                            top_pct=TOP_PCT,
                            decay=decay,
                            rebalance=rebalance,
                            universe=universe_code,
                        )
                    experiment = runner.run(experiment, **kwargs)
                    feedback = evaluator.evaluate(experiment)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    tb = traceback.format_exc()
                    experiment.error = error
        except Exception as exc:
            # Only record init/enter/exit errors if the inner run didn't already
            # fail — avoid overwriting the root-cause error with a cleanup error.
            if error is None:
                error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                experiment.error = error

        if error:
            status = "error"
        elif experiment.status == "candidate":
            status = "pass"
        elif experiment.status == "quick_pass":
            status = "quick_pass"
        else:
            status = "fail"

        result_path = combo_dir / "result.json"
        if experiment.report_path:
            result_path = Path(experiment.report_path).parent / "result.json"

        metrics: dict[str, Any] = feedback.metrics if feedback else {}
        payload = {
            "factor_id": factor_id,
            "combo_tag": tag,
            "universe": universe_name,
            "universe_code": universe_code,
            "status": status,
            "error": error,
            "traceback": tb,
            "params": {
                "top_pct": TOP_PCT,
                "decay": decay,
                "rebalance": rebalance,
                "universe": universe_name,
            },
            "metrics": _clean_json(metrics),
            "result_path": str(result_path),
            "report_path": experiment.report_path,
            "results_root": str(results_root_p),
            "results_subdir": results_subdir,
            "state_subdir": state_subdir,
            "results_dir": str(combo_dir),
        }

        result_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(result_path, payload)
        return payload
    finally:
        if old_factor_cache is None:
            os.environ.pop("AQ_FACTOR_CACHE", None)
        else:
            os.environ["AQ_FACTOR_CACHE"] = old_factor_cache
        if old_market_cache is None:
            os.environ.pop("AQ_MARKET_CACHE", None)
        else:
            os.environ["AQ_MARKET_CACHE"] = old_market_cache


# ---------------------------------------------------------------------------
# Scoring & selection
# ---------------------------------------------------------------------------


def _score_result(result: dict[str, Any]) -> float:
    """Composite score: calmar first (risk-adjusted), then penalize deep drawdowns.

    max_drawdown (simple_mdd) is included because it is a hard gate in
    _backtest_gate — scoring without it can promote combos that the gate
    will reject, wasting Phase C validation time.
    """
    metrics = result.get("metrics", {}) or {}
    calmar = metrics.get("simple_calmar")
    if calmar is None or (isinstance(calmar, float) and math.isnan(calmar)):
        return float("-inf")
    mdd = metrics.get("simple_mdd") or 0.0
    # Penalize deep drawdowns: score = calmar * (1 + mdd) since mdd is negative.
    # A -0.50 drawdown halves the score vs -0.10.
    penalty = 1.0 + float(mdd) if isinstance(mdd, (int, float)) and mdd < 0 else 1.0
    return float(calmar) * max(penalty, 0.01)


def _select_best(results: list[dict[str, Any]], n: int = 1) -> list[dict[str, Any]]:
    candidates = [r for r in results if r.get("status") in ("quick_pass", "pass")]
    return sorted(candidates, key=_score_result, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Factor data helpers
# ---------------------------------------------------------------------------


def _factor_values_exist(factor_id: str) -> bool:
    try:
        with FactorStorage(read_only=True) as fs:
            return factor_id in fs.get_existing_factor_ids()
    except Exception:
        return False


def _run_base_steps(
    *,
    factor_id: str,
    cfg: AgentConfig,
    results_root: Path,
) -> PipelineState:
    """Run step1~step4 using already-backfilled factor values."""
    config = PipelineConfig.from_factor_config(
        factor_id=factor_id,
        frequency="D",
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        results_root=str(results_root),
    )
    state = PipelineState(factor_id=factor_id, config=config)
    for step_fn in (
        step1_coverage_check,
        step2_neutralization_check,
        step3_icir_check,
        step4_monotonicity_check,
    ):
        state = step_fn(state)
        state.save(config.state_path())
        if state.is_rejected():
            raise RuntimeError(
                f"Base factor failed at {state.last_step()} while seeding sweep."
            )
    state.status = "quick_pass"
    state.save(config.state_path())
    return state


# ---------------------------------------------------------------------------
# Shared data cache (avoids redundant DuckDB queries across workers)
# ---------------------------------------------------------------------------

_CACHE_ENV_FACTOR = "AQ_FACTOR_CACHE"
_CACHE_ENV_MARKET = "AQ_MARKET_CACHE"


def _warm_shared_cache(
    factor_id: str,
    cfg: "AgentConfig",
    results_root: Path,
) -> tuple[str, str]:
    """Pre-load factor + market panels and write to parquet so every
    worker reads fast local files instead of hitting DuckDB.

    Returns (factor_cache_path, market_cache_path) so callers can pass
    them explicitly to workers — env vars are unreliable across
    ``ProcessPoolExecutor`` on spawn-based platforms (macOS).
    """
    import os as _os

    cache_dir = results_root / factor_id / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- factor panel ---
    factor_cache = cache_dir / "factor_panel.parquet"
    if not factor_cache.exists():
        from backtest.factor.storage import FactorStorage
        from backtest.strategy.base import StrategyBase

        with FactorStorage(read_only=True) as fs:
            factor_panel = StrategyBase._load_factor_panel(
                [factor_id], cfg.start_date, cfg.end_date, fs,
            )
        factor_panel.to_parquet(factor_cache, index=False)

    # --- full market panel (all symbols) ---
    market_cache = cache_dir / "market_panel.parquet"
    if not market_cache.exists():
        from backtest.data.storage import MarketStorage

        with MarketStorage(read_only=True) as ms:
            market_panel = ms.get_bars(
                symbols=None,
                start=cfg.start_date,
                end=cfg.end_date,
                columns=[
                    "close", "open", "high", "low", "circ_mv", "amount",
                    "is_st", "list_date", "limit_up", "limit_down",
                ],
            )
        market_panel.to_parquet(market_cache, index=False)

    return str(factor_cache), str(market_cache)


# ---------------------------------------------------------------------------
# Main sweep entry point
# ---------------------------------------------------------------------------


def run_sweep(
    factor_id: str,
    factor_file: Path,
    generated_dir: Path,
    results_root: Path,
    *,
    to_step: int | None = 7,
    workers: int = 4,
    validate_top_n: int = 1,
    universes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a multi-universe strategy sweep.

    Parameters
    ----------
    factor_id : str
    factor_file : Path
        Path to the factor.py source.
    generated_dir : Path
        ``alphas/exp/agent/`` directory.
    results_root : Path
        Root for all result artifacts.
    to_step : int | None
        Stop after this step (default 7 = detailed backtest).
    workers : int
        Max parallel workers *per universe*.
    validate_top_n : int
        Number of top combos to validate per universe with full detailed BT.
    universes : dict | None
        Override the default 4-index universe set.

    Returns
    -------
    dict
        ``{factor_id, factor_type, universes: {name: {best, all_results, ...}}, ...}``
    """
    universes = universes or UNIVERSES
    cfg = AgentConfig()
    code = factor_file.read_text(encoding="utf-8")
    validate_python_code(code)
    validate_transforms_imports(code)
    code = force_register_factor_id(code, factor_id)

    combos = _get_combos(factor_id)
    factor_type = "fundamental" if _is_fundamental(factor_id) else "price_volume"
    _print_progress(
        f"Preparing {factor_id} ({factor_type}): "
        f"{len(universes)} universes × {len(combos)} strategy combos = "
        f"{len(universes) * len(combos)} total"
    )

    # --- Phase A: base step1~step4 (universe-independent) -------------------
    if _factor_values_exist(factor_id):
        _print_progress("Reusing existing factor values; running base gates step1-step4")
        base_state = _run_base_steps(
            factor_id=factor_id, cfg=cfg, results_root=results_root,
        )
    else:
        _print_progress("Running base factor through register/backfill and step1-step4 once")
        experiment = AutoQuantFactorExperiment(factor_id=factor_id, factor_code=code)
        with AutoQuantFactorRunner(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            results_root=results_root,
            generated_dir=generated_dir,
        ) as runner:
            experiment = runner.run(experiment, from_step=1, to_step=4)
            if experiment.status not in ("quick_pass", "candidate"):
                raise RuntimeError(
                    f"Base factor failed at step1-4 (status={experiment.status}). "
                    f"Cannot start strategy sweep."
                )
        base_state = PipelineState.load(results_root / factor_id / "pipeline_state.json")
    _print_progress("Base factor passed step1-step4")

    # --- Pre-warm shared data cache for all workers --------------------------
    _print_progress("Pre-loading shared factor + market data for worker cache ...")
    factor_cache_path, market_cache_path = _warm_shared_cache(factor_id, cfg, results_root)
    _print_progress("Shared cache ready")

    # --- Phase B: per-universe sweep (serial universes) ---------------------
    universe_results: dict[str, dict[str, Any]] = {}

    for uni_name, uni_code in universes.items():
        _print_progress(
            f"--- Universe: {uni_name} ({uni_code}) "
            f"[{list(universes.keys()).index(uni_name) + 1}/{len(universes)}] ---"
        )

        # Seed per-combo state for this universe.
        uni_subdir = f"{factor_id}/{uni_name}"
        combo_state_subdirs: dict[tuple[int, str], str] = {}
        for decay, rebalance in combos:
            tag = _combo_tag(decay, rebalance)
            state_subdir = f"{uni_subdir}/{tag}"
            combo_state_subdirs[(decay, rebalance)] = state_subdir
            _seed_combo_state(
                base_state=base_state,
                factor_id=factor_id,
                results_root=results_root,
                results_subdir=uni_subdir,
                state_subdir=state_subdir,
            )

        # Run all combos for this universe in parallel.
        results: list[dict[str, Any]] = []
        max_workers = min(workers, len(combos)) if combos else 1
        stop_step = to_step if to_step is not None else 10

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_combo_worker,
                    factor_id,
                    code,
                    str(generated_dir),
                    str(results_root),
                    uni_subdir,
                    combo_state_subdirs[(decay, rebalance)],
                    uni_code,
                    uni_name,
                    decay,
                    rebalance,
                    5,
                    to_step,
                    factor_cache_path,
                    market_cache_path,
                ): (decay, rebalance)
                for decay, rebalance in combos
            }
            for future in as_completed(futures):
                decay, rebalance = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    tag = _combo_tag(decay, rebalance)
                    payload = {
                        "factor_id": factor_id,
                        "combo_tag": tag,
                        "universe": uni_name,
                        "universe_code": uni_code,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "params": {
                            "top_pct": TOP_PCT,
                            "decay": decay,
                            "rebalance": rebalance,
                            "universe": uni_name,
                        },
                        "metrics": {},
                        "result_path": None,
                        "report_path": None,
                        "results_root": str(results_root),
                        "results_subdir": uni_subdir,
                        "state_subdir": combo_state_subdirs[(decay, rebalance)],
                    }
                results.append(payload)
                _print_progress(
                    _progress_line(uni_name, len(results), len(combos), payload),
                )

        # --- Phase C: validate top combos for this universe (step7 full) -----
        if validate_top_n > 0:
            top_results = _select_best(results, validate_top_n)
            if top_results:
                full_results: dict[str, dict[str, Any]] = {}
                v_workers = min(workers, len(top_results)) if top_results else 1
                _print_progress(
                    f"  {uni_name}: validating top {len(top_results)} combo(s) "
                    f"from step7 with {v_workers} workers"
                )
                with ProcessPoolExecutor(max_workers=v_workers) as executor:
                    full_futures = {}
                    for result in top_results:
                        params = result["params"]
                        full_futures[
                            executor.submit(
                                _run_combo_worker,
                                factor_id,
                                code,
                                str(generated_dir),
                                result["results_root"],
                                result["results_subdir"],
                                result["state_subdir"],
                                uni_code,
                                uni_name,
                                params["decay"],
                                params["rebalance"],
                                7,
                                None,
                                factor_cache_path,
                                market_cache_path,
                            )
                        ] = result["combo_tag"]
                    for future in as_completed(full_futures):
                        tag = full_futures[future]
                        try:
                            full_results[tag] = future.result()
                        except Exception as exc:
                            full_results[tag] = {
                                "factor_id": factor_id,
                                "combo_tag": tag,
                                "universe": uni_name,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "metrics": {},
                            }
                        _print_progress(
                            _progress_line(
                                f"{uni_name}-full",
                                len(full_results),
                                len(top_results),
                                full_results[tag],
                            )
                        )

                for result in results:
                    full = full_results.get(result["combo_tag"])
                    if full is not None:
                        result["full_result"] = full
                        result["full_status"] = full.get("status")
                        result["full_metrics"] = full.get("metrics", {})
                        result["full_result_path"] = full.get("result_path")
                        result["full_report_path"] = full.get("report_path")
            else:
                _print_progress(f"  {uni_name}: no pass combos to validate")

        # Select best for this universe.
        best = _select_best(results, 1)
        universe_results[uni_name] = {
            "universe_code": uni_code,
            "n_combos": len(combos),
            "n_results": len(results),
            "best": best[0] if best else None,
            "all_results": results,
        }

        if best:
            b = best[0]
            _print_progress(
                f"  >>> {uni_name} best: {b['combo_tag']} "
                f"sharpe={_fmt_progress_metric(b.get('metrics', {}).get('simple_sharpe'))} "
                f"calmar={_fmt_progress_metric(b.get('metrics', {}).get('simple_calmar'))}"
            )

    # --- Phase D: cross-universe comparison ---------------------------------
    cross_summary = _build_cross_universe_summary(factor_id, factor_type, universe_results)
    cross_path = results_root / factor_id / "cross_universe.json"
    _write_json(cross_path, cross_summary)
    _print_progress(f"Cross-universe summary: {cross_path}")

    return cross_summary


def _build_cross_universe_summary(
    factor_id: str,
    factor_type: str,
    universe_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a cross-universe comparison summary."""
    best_per_universe: dict[str, dict[str, Any]] = {}
    ranking: list[dict[str, Any]] = []

    for uni_name, uni_data in universe_results.items():
        best = uni_data.get("best")
        if best is None:
            continue
        metrics = best.get("metrics", {})
        entry = {
            "universe": uni_name,
            "universe_code": uni_data["universe_code"],
            "combo_tag": best.get("combo_tag"),
            "status": best.get("status"),
            "sharpe": metrics.get("simple_sharpe"),
            "annual_return": metrics.get("simple_annual_return"),
            "max_drawdown": metrics.get("simple_mdd"),
            "calmar": metrics.get("simple_calmar"),
            "excess_sharpe": metrics.get(
                f"excess_sharpe_{_index_to_alias(uni_data['universe_code'])}"
            ),
            "report_path": best.get("report_path"),
            "result_path": best.get("result_path"),
        }
        best_per_universe[uni_name] = entry
        ranking.append(entry)

    # Rank by calmar (descending).
    ranking.sort(key=lambda x: x.get("calmar") or float("-inf"), reverse=True)

    return {
        "factor_id": factor_id,
        "factor_type": factor_type,
        "top_pct": TOP_PCT,
        "n_universes": len(universe_results),
        "best_per_universe": best_per_universe,
        "ranking": ranking,
        "best_overall": ranking[0] if ranking else None,
        "universes_tested": list(universe_results.keys()),
    }
