#!/usr/bin/env python3
"""Factor admission: promote a factor from the work DB into the stable library.

Pipeline:

  ``backfill``  → write factor values to the **work** DB.
  ``evaluation``→ read work DB, print metrics + reference thresholds (no gating).
  ``run_factor_pipeline.py`` → kick off factor eval + simple + detailed BT.
  ``admit``     → after a human looks at all three reports, run this.
                 Moves the factor's data from work → library, marks
                 ``status='admitted'`` in the registry.
  ``reject``    → opposite: clears the factor's column from work,
                 marks ``status='rejected'``.

Each factor has a single neutralization variant (recorded in the registry's
``variant`` field, see :mod:`backtest.factor.variants`). To compare a factor
under a different variant, re-register it with a new ``factor_id`` or
re-backfill (which overwrites the column).

The work DB is the temporary research playground. The library DB
(``factor_library.duckdb``) holds the stabilised factors and is the only
source consulted by the cross-factor correlation check during evaluation.

All status / history records live inside ``registry.json``.

CLI:
    python -m backtest.factor.admission admit  f_001
    python -m backtest.factor.admission reject f_002
    python -m backtest.factor.admission status                # list all
    python -m backtest.factor.admission status f_001          # show one factor
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from backtest.factor.admission_check import (
    ResidualICIRResult,
    RidgeCheckResult,
    StyleCloneRejectedError,
    TIER_REJECT,
    _get_ridge_thresholds,
    _residuals_to_insert_df,
    residual_icir_check,
    ridge_r2_check,
)
from backtest.factor.registry import (
    _load_registry,
    _save_registry,
    get_registry,
    sync_registry,
)
from backtest.factor.storage import FactorLibrary, FactorStorage
from backtest.factor.variants import CATEGORY_BARRA_L1

# Reference thresholds — read from config.yaml (single source of truth).
# ``evaluation`` may print "passes / does not pass" relative to these to
# help a human decide, but nothing in admission gates against them.
from backtest.config_loader import get_admission_thresholds

_RECOMMENDED = get_admission_thresholds()

RECOMMENDED_THRESHOLDS: dict[str, float | int | str | bool] = {
    "min_rankicir": _RECOMMENDED["min_rankicir"],
    "min_ic_positive_ratio": _RECOMMENDED["min_ic_positive_ratio"],
    "max_turnover": _RECOMMENDED["max_turnover"],
    "max_corr": _RECOMMENDED["max_corr"],
    "primary_horizon": _RECOMMENDED["primary_horizon"],
    "ret_type": _RECOMMENDED["ret_type"],
    "exclude_limit_up": _RECOMMENDED["exclude_limit_up"],
}

STATUS_ADMITTED = "admitted"
STATUS_REJECTED = "rejected"
STATUS_PENDING = "pending"
_VALID_STATUSES = frozenset({STATUS_ADMITTED, STATUS_REJECTED})

Status = Literal["admitted", "rejected"]


@dataclass
class AdmissionAction:
    """Outcome of an ``admit()`` / ``reject()`` call."""

    factor_id: str
    action: Status
    rows_promoted: int
    rows_cleared: int
    timestamp: str

    def __repr__(self) -> str:
        return (
            f"AdmissionAction({self.factor_id}, {self.action.upper()}, "
            f"promoted={self.rows_promoted}, cleared={self.rows_cleared})"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Alpha directory helpers
# ---------------------------------------------------------------------------

_ALPHAS_ROOT = Path(__file__).resolve().parents[2] / "alphas"


def _infer_source(func_module: str) -> str | None:
    """Infer factor source (user/agent) from its registered func_module."""
    if "alphas.exp.user" in func_module:
        return "user"
    if "alphas.exp.agent" in func_module:
        return "agent"
    return None


def _exp_path(factor_id: str, source: str) -> Path:
    """Path to the factor's source file under alphas/exp/."""
    return _ALPHAS_ROOT / "exp" / source / factor_id / "factor.py"


def _admitted_dir(factor_id: str) -> Path:
    """Path to the factor's admitted directory."""
    return _ALPHAS_ROOT / "admitted" / factor_id


def _move_factor_to_admitted(
    factor_id: str,
    func_module: str,
) -> tuple[str | None, Path | None]:
    """Move factor code from alphas/exp/ to alphas/admitted/.

    Returns ``(source, admitted_dir)`` if moved, or ``(None, None)`` if the
    factor's module is not under ``alphas.exp/`` (e.g. built-in Barra).
    """
    source = _infer_source(func_module)
    if source is None:
        return None, None

    src_file = _exp_path(factor_id, source)
    if not src_file.exists():
        return None, None

    admitted = _admitted_dir(factor_id)
    admitted.mkdir(parents=True, exist_ok=True)
    dst_file = admitted / "factor.py"
    shutil.move(str(src_file), str(dst_file))

    # Remove stale .pyc files
    pycache = src_file.parent / "__pycache__"
    if pycache.exists():
        for pyc in pycache.glob(f"{factor_id}.*"):
            pyc.unlink()

    return source, admitted


def _restore_factor_from_admitted(
    factor_id: str,
    func_module: str,
) -> Path | None:
    """Reverse of ``_move_factor_to_admitted`` — used by ``unadmit``.

    Moves ``alphas/admitted/<factor_id>/factor.py`` back to its original
    ``alphas/exp/<source>/<factor_id>/factor.py`` location.
    """
    source = _infer_source(func_module)
    if source is None:
        return None

    admitted_dir = _admitted_dir(factor_id)
    src_file = admitted_dir / "factor.py"
    if not src_file.exists():
        return None

    dst_file = _exp_path(factor_id, source)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_file), str(dst_file))

    # Clean up admitted directory (report/ stays until explicitly removed)
    if admitted_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(str(admitted_dir))

    return dst_file


_AGENT_CANDIDATES_ROOT = "results/agent/candidates"


def _resolve_results_dir(
    factor_id: str,
    *,
    results_root: str | Path = "results",
) -> Path | None:
    """Resolve the results directory for a factor, with agent-candidate fallback.

    Agent-generated factors have their pipeline artifacts under
    ``results/agent/candidates/<factor_id>/`` instead of the usual
    ``results/<factor_id>/``.  This helper returns the first existing path.
    """
    primary = Path(results_root) / factor_id
    if primary.exists():
        return primary
    # Fallback for agent-generated candidates
    fallback = Path(results_root) / "agent" / "candidates" / factor_id
    if fallback.exists():
        return fallback
    return None


def _bundle_backtest_report(
    factor_id: str,
    *,
    tag: str | None = None,
    results_root: str | Path = "results",
) -> dict | None:
    """Copy backtest artifacts into ``alphas/admitted/<factor_id>/report/``.

    Bundles:
      - ``pipeline.json`` (full pipeline config from ``pipeline_state.json``)
      - ``strategy_config.json``
      - ``eval_summary.json`` (factor evaluation)
      - ``pipeline_report.md`` (human-readable report)
      - ``plots/`` (all PNG charts)
      - ``detailed_summary.json`` (detailed backtest summary)

    Returns a dict with ``report_dir`` and list of bundled files, or ``None``
    if no results directory exists.
    """
    factor_results = _resolve_results_dir(factor_id, results_root=results_root)
    if factor_results is None:
        return None

    # Resolve tag directory (standard layout: <fid>/<tag>/pipeline_report.md).
    # When the results directory itself contains pipeline_report.md (flat
    # layout, used by agent candidates), treat the directory as the tag_dir.
    tag_dir: Path | None = None
    if tag is not None:
        candidate = factor_results / tag
        if candidate.exists():
            tag_dir = candidate
    else:
        candidates = sorted(
            p for p in factor_results.iterdir()
            if p.is_dir() and p.name != "factor_eval"
                and (p / "pipeline_report.md").exists()
        )
        if candidates:
            tag_dir = candidates[0]
        elif (factor_results / "pipeline_report.md").exists():
            # Flat directory (e.g. agent candidates) — no tag subdirectory
            tag_dir = factor_results

    report_dir = _admitted_dir(factor_id) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    bundled: list[str] = []

    # 1. Pipeline config (from pipeline_state.json)
    pipeline_state = factor_results / "pipeline_state.json"
    if pipeline_state.exists():
        dst = report_dir / "pipeline.json"
        shutil.copy2(str(pipeline_state), str(dst))
        bundled.append("pipeline.json")

    # 2. Strategy config
    strategy_cfg = factor_results / "strategy_config.json"
    if strategy_cfg.exists():
        dst = report_dir / "strategy_config.json"
        shutil.copy2(str(strategy_cfg), str(dst))
        bundled.append("strategy_config.json")

    # 3. Factor eval summary
    eval_sum = factor_results / "factor_eval" / "eval_summary.json"
    if eval_sum.exists():
        dst = report_dir / "eval_summary.json"
        shutil.copy2(str(eval_sum), str(dst))
        bundled.append("eval_summary.json")

    if tag_dir is None:
        return {
            "report_dir": str(report_dir.relative_to(_ALPHAS_ROOT)),
            "bundled": bundled,
            "tag": None,
        }

    # 4. Pipeline report markdown
    report_md = tag_dir / "pipeline_report.md"
    if report_md.exists():
        dst = report_dir / "pipeline_report.md"
        shutil.copy2(str(report_md), str(dst))
        bundled.append("pipeline_report.md")

    # 5. Plots
    plots_src = tag_dir / "plots"
    if plots_src.exists():
        plots_dst = report_dir / "plots"
        plots_dst.mkdir(parents=True, exist_ok=True)
        for png in sorted(plots_src.glob("*.png")):
            shutil.copy2(str(png), str(plots_dst / png.name))
        bundled.append(f"plots/ ({len(list(plots_dst.glob('*.png')))} files)")

    # 6. Detailed backtest summary
    detailed_summary = tag_dir / "detailed" / "summary.json"
    if detailed_summary.exists():
        dst = report_dir / "detailed_summary.json"
        shutil.copy2(str(detailed_summary), str(dst))
        bundled.append("detailed_summary.json")

    # Tag name: use actual tag subdir name, or "<flat>" for flat layouts
    tag_name = "<flat>" if tag_dir == factor_results else tag_dir.name
    return {
        "report_dir": str(report_dir.relative_to(_ALPHAS_ROOT)),
        "bundled": bundled,
        "tag": tag_name,
    }


# Categories that bootstrap the library — they ARE the regressors used by
# the ridge R² check, so they're admitted before the check exists for anything
# else. Always-skip these from the gate. Only the 7 Barra L1 composites are
# library-resident; the L3 helpers live in code, not in any DB.
_BOOTSTRAP_CATEGORIES: frozenset[str] = frozenset({
    CATEGORY_BARRA_L1,
})


def _finalize_action(
    factor_id: str,
    status: Status,
    *,
    rows_promoted: int,
    rows_cleared: int,
    notes: str | None,
    registry: dict,
    persist: bool,
    strategy_config: dict | None = None,
    ridge_check: RidgeCheckResult | None = None,
    residual_icir_check: ResidualICIRResult | None = None,
    bundle_info: dict | None = None,
    func_module: str | None = None,
    depends_on: list[str] | None = None,
    admission_mode: str = "raw",
) -> AdmissionAction:
    """Stamp the action onto the registry."""
    entry = {
        "action": status,
        "rows_promoted": rows_promoted,
        "rows_cleared": rows_cleared,
        "timestamp": _now_iso(),
        "notes": notes,
    }
    if strategy_config is not None:
        entry["strategy_config"] = strategy_config
    if ridge_check is not None:
        entry["ridge_check"] = ridge_check.as_meta()
    if residual_icir_check is not None:
        entry["residual_icir_check"] = residual_icir_check.as_meta()
    if bundle_info is not None:
        entry["bundle_info"] = bundle_info
    if depends_on is not None:
        entry["depends_on"] = depends_on
    entry["admission_mode"] = admission_mode

    meta = registry[factor_id]
    meta["status"] = status
    meta["admission"] = entry
    if func_module is not None:
        meta["func_module"] = func_module
    if ridge_check is not None:
        meta["tier"] = ridge_check.tier
        meta["r2"] = float(ridge_check.r2)
    if residual_icir_check is not None:
        meta["residual_icir_passed"] = residual_icir_check.passed
        meta["residual_annual_icirs"] = residual_icir_check.annual_icirs
    if depends_on is not None:
        meta["depends_on"] = depends_on
    meta["admission_mode"] = admission_mode
    history = meta.setdefault("admission_history", [])
    history.append(entry)
    del history[:-20]

    if persist:
        _save_registry(registry)
        sync_registry()

    return AdmissionAction(
        factor_id=factor_id,
        action=status,
        rows_promoted=rows_promoted,
        rows_cleared=rows_cleared,
        timestamp=entry["timestamp"],
    )


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------


def admit(
    factor_id: str,
    *,
    notes: str | None = None,
    registry: dict | None = None,
    strategy_config: dict | None = None,
    force: bool = False,
    skip_ridge_check: bool = False,
    skip_residual_icir_check: bool = False,
    tag: str | None = None,
    results_root: str | Path = "results",
) -> AdmissionAction:
    """Promote a factor from the work DB into the stable library.

    Steps:
      1. Read factor column from ``FactorStorage`` (work DB).
      2. Run :func:`ridge_r2_check` against the 6 Barra L1 styles in the
         library (skipped for ``barra_l3`` / ``barra_l1`` categories — they
         ARE the regressors). A ``reject`` tier blocks promotion unless
         ``force=True``; the tier + R² are stamped onto meta either way.
      3. Run :func:`residual_icir_check` against ALL admitted factors.
         The residual RankICIR must be positive (annualised > threshold)
         for at least one horizon, proving the factor adds incremental
         information beyond what the library already captures. Blocked
         unless ``force=True``; skipped for bootstrap categories.
      4. Upsert into ``FactorLibrary`` (library DB).
      5. Drop the column from the work DB.
      6. Move factor code from ``alphas/exp/<source>/`` to
         ``alphas/admitted/<factor_id>/factor.py``.
      7. Bundle backtest artifacts into
         ``alphas/admitted/<factor_id>/report/``.
      8. Mark ``registry[factor_id]["status"] = "admitted"`` and update
         ``func_module`` to point at the new location.

    Raises
    ------
    KeyError
        If ``factor_id`` is not registered.
    ValueError
        If the work DB has no data for this factor, or either gate
        returned a reject verdict and ``force`` is False.
    """
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    meta = target[factor_id]
    category = str(meta.get("category", ""))
    should_check = (
        not skip_ridge_check and category not in _BOOTSTRAP_CATEGORIES
    )
    should_check_residual = (
        not skip_residual_icir_check and category not in _BOOTSTRAP_CATEGORIES
    )

    ridge_result: RidgeCheckResult | None = None
    residual_icir_result: ResidualICIRResult | None = None
    admission_mode: str = "raw"
    depends_on: list[str] = []

    if should_check:
        ridge_result = ridge_r2_check(factor_id)
        ridge_th = _get_ridge_thresholds()
        if ridge_result.r2 >= ridge_th["smart_beta_max"]:
            admission_mode = "residual"
            depends_on = list(ridge_result.regressor_ids)
        if ridge_result.tier == TIER_REJECT and not force:
            raise StyleCloneRejectedError(
                f"{factor_id} blocked by ridge_r2_check: R²={ridge_result.r2:.3f} "
                f"-> tier=reject. Override with force=True if you really want "
                f"this style-clone in the library."
            )

    if should_check_residual:
        precomputed = (
            ridge_result.residuals_df
            if (ridge_result is not None and ridge_result.residuals_df is not None)
            else None
        )
        precomputed_n = (
            len(depends_on) if depends_on
            else (ridge_result.n_regressors if ridge_result is not None else -1)
        )
        residual_icir_result = residual_icir_check(
            factor_id,
            precomputed_residuals=precomputed,
            precomputed_n_regressors=precomputed_n,
        )
        if not residual_icir_result.passed and not force:
            annuals = residual_icir_result.annual_icirs
            raise ValueError(
                f"{factor_id} blocked by residual_icir_check: "
                f"annualised residual RankICIRs={ {h: f'{v:.4f}' for h, v in annuals.items()} }, "
                f"threshold={residual_icir_result.threshold}. "
                f"No horizon adds incremental information beyond the "
                f"{residual_icir_result.n_regressors} existing admitted factors. "
                f"Override with force=True."
            )

    with FactorStorage() as work, FactorLibrary() as lib:
        # Save residuals for library-side UPSERT after promotion.  Writing
        # residuals to the *library* (not work DB) avoids corrupting the
        # work DB if promote_from_work fails partway through.
        residuals_for_lib: pd.DataFrame | None = None
        if (
            admission_mode == "residual"
            and ridge_result is not None
            and ridge_result.residuals_df is not None
        ):
            residuals_for_lib = _residuals_to_insert_df(
                ridge_result.residuals_df, factor_id,
            )

        rows_promoted = lib.promote_from_work(factor_id, work)
        if rows_promoted == 0:
            raise ValueError(
                f"No data in work DB for {factor_id}. "
                f"Did you run `python -m backtest.factor.backfill {factor_id}` first?"
            )
        rows_cleared = work.delete_factor(factor_id)

        if residuals_for_lib is not None:
            # UPSERT residuals over the just-promoted neutralized values.
            # Dates where Ridge skipped (too few observations) keep their
            # original neutralized (barra_ind_size) values.
            # allow_unadmitted=True: the status flip hasn't happened yet
            # (it's done by _finalize_action below), same pattern as
            # promote_from_work.
            lib.insert_factors(residuals_for_lib, allow_unadmitted=True)

    # Move source code to admitted/ and bundle backtest report
    func_module = meta.get("func_module", "")
    source, admitted_dir = _move_factor_to_admitted(factor_id, func_module)
    new_func_module = None
    if admitted_dir is not None:
        new_func_module = f"alphas.admitted.{factor_id}.factor"

    bundle_info = _bundle_backtest_report(
        factor_id, tag=tag, results_root=results_root,
    )

    return _finalize_action(
        factor_id, STATUS_ADMITTED,
        rows_promoted=rows_promoted, rows_cleared=rows_cleared,
        notes=notes, registry=target, persist=persist,
        strategy_config=strategy_config,
        ridge_check=ridge_result,
        residual_icir_check=residual_icir_result,
        bundle_info=bundle_info,
        func_module=new_func_module,
        depends_on=depends_on,
        admission_mode=admission_mode,
    )


def reject(
    factor_id: str,
    *,
    notes: str | None = None,
    registry: dict | None = None,
    strategy_config: dict | None = None,
) -> AdmissionAction:
    """Discard a factor — clear its work-DB column, mark ``status='rejected'``.

    The library DB is **not** touched. If the factor was previously admitted,
    this raises ``ValueError``.
    """
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    if target[factor_id].get("status") == STATUS_ADMITTED:
        raise ValueError(
            f"{factor_id} is already admitted. Use `unadmit` to drop it "
            f"from the library, or `reject` is only for pending factors."
        )

    with FactorStorage() as work:
        rows_cleared = work.delete_factor(factor_id)

    return _finalize_action(
        factor_id, STATUS_REJECTED,
        rows_promoted=0, rows_cleared=rows_cleared,
        notes=notes, registry=target, persist=persist,
        strategy_config=strategy_config,
    )


def mark_rejected(
    factor_id: str,
    *,
    notes: str | None = None,
    registry: dict | None = None,
    strategy_config: dict | None = None,
) -> AdmissionAction:
    """Mark a factor as rejected in the registry WITHOUT clearing the work DB.

    Use this when the pipeline rejects a factor but you want to preserve
    its work-DB data for inspection (e.g., diagnostic reports).
    """
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    return _finalize_action(
        factor_id, STATUS_REJECTED,
        rows_promoted=0, rows_cleared=0,
        notes=notes, registry=target, persist=persist,
        strategy_config=strategy_config,
    )


def unadmit(
    factor_id: str,
    *,
    notes: str | None = None,
    registry: dict | None = None,
    force: bool = False,
) -> AdmissionAction:
    """Drop an admitted factor from the library DB and mark it rejected.

    This is the reverse of :func:`admit`: removes the column from the
    library DuckDB and flips the registry status back to ``rejected``.
    Only works on factors currently marked ``admitted``, unless
    ``force=True``.

    After unadmit, the column is gone from the library but the factor
    must be re-backfilled to the work DB before it can be re-admitted.
    Stale admission metadata (tier, r2, residual_icir_*) is cleared.

    Raises
    ------
    KeyError
        If ``factor_id`` is not registered.
    ValueError
        If the factor is not currently admitted and ``force`` is False.
    """
    persist = registry is None
    target = registry if registry is not None else _load_registry()
    if factor_id not in target:
        raise KeyError(f"factor_id '{factor_id}' not found in registry")

    if not force and target[factor_id].get("status") != STATUS_ADMITTED:
        raise ValueError(
            f"{factor_id} is not admitted (status="
            f"{target[factor_id].get('status', 'unknown')}). "
            f"Only admitted factors can be unadmitted. "
            f"Use force=True to override."
        )

    import duckdb
    from backtest.factor.storage import FACTOR_LIBRARY_DB_PATH, _quote_ident

    qid = _quote_ident(factor_id)
    con = duckdb.connect(str(FACTOR_LIBRARY_DB_PATH))
    try:
        cols = [
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='factors_daily'"
            ).fetchall()
        ]
        if factor_id in cols:
            con.execute(f"ALTER TABLE factors_daily DROP COLUMN {qid}")
            rows_cleared = 1
        else:
            rows_cleared = 0
    finally:
        con.close()

    # Restore factor code from admitted/ back to exp/
    func_module = target[factor_id].get("func_module", "")
    restored = _restore_factor_from_admitted(factor_id, func_module)
    new_func_module = None
    if restored is not None:
        source = _infer_source(func_module)
        if source is not None:
            new_func_module = f"alphas.exp.{source}.{factor_id}"

    # Clear stale admission metadata left from the original admit
    meta = target[factor_id]
    for key in ("tier", "r2", "residual_icir_passed", "residual_annual_icirs",
                 "bundle_info", "depends_on", "admission_mode"):
        meta.pop(key, None)

    return _finalize_action(
        factor_id, STATUS_REJECTED,
        rows_promoted=0, rows_cleared=rows_cleared,
        notes=notes, registry=target, persist=persist,
        func_module=new_func_module,
    )


def _discover_strategy_config(
    factor_id: str,
    *,
    results_root: str | Path = "results",
    tag: str | None = None,
) -> dict | None:
    """Read strategy config from pipeline artifacts.

    Tries in order:
    1. ``results/<factor_id>/<tag>/pipeline.json`` (legacy / agent candidates)
    2. ``results/<factor_id>/<tag>/strategy_config.json`` (new pipeline)
    3. ``results/<factor_id>/strategy_config.json`` (fallback, factor-level)

    Returns ``None`` if nothing is found. Raises ``ValueError`` if multiple
    tag subdirectories exist and ``tag`` is unspecified.
    """
    factor_dir = _resolve_results_dir(factor_id, results_root=results_root)
    if factor_dir is None:
        return None

    def _load_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Legacy pipeline.json has top-level "strategy_config" block.
        cfg = payload.get("strategy_config") if isinstance(payload, dict) else None
        if isinstance(cfg, dict):
            return cfg
        # New strategy_config.json is the config itself.
        return payload if isinstance(payload, dict) else None

    if tag is not None:
        tag_dir = factor_dir / tag
        for name in ("pipeline.json", "strategy_config.json"):
            cfg = _load_json(tag_dir / name)
            if cfg is not None:
                return cfg
        available = sorted(
            p.name for p in factor_dir.iterdir()
            if p.is_dir() and p.name != "factor_eval"
        )
        raise FileNotFoundError(
            f"No strategy config found at {tag_dir}. Available tags: {available}"
        )

    # Auto-discover: collect tag subdirs that have a strategy config file.
    tag_candidates = sorted(
        p for p in factor_dir.iterdir()
        if p.is_dir()
        and p.name != "factor_eval"
        and any((p / name).exists() for name in ("pipeline.json", "strategy_config.json"))
    )
    if len(tag_candidates) > 1:
        names = [p.name for p in tag_candidates]
        raise ValueError(
            f"Found multiple pipeline.json / strategy_config.json under {factor_dir}: {names}. "
            f"Pass --tag <tag> to pick one (or --no-strategy-config to skip)."
        )
    if len(tag_candidates) == 1:
        for name in ("pipeline.json", "strategy_config.json"):
            cfg = _load_json(tag_candidates[0] / name)
            if cfg is not None:
                return cfg

    # Factor-level fallback (new pipeline writes strategy_config.json here).
    cfg = _load_json(factor_dir / "strategy_config.json")
    if cfg is not None:
        return cfg

    return None


def _load_pipeline_strategy_config(path: Path) -> dict | None:
    """Read ``pipeline.json``, return ``strategy_config`` block or None."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload.get("strategy_config")
    if not isinstance(cfg, dict):
        return None
    return cfg


# ---------------------------------------------------------------------------
# Registry queries
# ---------------------------------------------------------------------------


def _factor_ids_where(predicate) -> list[str]:
    return sorted(
        fid for fid, meta in get_registry().items()
        if predicate(meta.get("status"))
    )


def get_admitted_factor_ids() -> list[str]:
    """Factor IDs whose status is admitted."""
    return _factor_ids_where(lambda st: st == STATUS_ADMITTED)


def get_rejected_factor_ids() -> list[str]:
    """Factor IDs whose status is rejected."""
    return _factor_ids_where(lambda st: st == STATUS_REJECTED)


def get_pending_factor_ids() -> list[str]:
    """Factor IDs that have not been admitted or rejected."""
    return _factor_ids_where(lambda st: st not in _VALID_STATUSES)


# ---------------------------------------------------------------------------
# Threshold reference helper (purely informational)
# ---------------------------------------------------------------------------


def check_recommended_thresholds(
    metrics: dict,
    config: dict | None = None,
) -> dict[str, bool]:
    """Compare a factor's metrics against the reference thresholds.

    Returns a dict of ``{check_name: passed_bool}``. **Does not gate
    anything** — purely a convenience for evaluation output and Agent
    decision-making.
    """
    cfg = {**RECOMMENDED_THRESHOLDS, **(config or {})}
    return {
        "rankicir": metrics.get("rankicir", float("-inf")) >= cfg["min_rankicir"],
        "ic_positive_ratio": metrics.get("ic_positive_ratio", 0.0)
            >= cfg["min_ic_positive_ratio"],
        "turnover": metrics.get("turnover", float("inf")) < cfg["max_turnover"],
        "max_corr": metrics.get("max_corr", float("inf")) < cfg["max_corr"],
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_action(action: AdmissionAction) -> None:
    """Pretty-print the outcome of admit() / reject()."""
    print(f"\n{'=' * 60}")
    print(f"Admission: {action.factor_id}  ->  {action.action.upper()}")
    print(f"{'=' * 60}")
    print(f"  rows promoted to library: {action.rows_promoted:,}")
    print(f"  work column cleared    : {action.rows_cleared}")
    print(f"  timestamp              : {action.timestamp}")
    print(f"{'=' * 60}\n")


def print_status(factor_id: str | None = None) -> None:
    """Show admission status for one factor or all factors."""
    registry = get_registry()
    if factor_id is not None:
        if factor_id not in registry:
            print(f"factor_id '{factor_id}' not found")
            return
        rows = [(factor_id, registry[factor_id])]
    else:
        rows = sorted(registry.items())

    if not rows:
        print("(no factors registered)")
        return

    print(f"\n{'factor_id':<14}{'variant':<18}{'freq':<6}{'status':<12}{'name'}")
    print("-" * 78)
    for fid, meta in rows:
        st = meta.get("status", STATUS_PENDING)
        variant = meta.get("variant", "-")
        freq = meta.get("frequency", "-")
        print(f"{fid:<14}{variant:<18}{freq:<6}{st:<12}{meta.get('name', '')}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Factor admission")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_meta_flags(sp):
        sp.add_argument("--tag", default=None,
                        help="results/<fid>/<tag>/ subdirectory. When set, "
                             "admit/reject reads pipeline.json's strategy_config "
                             "block into history.")
        sp.add_argument("--results-root", default="results",
                        help="results root directory, default 'results'.")
        sp.add_argument("--no-strategy-config", action="store_true",
                        help="Skip reading pipeline.json; history won't carry "
                             "strategy_config block.")

    p_admit = sub.add_parser("admit", help="Promote factor from work DB to library DB")
    p_admit.add_argument("factor_id")
    p_admit.add_argument("--notes", default=None, help="Free-form note for history")
    p_admit.add_argument("--force", action="store_true",
                         help="Bypass the ridge R² reject gate. Use only when "
                              "you really want a style-clone (R²>=0.80) in the library.")
    p_admit.add_argument("--skip-ridge-check", action="store_true",
                         help="Don't run the ridge check at all (e.g. when "
                              "library doesn't yet contain the 6 Barra L1).")
    p_admit.add_argument("--skip-residual-icir-check", action="store_true",
                         help="Skip the residual ICIR incremental-info check.")
    _add_meta_flags(p_admit)

    p_reject = sub.add_parser("reject", help="Discard factor — clear work DB, mark rejected")
    p_reject.add_argument("factor_id")
    p_reject.add_argument("--notes", default=None, help="Free-form note for history")
    _add_meta_flags(p_reject)

    p_unadmit = sub.add_parser("unadmit", help="Drop an admitted factor from the library DB")
    p_unadmit.add_argument("factor_id")
    p_unadmit.add_argument("--notes", default=None, help="Free-form note for history")
    p_unadmit.add_argument("--no-strategy-config", action="store_true",
                           help="Skip reading pipeline.json (unadmit doesn't need it)")
    p_unadmit.add_argument("--force", action="store_true",
                           help="Force unadmit even if status is not 'admitted'")

    p_status = sub.add_parser("status", help="Show admission status")
    p_status.add_argument("factor_id", nargs="?", default=None)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "unadmit":
        try:
            action = unadmit(
                args.factor_id, notes=args.notes,
                force=getattr(args, "force", False),
            )
        except (KeyError, ValueError) as exc:
            parser.exit(2, f"unadmit failed: {exc}\n")
        print_action(action)
        print()
        return

    if args.cmd in ("admit", "reject"):
        strategy_config: dict | None = None
        if not args.no_strategy_config:
            try:
                strategy_config = _discover_strategy_config(
                    args.factor_id,
                    results_root=args.results_root, tag=args.tag,
                )
            except (ValueError, FileNotFoundError) as exc:
                parser.exit(2, f"{args.cmd} failed: {exc}\n")
        try:
            if args.cmd == "admit":
                action = admit(
                    args.factor_id, notes=args.notes,
                    strategy_config=strategy_config,
                    force=args.force,
                    skip_ridge_check=args.skip_ridge_check,
                    skip_residual_icir_check=args.skip_residual_icir_check,
                    tag=args.tag,
                    results_root=args.results_root,
                )
            else:
                action = reject(
                    args.factor_id, notes=args.notes,
                    strategy_config=strategy_config,
                )
        except StyleCloneRejectedError as exc:
            parser.exit(3, f"{args.cmd} rejected by gate: {exc}\n")
        except (KeyError, ValueError) as exc:
            parser.exit(2, f"{args.cmd} failed: {exc}\n")
        print_action(action)
        if strategy_config:
            print(f"  strategy_config recorded ({len(strategy_config)} fields)")
        if args.cmd == "admit":
            entry = get_registry()[args.factor_id].get("admission", {})
            rc = entry.get("ridge_check")
            if rc is not None:
                print(f"  ridge check          : R²={rc['r2']:.3f} "
                      f"tier={rc['tier']} n_obs={rc['n_obs']:,}")
            ric = entry.get("residual_icir_check")
            if ric is not None:
                annuals = ric.get("annual_icirs", {})
                passed = "PASS" if ric.get("passed") else "FAIL"
                print(f"  residual ICIR check  : {passed} "
                      f"annual_icirs={ {h: f'{v:.3f}' for h, v in annuals.items()} }, "
                      f"n_regressors={ric.get('n_regressors', 0)}, "
                      f"threshold={ric.get('threshold', 0)}")
            bundle = entry.get("bundle_info")
            if bundle is not None:
                print(f"  code moved to        : alphas/admitted/{args.factor_id}/factor.py")
                print(f"  report bundled to    : {bundle.get('report_dir', 'N/A')}")
                for item in bundle.get("bundled", []):
                    print(f"    - {item}")
        print()
        return

    if args.cmd == "status":
        print_status(args.factor_id)
        return


if __name__ == "__main__":
    main()
