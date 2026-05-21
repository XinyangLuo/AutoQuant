"""Pipeline step functions (step1~step9).

Each function is a pure transform: takes PipelineState, returns updated
PipelineState.  They are called by the CLI dispatcher in __main__.py.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.admission import admit
from backtest.factor.admission_check import TIER_REJECT, ridge_r2_check
from backtest.factor.evaluation import _corr_with_existing, evaluate
from backtest.factor.registry import get_factor_meta
from backtest.factor.storage import FactorLibrary, FactorStorage
from backtest.simulation.config import SimulationConfig
from backtest.simulation.detailed import DetailedSimulator
from backtest.simulation.models import BacktestResult
from backtest.simulation.simple import SimpleSimulator
from backtest.strategy.config import (
    BacktestConfig,
    FactorConfig,
    SelectionConfig,
    StrategyConfig,
    UniverseConfig,
    WeightingConfig,
)
from backtest.strategy.strategies.single_factor import SingleFactorStrategy

from .config import PipelineConfig
from .state import PipelineState, StepResult

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _reject(state: PipelineState, step: str, reason: str, metrics: dict | None = None) -> PipelineState:
    state.record(
        step,
        StepResult(
            passed=False,
            reason=reason,
            metrics=metrics or {},
        ),
    )
    return state


def _pass(state: PipelineState, step: str, metrics: dict | None = None) -> PipelineState:
    state.record(
        step,
        StepResult(
            passed=True,
            metrics=metrics or {},
        ),
    )
    return state


# ---------------------------------------------------------------------------
# Step 1: Coverage check
# ---------------------------------------------------------------------------


def step1_coverage_check(state: PipelineState) -> PipelineState:
    """Check cross-sectional missing rate.

    Price/volume factors: max missing rate < 10%.
    Financial factors: max missing rate < 30%.
    """
    config = state.config

    meta = get_factor_meta(config.factor_id)
    data_sources = meta.get("data_sources", [])
    is_financial = any(
        src in ("income_q", "balancesheet_q", "cashflow_q")
        for src in data_sources
    )
    threshold = (
        config.thresholds.max_missing_rate_fin
        if is_financial
        else config.thresholds.max_missing_rate_pv
    )

    with FactorStorage() as fs:
        factor_df = fs.get_factor(config.factor_id, config.start_date, config.end_date)

    if factor_df.empty:
        return _reject(
            state, "step1",
            "No factor data in work DB. Run backfill first.",
        )

    # Compute per-date missing rate against market universe
    missing_rates: list[float] = []
    dates = factor_df["date"].unique()
    with MarketStorage() as ms:
        for d in dates:
            d_str = pd.Timestamp(d).strftime("%Y%m%d")
            try:
                panel = ms.get_panel(d_str)
                universe_n = len(panel)
            except Exception:
                universe_n = 0
            if universe_n == 0:
                continue
            non_null = factor_df[factor_df["date"] == d]["value"].notna().sum()
            missing_rates.append(1.0 - non_null / universe_n)

    if not missing_rates:
        return _reject(state, "step1", "Could not compute missing rates.")

    max_missing = max(missing_rates)
    mean_missing = sum(missing_rates) / len(missing_rates)

    metrics = {
        "max_missing_rate": float(max_missing),
        "mean_missing_rate": float(mean_missing),
        "threshold": float(threshold),
        "is_financial": is_financial,
        "n_dates": len(dates),
    }

    if max_missing > threshold:
        return _reject(
            state, "step1",
            f"Missing rate {max_missing:.1%} exceeds threshold {threshold:.1%}",
            metrics,
        )

    return _pass(state, "step1", metrics)


# ---------------------------------------------------------------------------
# Step 2: Neutralization verification
# ---------------------------------------------------------------------------


def step2_neutralization_check(state: PipelineState) -> PipelineState:
    """Verify barra_ind_size neutralization succeeded.

    Checks:
    1. corr with size_z < 0.05
    2. corr with all industry dummies < 0.05
    3. max corr with existing library factors < 0.5
    """
    config = state.config

    with FactorStorage() as fs:
        factor_df = fs.get_factor(config.factor_id, config.start_date, config.end_date)

    if factor_df.empty:
        return _reject(state, "step2", "No factor data available.")

    start = factor_df["date"].min().strftime("%Y%m%d")
    end = factor_df["date"].max().strftime("%Y%m%d")

    # 1. Correlation with size_z
    from backtest.factor.builtin.barra.size import SIZE_LNCAP_ID

    size_corr = 0.0
    with FactorStorage() as fs:
        size_df = fs.get_factor(SIZE_LNCAP_ID, start=start, end=end)
    if not size_df.empty:
        merged = factor_df.merge(
            size_df.rename(columns={"value": "size_z"}),
            on=["date", "symbol"],
            how="inner",
        )
        if not merged.empty:
            daily_corr = merged.groupby("date").apply(
                lambda g: _pearson(g["value"], g["size_z"]),
                include_groups=False,
            )
            size_corr = float(daily_corr.abs().mean())

    # 2. Correlation with industry dummies
    max_ind_corr = 0.0
    with MarketStorage() as ms:
        industry = ms.get_industry_panel_range(start=start, end=end, level="L1")
    if not industry.empty:
        merged = factor_df.merge(industry, on=["date", "symbol"], how="inner")
        if not merged.empty:
            # Build dummies per date to avoid huge memory
            max_ind_corr = _max_industry_corr(merged)

    # 3. Max correlation with existing library factors
    max_existing_corr = 0.0
    max_existing_factor: str | None = None
    with FactorLibrary() as lib:
        existing_ids = lib.get_existing_factor_ids()
        if existing_ids and config.factor_id not in existing_ids:
            corr_df = _corr_with_existing(
                factor_df, config.factor_id, lib, top_k=1,
            )
            if not corr_df.empty:
                max_existing_corr = float(abs(corr_df.iloc[0]["corr"]))
                max_existing_factor = str(corr_df.iloc[0]["factor_id"])

    passed = (
        size_corr < config.max_corr_size
        and max_ind_corr < config.max_corr_industry
        and max_existing_corr < config.max_corr_existing
    )

    metrics = {
        "size_corr": float(size_corr),
        "max_industry_corr": float(max_ind_corr),
        "max_existing_corr": float(max_existing_corr),
        "max_existing_factor": max_existing_factor,
    }

    if not passed:
        violations = []
        if size_corr >= config.max_corr_size:
            violations.append(f"size_corr={size_corr:.3f} >= {config.max_corr_size}")
        if max_ind_corr >= config.max_corr_industry:
            violations.append(f"ind_corr={max_ind_corr:.3f} >= {config.max_corr_industry}")
        if max_existing_corr >= config.max_corr_existing:
            violations.append(
                f"existing_corr={max_existing_corr:.3f} (with {max_existing_factor}) "
                f">= {config.max_corr_existing}"
            )
        return _reject(state, "step2", "; ".join(violations), metrics)

    return _pass(state, "step2", metrics)


def _pearson(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation, NaN-aware."""
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask].values, b[mask].values)[0, 1])


def _max_industry_corr(merged: pd.DataFrame) -> float:
    """Max abs Pearson corr between factor value and any industry dummy."""
    max_corr = 0.0
    for date, group in merged.groupby("date"):
        dummies = pd.get_dummies(group["industry_code"], prefix="ind")
        for col in dummies.columns:
            corr = _pearson(group["value"], dummies[col])
            max_corr = max(max_corr, abs(corr))
    return max_corr


# ---------------------------------------------------------------------------
# Step 3: ICIR gate
# ---------------------------------------------------------------------------


def step3_icir_check(state: PipelineState) -> PipelineState:
    """ICIR gate with frequency-aware thresholds.

    Daily: check 1D and 5D horizons, EITHER passes.
    Monthly: check 20D horizon.
    """
    config = state.config

    eval_result = evaluate(
        config.factor_id,
        config.start_date,
        config.end_date,
        horizons=config.eval_horizons,
        ret_type=config.ret_type,
        corr_top_k=0,
        exclude_limit_up=True,
        run_decile_backtest=False,
    )

    th = config.thresholds
    check_horizons = config.icir_check_horizons

    passed_any = False
    best_horizon: int | None = None
    best_metrics: dict = {}

    for h in check_horizons:
        ic = eval_result.ic_metrics.get(h, {})
        if not ic:
            continue

        raw_icir = ic.get("icir", float("-inf"))
        annual_icir = (
            raw_icir * math.sqrt(252 / h)
            if raw_icir and not (isinstance(raw_icir, float) and math.isnan(raw_icir))
            else float("-inf")
        )
        abs_ic = abs(ic.get("ic_mean", 0) or 0)
        tstat = ic.get("ic_tstat", float("-inf"))
        pos_ratio = ic.get("ic_positive_ratio", 0.0)

        checks = {
            "abs_ic": abs_ic > th.min_abs_ic,
            "annual_icir": annual_icir > th.min_annual_icir,
            "tstat": tstat > th.min_ic_tstat,
            "pos_ratio": pos_ratio > th.min_ic_positive_ratio,
        }

        if all(checks.values()):
            passed_any = True
            best_horizon = h
            best_metrics = {
                "horizon": h,
                "abs_ic": float(abs_ic),
                "annual_icir": float(annual_icir),
                "tstat": float(tstat),
                "pos_ratio": float(pos_ratio),
                "raw_icir": float(raw_icir) if raw_icir else None,
            }
            break

    # If none passed, record the first checked horizon for diagnostics
    if not passed_any and check_horizons:
        h = check_horizons[0]
        ic = eval_result.ic_metrics.get(h, {})
        raw_icir = ic.get("icir", float("-inf"))
        annual_icir = (
            raw_icir * math.sqrt(252 / h)
            if raw_icir and not (isinstance(raw_icir, float) and math.isnan(raw_icir))
            else float("-inf")
        )
        best_metrics = {
            "horizon": h,
            "abs_ic": float(abs(ic.get("ic_mean", 0) or 0)),
            "annual_icir": float(annual_icir),
            "tstat": float(ic.get("ic_tstat", float("-inf"))),
            "pos_ratio": float(ic.get("ic_positive_ratio", 0.0)),
        }

    all_ic = {
        h: eval_result.ic_metrics.get(h, {})
        for h in config.eval_horizons
    }

    metrics = {
        "best_horizon": best_horizon,
        "checked_horizons": check_horizons,
        **best_metrics,
        "all_ic_metrics": all_ic,
    }

    # Persist eval result artifact
    eval_dir = Path(config.results_root) / config.factor_id / "factor_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "eval_summary.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "factor_id": config.factor_id,
                "metrics_by_horizon": eval_result.summary().to_dict(orient="records"),
                "ic_metrics": {str(h): v for h, v in eval_result.ic_metrics.items()},
                "rank_ic_metrics": {str(h): v for h, v in eval_result.rank_ic_metrics.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    state.artifacts["eval_result"] = str(eval_path)

    if passed_any:
        return _pass(state, "step3", metrics)

    reason_parts = []
    m = best_metrics
    if m.get("abs_ic", 0) <= th.min_abs_ic:
        reason_parts.append(f"|IC|={m.get('abs_ic', 0):.4f} <= {th.min_abs_ic}")
    if m.get("annual_icir", float("-inf")) <= th.min_annual_icir:
        reason_parts.append(f"ICIR={m.get('annual_icir', 0):.3f} <= {th.min_annual_icir}")
    if m.get("tstat", float("-inf")) <= th.min_ic_tstat:
        reason_parts.append(f"t={m.get('tstat', 0):.2f} <= {th.min_ic_tstat}")
    if m.get("pos_ratio", 0) <= th.min_ic_positive_ratio:
        reason_parts.append(f"pos_ratio={m.get('pos_ratio', 0):.1%} <= {th.min_ic_positive_ratio}")
    reason = f"horizon={m.get('horizon', 'N/A')}: " + "; ".join(reason_parts)
    return _reject(state, "step3", reason, metrics)


# ---------------------------------------------------------------------------
# Step 4: Monotonicity check
# ---------------------------------------------------------------------------


def step4_monotonicity_check(state: PipelineState) -> PipelineState:
    """10-group quantile, Spearman corr(group_id, mean_return) > 0.7."""
    config = state.config

    # Re-run evaluate to get group_returns (or load from step3 artifact)
    eval_result = evaluate(
        config.factor_id,
        config.start_date,
        config.end_date,
        horizons=config.eval_horizons,
        ret_type=config.ret_type,
        corr_top_k=0,
        exclude_limit_up=True,
        run_decile_backtest=False,
    )

    primary_h = config.icir_check_horizons[0]
    group_rets = eval_result.group_returns.get(primary_h)

    if group_rets is None or group_rets.empty:
        return _reject(
            state, "step4",
            f"No group returns for horizon {primary_h}",
        )

    groups = group_rets["group"].values.astype(float)
    mean_rets = group_rets["mean_ret"].values.astype(float)

    if len(groups) < 3:
        return _reject(
            state, "step4",
            f"Only {len(groups)} groups available (need >= 3)",
        )

    spearman = _spearman_corr(groups, mean_rets)
    passed = spearman > config.thresholds.min_monotonicity

    metrics = {
        "spearman": float(spearman),
        "threshold": float(config.thresholds.min_monotonicity),
        "n_groups": len(groups),
        "group_mean_returns": {
            int(g): float(r) for g, r in zip(groups, mean_rets)
        },
    }

    if passed:
        return _pass(state, "step4", metrics)

    return _reject(
        state, "step4",
        f"Spearman={spearman:.3f} <= {config.thresholds.min_monotonicity}",
        metrics,
    )


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation — pure numpy, no scipy."""
    x_rank = pd.Series(x).rank().values
    y_rank = pd.Series(y).rank().values
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = math.sqrt((x_rank**2).sum() * (y_rank**2).sum())
    if denom == 0:
        return float("nan")
    return float((x_rank * y_rank).sum() / denom)


# ---------------------------------------------------------------------------
# Step 5: Strategy config
# ---------------------------------------------------------------------------


def step5_build_strategy(
    state: PipelineState,
    top_pct: float | None = None,
    decay: int | None = None,
    universe: str | None = None,
    rebalance: str | None = None,
) -> PipelineState:
    """Build default strategy configuration.

    Params are taken from (in order of priority):
    1. CLI kwargs passed to this function
    2. state.retry_params from a previous failed attempt
    3. PipelineConfig defaults
    """
    config = state.config

    # Resolve params with priority: CLI > retry_params > defaults
    rp = state.retry_params
    _top_pct = top_pct if top_pct is not None else rp.get("top_pct", config.default_top_pct)
    _decay = decay if decay is not None else rp.get("decay", config.default_decay)
    _rebalance = rebalance if rebalance is not None else rp.get("rebalance", config.default_rebalance)
    _universe = universe if universe is not None else rp.get("universe", config.default_universe)

    strategy_config = StrategyConfig(
        name=f"{config.factor_id}_pipeline",
        strategy_type="single_factor_topk",
        rebalance_freq=_rebalance,
        delay=1,
        universe=UniverseConfig(
            exclude_st=True,
            exclude_new_ipo_days=252,
            include_kcb=False,
            index_members=_universe,
        ),
        factors=[FactorConfig(id=config.factor_id, direction="desc")],
        selection=SelectionConfig(method="topk", top_pct=_top_pct),
        weighting=WeightingConfig(method="equal"),
        decay=_decay if _decay > 0 else None,
        backtest=BacktestConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            benchmark=config.benchmark,
        ),
    )

    state.strategy_config = strategy_config

    # Persist strategy config artifact
    cfg_dir = Path(config.results_root) / config.factor_id
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "strategy_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": strategy_config.name,
                "rebalance_freq": strategy_config.rebalance_freq,
                "top_pct": _top_pct,
                "decay": _decay,
                "universe": _universe,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    state.artifacts["strategy_config"] = str(cfg_path)

    return _pass(state, "step5", {
        "top_pct": _top_pct,
        "decay": _decay,
        "rebalance": _rebalance,
        "universe": _universe,
    })


# ---------------------------------------------------------------------------
# Step 6: Simple backtest gate
# ---------------------------------------------------------------------------


_MARKET_BUFFER_DAYS = 10


def step6_simple_backtest(state: PipelineState) -> PipelineState:
    """Vectorised simple backtest with threshold gates."""
    config = state.config

    if state.strategy_config is None:
        return _reject(state, "step6", "No strategy config. Run step5 first.")

    strategy = SingleFactorStrategy(state.strategy_config)
    signals = strategy.run(config.start_date, config.end_date)
    state.signals = signals

    if signals.empty:
        return _reject(state, "step6", "Strategy produced no signals.")

    # Load market data
    market_end = (
        pd.to_datetime(config.end_date)
        + pd.Timedelta(days=_MARKET_BUFFER_DAYS)
    ).strftime("%Y%m%d")
    symbols = signals["symbol"].unique().tolist()
    with MarketStorage() as ms:
        market_data = ms.get_bars(
            symbols=symbols, start=config.start_date, end=market_end,
        )

    sim = SimpleSimulator(SimulationConfig(initial_cash=1e8))
    result = sim.run(signals, market_data)

    # Persist
    tag = _build_tag(state)
    out_dir = Path(config.results_root) / config.factor_id / tag / "simple"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save(str(out_dir), metadata={
        "strategy": {"name": state.strategy_config.name, "factor": config.factor_id},
        "simulation": {"engine": "SimpleSimulator", "initial_cash": 1e8},
    })
    state.artifacts["simple_bt"] = str(out_dir)

    # Metrics + threshold check
    metrics = result.summary()
    state.simple_bt_metrics = metrics

    th = config.thresholds
    checks = {
        "sharpe": (metrics.get("sharpe") or float("-inf")) > th.min_sharpe_simple,
        "annual_return": (metrics.get("annual_return") or float("-inf")) > th.min_annual_return_simple,
        "max_drawdown": (metrics.get("max_drawdown") or float("-inf")) > -th.max_max_drawdown,
        "calmar": (metrics.get("calmar") or float("-inf")) > th.min_calmar_simple,
        "annual_turnover": (metrics.get("annual_turnover") or float("inf")) < th.max_annual_turnover,
    }
    passed = all(checks.values())

    if passed:
        return _pass(state, "step6", metrics)

    violations = _bt_violations(metrics, checks, th, "simple")
    return _reject(state, "step6", "; ".join(violations), metrics)


# ---------------------------------------------------------------------------
# Step 7: Detailed backtest gate
# ---------------------------------------------------------------------------


def step7_detailed_backtest(state: PipelineState) -> PipelineState:
    """Event-driven detailed backtest with threshold gates."""
    config = state.config

    if state.strategy_config is None or state.signals is None:
        return _reject(state, "step7", "No strategy/signals. Run step5-6 first.")

    market_end = (
        pd.to_datetime(config.end_date)
        + pd.Timedelta(days=_MARKET_BUFFER_DAYS)
    ).strftime("%Y%m%d")
    symbols = state.signals["symbol"].unique().tolist()

    with MarketStorage() as ms:
        market_data = ms.get_bars(
            symbols=symbols, start=config.start_date, end=market_end,
        )
        dividends = ms.get_dividends(
            symbols=symbols, start=config.start_date, end=market_end,
        )

    sim = DetailedSimulator(SimulationConfig(
        initial_cash=1e8,
        commission_rate=0.0003,
        price_type="o2o",
        allow_short=False,
    ))
    result = sim.run(state.signals, market_data, dividends)

    # Persist
    tag = _build_tag(state)
    out_dir = Path(config.results_root) / config.factor_id / tag / "detailed"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save(str(out_dir), metadata={
        "strategy": {"name": state.strategy_config.name, "factor": config.factor_id},
        "simulation": {"engine": "DetailedSimulator", "initial_cash": 1e8},
    })
    state.artifacts["detailed_bt"] = str(out_dir)

    # Metrics + threshold check
    metrics = result.summary()
    state.detailed_bt_metrics = metrics

    th = config.thresholds
    checks = {
        "sharpe": (metrics.get("sharpe") or float("-inf")) > th.min_sharpe_detailed,
        "annual_return": (metrics.get("annual_return") or float("-inf")) > th.min_annual_return_detailed,
        "max_drawdown": (metrics.get("max_drawdown") or float("-inf")) > -th.max_max_drawdown,
        "calmar": (metrics.get("calmar") or float("-inf")) > th.min_calmar_detailed,
        "annual_turnover": (metrics.get("annual_turnover") or float("inf")) < th.max_annual_turnover,
    }
    passed = all(checks.values())

    if passed:
        return _pass(state, "step7", metrics)

    violations = _bt_violations(metrics, checks, th, "detailed")
    return _reject(state, "step7", "; ".join(violations), metrics)


# ---------------------------------------------------------------------------
# Shared backtest helpers
# ---------------------------------------------------------------------------


def _build_tag(state: PipelineState) -> str:
    """Mirror run_factor_pipeline.py tag format."""
    cfg = state.strategy_config
    if cfg is None:
        return "default"
    sel = cfg.selection
    if sel.top_pct is not None:
        tag = f"top{int(round(sel.top_pct * 100))}pct"
    elif sel.top_k is not None:
        tag = f"top{sel.top_k}"
    else:
        tag = "top10pct"
    decay = cfg.decay or 0
    return f"{tag}_{cfg.rebalance_freq.lower()}_d{decay}"


def _bt_violations(
    metrics: dict,
    checks: dict[str, bool],
    th,
    prefix: str,
) -> list[str]:
    """Build human-readable violation strings."""
    violations = []
    if not checks.get("sharpe"):
        violations.append(f"Sharpe={metrics.get('sharpe', 0):.3f} <= threshold")
    if not checks.get("annual_return"):
        violations.append(f"ann_ret={metrics.get('annual_return', 0):.2%} <= threshold")
    if not checks.get("max_drawdown"):
        violations.append(f"max_dd={metrics.get('max_drawdown', 0):.2%} <= threshold")
    if not checks.get("calmar"):
        violations.append(f"Calmar={metrics.get('calmar', 0):.3f} <= threshold")
    if not checks.get("annual_turnover"):
        violations.append(f"turnover={metrics.get('annual_turnover', 0):.1f}x >= threshold")
    return violations


# ---------------------------------------------------------------------------
# Step 8: Ridge R2 classification
# ---------------------------------------------------------------------------


def step8_ridge_r2(state: PipelineState) -> PipelineState:
    """Ridge R2 classification against Barra L1 factors."""
    config = state.config

    try:
        ridge_result = ridge_r2_check(config.factor_id)
    except Exception as exc:
        return _reject(state, "step8", f"Ridge check failed: {exc}")

    state.ridge_result = ridge_result

    passed = ridge_result.tier != TIER_REJECT

    metrics = {
        "r2": float(ridge_result.r2),
        "tier": ridge_result.tier,
        "residual_icir": ridge_result.residual_icir,
        "n_obs": ridge_result.n_obs,
    }

    if passed:
        return _pass(state, "step8", metrics)

    return _reject(
        state, "step8",
        f"R2={ridge_result.r2:.3f} >= 0.80, tier=reject (style clone)",
        metrics,
    )


# ---------------------------------------------------------------------------
# Step 9: Report + admission
# ---------------------------------------------------------------------------


def step9_report_and_admit(state: PipelineState) -> PipelineState:
    """Generate markdown report and call admit()."""
    config = state.config

    # Generate report
    from backtest.factor.pipeline._report import generate_pipeline_report

    report_path = generate_pipeline_report(state)
    state.artifacts["report"] = str(report_path)

    # Build admission metadata
    meta = {
        "pipeline_version": "1.0",
        "step_results": {
            step: {
                "passed": r.passed,
                "metrics": r.metrics,
            }
            for step, r in state.step_results.items()
        },
        "strategy_config": {
            "top_pct": config.default_top_pct,
            "decay": config.default_decay,
            "rebalance": config.default_rebalance,
        },
        "ridge": {
            "r2": state.ridge_result.r2 if state.ridge_result else None,
            "tier": state.ridge_result.tier if state.ridge_result else None,
        },
    }

    try:
        action = admit(
            config.factor_id,
            notes=f"Auto-admitted via pipeline. Report: {report_path}",
            strategy_config=meta,
        )
        state.status = "admitted"
        return _pass(state, "step9", {
            "rows_promoted": action.rows_promoted,
            "report_path": str(report_path),
        })
    except Exception as exc:
        return _reject(state, "step9", f"Admission failed: {exc}")
