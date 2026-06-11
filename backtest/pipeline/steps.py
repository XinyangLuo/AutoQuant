"""Pipeline step functions (step1~step10).

Each function is a pure transform: takes PipelineState, returns updated
PipelineState.  They are called by the CLI dispatcher in __main__.py.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.admission import admit
from backtest.factor.admission_check import (
    CandidateNotBackfilledError,
    InsufficientOverlapError,
    TIER_REJECT,
    residual_icir_check,
    ridge_r2_check,
)
from backtest.config_loader import get_section
from backtest.factor.evaluation import (
    _corr_with_existing,
    _ic_series,
    _rank_ic_series,
    evaluate,
)
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

FULL_MARKET_UNIVERSE = "__full_market__"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_universe_override(universe: str | None, default_universe: str | None) -> str | None:
    """Resolve CLI/sweep universe values.

    ``None`` preserves legacy behavior: use the configured default universe.
    ``FULL_MARKET_UNIVERSE`` is an explicit request for the full-market
    universe, bypassing ``PipelineConfig.default_universe``.
    """
    if universe == FULL_MARKET_UNIVERSE:
        return None
    return universe if universe is not None else default_universe


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


def _build_universe(
    index_members: str | None = None,
    overrides: dict | None = None,
) -> UniverseConfig:
    """Build UniverseConfig with hardcoded defaults + optional per-factor overrides.

    When *index_members* is set (index universe mode), board and market-cap
    filters are disabled — the index definition already determines the
    investable set. ST and new-IPO filters remain active.
    """
    is_index_universe = index_members is not None
    cfg = {
        "exclude_st": True,
        "exclude_new_ipo_days": 252,
        "include_cyb": True,
        "include_kcb": False if not is_index_universe else True,
        "include_bse": False if not is_index_universe else True,
        "min_market_cap": 500_000_000 if not is_index_universe else None,
        "min_avg_amount": 10_000_000,
    }
    if overrides and "universe" in overrides:
        cfg.update(overrides["universe"])

    return UniverseConfig(
        exclude_st=cfg["exclude_st"],
        exclude_new_ipo_days=cfg["exclude_new_ipo_days"],
        include_cyb=cfg["include_cyb"],
        include_kcb=cfg["include_kcb"],
        include_bse=cfg["include_bse"],
        index_members=index_members,
        min_market_cap=cfg["min_market_cap"],
        min_avg_amount=cfg["min_avg_amount"],
    )


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

    with FactorStorage(read_only=True) as fs:
        factor_df = fs.get_factor(config.factor_id, config.start_date, config.end_date)

    if factor_df.empty:
        return _reject(
            state, "step1",
            "No factor data in work DB. Run backfill first.",
        )

    # Compute per-date missing rate against market universe (batched)
    with MarketStorage(read_only=True) as ms:
        market_df = ms.get_bars(
            start=config.start_date,
            end=config.end_date,
            columns=["symbol"],
        )
    if market_df.empty:
        return _reject(state, "step1", "No market data for coverage check.")

    universe_counts = market_df.groupby("date").size()
    factor_counts = factor_df.groupby("date")["value"].agg(["count", "size"])
    # Align: factor_counts may have fewer dates than universe_counts
    aligned = factor_counts.join(universe_counts.rename("universe"), how="inner")
    if aligned.empty:
        return _reject(state, "step1", "No overlapping dates between factor and market data.")

    missing_rates = 1.0 - aligned["count"] / aligned["universe"]

    max_missing = max(missing_rates)
    mean_missing = sum(missing_rates) / len(missing_rates)
    # 95th percentile is robust to single-day IPO-wave spikes while still
    # catching systematic coverage problems.
    pct95 = float(np.percentile(missing_rates, 95))

    metrics = {
        "max_missing_rate": float(max_missing),
        "mean_missing_rate": float(mean_missing),
        "pct95_missing_rate": pct95,
        "threshold": float(threshold),
        "is_financial": is_financial,
        "n_dates": factor_df["date"].nunique(),
    }

    if pct95 > threshold:
        return _reject(
            state, "step1",
            f"95th percentile missing rate {pct95:.1%} exceeds threshold {threshold:.1%} "
            f"(max={max_missing:.1%}, mean={mean_missing:.1%})",
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

    Existing-factor correlation is deferred to step8 (ridge R2) — step2
    only verifies that the neutralization pipeline itself worked.
    """
    config = state.config

    with FactorStorage(read_only=True) as fs:
        factor_df = fs.get_factor(config.factor_id, config.start_date, config.end_date)

    if factor_df.empty:
        return _reject(state, "step2", "No factor data available.")

    start = factor_df["date"].min().strftime("%Y%m%d")
    end = factor_df["date"].max().strftime("%Y%m%d")

    # 1. Correlation with size_z
    from backtest.factor.storage import FactorLibrary
    from backtest.factor.variants import SIZE_L1_ID

    size_corr = 0.0
    with FactorLibrary() as lib:
        size_df = lib.get_factor(SIZE_L1_ID, start=start, end=end)
    if not size_df.empty:
        merged = factor_df.merge(
            size_df.rename(columns={"value": "size_z"}),
            on=["date", "symbol"],
            how="inner",
        )
        if not merged.empty:
            daily_corr = merged.groupby("date").apply(
                lambda g: _ic_series(g["value"], g["size_z"]),
                include_groups=False,
            )
            size_corr = float(daily_corr.abs().mean())

    # 2. Correlation with industry dummies
    max_ind_corr = 0.0
    with MarketStorage(read_only=True) as ms:
        industry = ms.get_industry_panel_range(start=start, end=end, level="L1")
    if not industry.empty:
        merged = factor_df.merge(industry, on=["date", "symbol"], how="inner")
        if not merged.empty:
            max_ind_corr = _max_industry_corr(merged)

    # Existing-factor correlation: computed for diagnostics but NOT gated here.
    # The ridge R2 check in step8 is the single admission gate for style overlap.
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
        return _reject(state, "step2", "; ".join(violations), metrics)

    return _pass(state, "step2", metrics)


def _max_industry_corr(merged: pd.DataFrame) -> float:
    """Max abs Pearson corr between factor value and any industry dummy.

    Uses pre-encoded categorical codes + numpy boolean masks instead of
    repeated ``pd.get_dummies`` per date.  Follows the same pattern as
    :func:`backtest.factor.transforms.cs_ols_residualize`.
    """
    cat = merged["industry_code"].astype("category")
    codes = cat.cat.codes.to_numpy()
    vals = merged["value"].to_numpy(dtype=float)

    max_corr = 0.0
    for _, idx in merged.groupby("date", sort=False).groups.items():
        positions = np.asarray(idx, dtype=int)
        day_vals = vals[positions]
        valid = ~np.isnan(day_vals)
        if valid.sum() < 2:
            continue
        day_codes = codes[positions]
        present = np.unique(day_codes[valid])
        if present.size <= 1:
            continue
        # Drop first category to avoid collinearity (consistent with cs_ols_residualize)
        dummies = (day_codes[valid][:, None] == present[None, 1:]).astype(float)
        v = day_vals[valid]
        for j in range(dummies.shape[1]):
            c = np.corrcoef(v, dummies[:, j])[0, 1]
            if not np.isnan(c):
                max_corr = max(max_corr, abs(c))
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
        run_decile_backtest=True,
    )
    state.eval_result = eval_result

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
        h: {
            **eval_result.ic_metrics.get(h, {}),
            **{f"rank_{k}": v for k, v in eval_result.rank_ic_metrics.get(h, {}).items()},
        }
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

    # Generate IC time series plots now while data is in memory.
    # Report only references these — never re-runs evaluate().
    try:
        from backtest.factor.evaluation import plot_evaluation

        plots_dir = eval_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for h in (1, 5, 20):
            if h in eval_result.ic_series:
                plot_evaluation(eval_result, horizon=h,
                                output_path=str(plots_dir / f"ic_ts_h{h}.png"))

        # IC decay overview (from all_ic_metrics, no extra evaluate())
        _gen_ic_decay_plot(all_ic, plots_dir)

        state.artifacts["eval_plots_dir"] = str(plots_dir)
    except Exception:
        pass  # plot failure is non-fatal

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

    # Reuse eval_result from step3 if available.
    # Backward-compat: old states may have eval_result as a raw dict.
    eval_result = state.eval_result
    if isinstance(eval_result, dict):
        try:
            from backtest.factor.evaluation import EvaluationResult
            eval_result = EvaluationResult.from_dict(eval_result)
        except Exception:
            eval_result = None

    if eval_result is None:
        eval_result = evaluate(
            config.factor_id,
            config.start_date,
            config.end_date,
            horizons=config.eval_horizons,
            ret_type=config.ret_type,
            corr_top_k=0,
            exclude_limit_up=True,
            run_decile_backtest=True,
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

    spearman = _rank_ic_series(pd.Series(groups), pd.Series(mean_rets))
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
        # Generate decile backtest plot.  step3 already computes
        # decile_result (run_decile_backtest=True); the fallback
        # evaluate() below only fires when step4 runs standalone.
        try:
            if getattr(eval_result, "decile_result", None) is None:
                decile_eval = evaluate(
                    config.factor_id,
                    config.start_date,
                    config.end_date,
                    horizons=[20],
                    ret_type=config.ret_type,
                    corr_top_k=0,
                    exclude_limit_up=True,
                    run_decile_backtest=True,
                )
                eval_result.decile_result = decile_eval.decile_result

            if eval_result.decile_result is not None:
                from backtest.simulation.decile import plot_decile_backtest
                eval_dir = Path(config.results_root) / config.factor_id / "factor_eval"
                decile_png = eval_dir / "decile_backtest" / f"{config.factor_id}_decile.png"
                decile_png.parent.mkdir(parents=True, exist_ok=True)
                plot_decile_backtest(eval_result.decile_result, str(decile_png))
                state.eval_result = eval_result
        except Exception:
            pass  # plot failure is non-fatal

        # Generate group returns bar chart from step4 metrics.
        try:
            _gen_group_returns_plot(metrics.get("group_mean_returns", {}),
                                    Path(config.results_root) / config.factor_id
                                    / "factor_eval" / "plots")
        except Exception:
            pass

        return _pass(state, "step4", metrics)

    return _reject(
        state, "step4",
        f"Spearman={spearman:.3f} <= {config.thresholds.min_monotonicity}",
        metrics,
    )


# ---------------------------------------------------------------------------
# Step 5: Strategy config
# ---------------------------------------------------------------------------


def step5_build_strategy(
    state: PipelineState,
    top_pct: float | None = None,
    top_k: int | None = None,
    decay: int | None = None,
    universe: str | None = None,
    rebalance: str | None = None,
) -> PipelineState:
    """Build default strategy configuration.

    Params are taken from (in order of priority):
    1. CLI kwargs passed to this function
    2. PipelineConfig defaults

    ``top_k`` and ``top_pct`` are mutually exclusive — exactly one must be
    specified.  ``top_k`` takes priority if both are provided.
    """
    config = state.config

    # Resolve params with priority: CLI > config defaults
    _top_k = top_k if top_k is not None else config.default_top_k
    _top_pct = top_pct if top_pct is not None else config.default_top_pct
    _decay = decay if decay is not None else config.default_decay
    _rebalance = rebalance if rebalance is not None else config.default_rebalance
    _universe = _resolve_universe_override(universe, config.default_universe)

    # Benchmark follows the universe index; when universe is None (full market)
    # fall back to the configured default (HS300).
    _benchmark = _universe if _universe else config.benchmark

    # Determine selection: top_k takes priority if specified
    if _top_k is not None:
        selection = SelectionConfig(method="topk", top_k=_top_k)
    elif _top_pct is not None:
        selection = SelectionConfig(method="topk", top_pct=_top_pct)
    else:
        raise ValueError(
            "Neither top_k nor top_pct is specified. "
            "Set exactly one of default_top_k or default_top_pct in config.yaml, "
            "or pass it via CLI."
        )

    strategy_config = StrategyConfig(
        name=f"{config.factor_id}_pipeline",
        strategy_type="single_factor_topk",
        rebalance_freq=_rebalance,
        delay=1,
        universe=_build_universe(_universe, overrides=config.strategy_overrides),
        factors=[FactorConfig(id=config.factor_id, direction="desc")],
        selection=selection,
        weighting=WeightingConfig(method="equal"),
        decay=_decay if _decay > 0 else None,
        backtest=BacktestConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            benchmark=_benchmark,
        ),
    )

    state.strategy_config = strategy_config

    # Persist strategy config artifact. Sweep combos share a universe-level
    # results_dir but each has its own state directory, so strategy config must
    # follow state_subdir when present to avoid cross-combo overwrites.
    cfg_dir = config.state_path().parent if config.state_subdir else config.results_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "strategy_config.json"
    # Save the full StrategyConfig dict so step7 can restore it in a later
    # process.  The old minimal-metadata format was incompatible with
    # StrategyConfig.from_dict (universe as string vs dict).
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(strategy_config.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    state.artifacts["strategy_config"] = str(cfg_path)

    metrics: dict = {
        "decay": _decay,
        "rebalance": _rebalance,
        "universe": _universe,
    }
    if _top_k is not None:
        metrics["top_k"] = _top_k
    else:
        metrics["top_pct"] = _top_pct
    return _pass(state, "step5", metrics)


def _bt_threshold_map(suffix: str, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a threshold_map for a backtest gate.

    All backtest gates share the same absolute + excess metric structure;
    only the threshold suffix (``_simple`` / ``_detailed``) differs.
    """
    m = {
        "sharpe": f"min_sharpe{suffix}",
        "annual_return": f"min_annual_return{suffix}",
        "max_drawdown": f"max_max_drawdown{suffix}" if suffix == "_detailed" else "max_max_drawdown",
        "calmar": f"min_calmar{suffix}",
        "excess_sharpe": f"min_excess_sharpe{suffix}",
        "excess_annual_return": f"min_excess_annual_return{suffix}",
        "excess_max_drawdown": f"max_excess_max_drawdown{suffix}",
        "excess_calmar": f"min_excess_calmar{suffix}",
    }
    if extra:
        m.update(extra)
    return m


# ---------------------------------------------------------------------------
# Step 6: Simple backtest gate
# ---------------------------------------------------------------------------


def _load_simulation_config(
    price_type: str = "o2o",
    overrides: dict | None = None,
) -> SimulationConfig:
    """Build SimulationConfig from hardcoded defaults + optional per-factor overrides."""
    defaults = SimulationConfig()
    cfg: dict[str, Any] = {
        "initial_cash": defaults.initial_cash,
        "commission_rate": defaults.commission_rate,
        "stamp_duty_rate": defaults.stamp_duty_rate,
        "transfer_fee_rate": defaults.transfer_fee_rate,
        "allow_short": defaults.allow_short,
    }
    if overrides:
        cfg.update(overrides)

    return SimulationConfig(
        initial_cash=float(cfg["initial_cash"]),
        commission_rate=float(cfg["commission_rate"]),
        stamp_duty_rate=float(cfg["stamp_duty_rate"]),
        transfer_fee_rate=float(cfg["transfer_fee_rate"]),
        allow_short=cfg["allow_short"],
        price_type=price_type,
    )


_MARKET_BUFFER_DAYS = 10


def step6_simple_backtest(state: PipelineState) -> PipelineState:
    """Vectorised simple backtest with threshold gates."""
    config = state.config

    if state.strategy_config is None:
        return _reject(state, "step6", "No strategy config. Run step5 first.")

    sc = state.strategy_config
    if isinstance(sc, dict):
        sc = StrategyConfig.from_dict(sc)
    strategy = SingleFactorStrategy(sc)

    # Pre-load market data once so the strategy doesn't load its own copy.
    # Keep a post-end buffer for the simulator's T+1 return calculation, but
    # pass only the original date range into strategy/universe filtering.
    market_end = (
        pd.to_datetime(config.end_date)
        + pd.Timedelta(days=_MARKET_BUFFER_DAYS)
    ).strftime("%Y%m%d")
    with MarketStorage(read_only=True) as ms:
        market_panel = ms.get_bars(
            symbols=None,
            start=config.start_date, end=market_end,
            columns=["close", "open", "high", "low", "adj_factor", "circ_mv", "amount",
                     "is_st", "list_date", "limit_up", "limit_down"],
        )
        market_for_strategy = market_panel[
            market_panel["date"] <= pd.to_datetime(config.end_date)
        ]

        signals = strategy.run(
            config.start_date,
            config.end_date,
            market_storage=ms,
            market_panel=market_for_strategy,
        )
    state.signals = signals

    if signals.empty:
        return _reject(state, "step6", "Strategy produced no signals.")

    # Persist signals so step7 can resume in a later process.
    tag = _build_tag(state)
    signals_dir = config.results_dir() / tag
    signals_dir.mkdir(parents=True, exist_ok=True)
    signals_path = signals_dir / "signals.parquet"
    signals.to_parquet(signals_path, index=False)
    state.artifacts["signals"] = str(signals_path)

    # Reuse pre-loaded data filtered to signal symbols for the simulator.
    signal_symbols = set(signals["symbol"].unique())
    market_data = market_panel[market_panel["symbol"].isin(signal_symbols)]
    sim_cfg = _load_simulation_config(overrides=config.simulation_overrides)
    sim = SimpleSimulator(sim_cfg)
    result = sim.run(signals, market_data)

    return _backtest_gate(state, result, "step6", "simple", _bt_threshold_map("_simple"))


# ---------------------------------------------------------------------------
# Step 7: Detailed backtest gate
# ---------------------------------------------------------------------------


def step7_detailed_backtest(state: PipelineState) -> PipelineState:
    """Event-driven detailed backtest with threshold gates."""
    config = state.config

    # Restore strategy_config and signals from persisted artifacts when
    # resuming from step7 in a different process (e.g. sweep validate-top-n).
    if isinstance(state.strategy_config, dict):
        state.strategy_config = StrategyConfig.from_dict(state.strategy_config)
    elif state.strategy_config is None:
        cfg_path = state.artifacts.get("strategy_config")
        if cfg_path and Path(cfg_path).exists():
            with Path(cfg_path).open("r", encoding="utf-8") as f:
                state.strategy_config = StrategyConfig.from_dict(json.load(f))

    if state.signals is None:
        signals_path = state.artifacts.get("signals")
        if signals_path and Path(signals_path).exists():
            state.signals = pd.read_parquet(signals_path)
            state.signals["date"] = pd.to_datetime(state.signals["date"])

    if state.strategy_config is None or state.signals is None:
        return _reject(state, "step7", "No strategy/signals. Run step5-6 first.")

    market_data, dividends = _load_market_data(config, state.signals, with_dividends=True)
    price_type = "o2o" if config.ret_type == "open" else "c2c"
    sim_cfg = _load_simulation_config(price_type=price_type, overrides=config.simulation_overrides)
    sim = DetailedSimulator(sim_cfg)
    result = sim.run(state.signals, market_data, dividends)

    return _backtest_gate(state, result, "step7", "detailed",
        _bt_threshold_map("_detailed", extra={"annual_turnover": "max_annual_turnover_detailed"}))


# ---------------------------------------------------------------------------
# Shared backtest helpers
# ---------------------------------------------------------------------------


def _load_market_data(
    config: PipelineConfig,
    signals: pd.DataFrame,
    *,
    with_dividends: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load market bars (and optionally dividends) for backtest."""
    market_end = (
        pd.to_datetime(config.end_date)
        + pd.Timedelta(days=_MARKET_BUFFER_DAYS)
    ).strftime("%Y%m%d")
    symbols = signals["symbol"].unique().tolist()

    # Only request columns actually used by the simulators.
    # limit_up / limit_down are required by DetailedSimulator.validate_columns().
    columns = [
        "close", "open", "high", "low", "adj_factor",
        "volume", "amount", "turnover_rate", "circ_mv",
        "is_st", "list_date", "st_status",
        "limit_up", "limit_down",
    ]

    with MarketStorage(read_only=True) as ms:
        market_data = ms.get_bars(
            symbols=symbols, start=config.start_date, end=market_end,
            columns=columns,
        )
        if with_dividends:
            dividends = ms.get_dividends(
                symbols=symbols, start=config.start_date, end=market_end,
            )
            return market_data, dividends
    return market_data


def _backtest_gate(
    state: PipelineState,
    result: "BacktestResult",
    step: str,
    sub_dir: str,
    threshold_map: dict[str, str],
) -> PipelineState:
    """Persist result, check thresholds, record pass/reject.

    Excess-metric keys in *threshold_map* are generic (e.g. ``excess_sharpe``).
    They are resolved at runtime to the per-benchmark keys that
    ``BacktestResult.summary()`` produces (e.g. ``excess_sharpe_hs300``)
    by appending the benchmark alias derived from the strategy's benchmark.
    """
    config = state.config
    tag = _build_tag(state)
    out_dir = config.results_dir() / tag / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = _load_simulation_config(overrides=config.simulation_overrides)
    result.save(str(out_dir), metadata={
        "strategy": {"name": state.strategy_config.name, "factor": config.factor_id},
        "simulation": {"engine": sub_dir.capitalize(), "initial_cash": sim_cfg.initial_cash},
    })
    state.artifacts[sub_dir + "_bt"] = str(out_dir)

    metrics = result.summary()
    if step == "step6":
        state.simple_bt_metrics = metrics
    else:
        state.detailed_bt_metrics = metrics

    # Resolve the benchmark alias so generic excess keys (e.g. "excess_sharpe")
    # can be mapped to per-benchmark metric keys (e.g. "excess_sharpe_hs300").
    bench_code = (
        state.strategy_config.backtest.benchmark
        if state.strategy_config and state.strategy_config.backtest
        else "000300.SH"
    )
    from backtest.evaluation.benchmark import _INDEX_TO_BENCHMARK_ALIAS

    bench_alias = _INDEX_TO_BENCHMARK_ALIAS.get(bench_code)
    if bench_alias is None:
        import warnings
        warnings.warn(
            f"Benchmark code '{bench_code}' is not in the known benchmark "
            f"alias map ({list(_INDEX_TO_BENCHMARK_ALIAS.keys())}). "
            f"Falling back to 'hs300' for excess metric gate checks.",
            stacklevel=2,
        )
        bench_alias = "hs300"

    th = config.thresholds
    checks: dict[str, bool] = {}
    for metric_key, th_key in threshold_map.items():
        threshold = getattr(th, th_key)
        # None means the threshold is disabled — skip it.
        if threshold is None:
            continue

        # Resolve generic excess keys to per-benchmark metric keys.
        if metric_key.startswith("excess_"):
            resolved_key = f"{metric_key}_{bench_alias}"
        else:
            resolved_key = metric_key

        val = metrics.get(resolved_key)
        # NaN means the engine doesn't compute this metric (e.g. SimpleSimulator
        # doesn't track turnover).  Skip the check rather than failing.
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        if "max_drawdown" in metric_key:
            checks[metric_key] = val > -threshold
        elif metric_key in ("annual_turnover",):
            checks[metric_key] = val < threshold
        else:
            checks[metric_key] = val > threshold

    if not checks:
        # All thresholds were disabled (None/null) — warn because the step
        # passes with zero verification.
        import warnings
        warnings.warn(
            f"{step}: all thresholds are disabled — gate passes vacuously.",
            stacklevel=2,
        )
    if all(checks.values()):
        _gen_backtest_nav_plot(state, sub_dir, out_dir, result.nav_df)
        return _pass(state, step, metrics)

    violations = []
    for metric_key, passed in checks.items():
        if not passed:
            resolved_key = (
                f"{metric_key}_{bench_alias}"
                if metric_key.startswith("excess_") else metric_key
            )
            val = metrics.get(resolved_key, float("nan"))
            label = metric_key.replace("_", " ").title()
            # Gate direction: "higher is better" for all metrics except
            # max_drawdown (checked as val > -threshold) and annual_turnover
            # (checked as val < threshold).
            if "max_drawdown" in metric_key:
                direction = "deeper than"
            elif metric_key in ("annual_turnover",):
                direction = "exceeds"
            else:
                direction = "below"
            violations.append(f"{label}={val:.3f} {direction} threshold")
    return _reject(state, step, "; ".join(violations), metrics)


def _build_tag(state: PipelineState) -> str:
    """Build strategy tag for artifact paths (delegates to shared implementation)."""
    from backtest.pipeline._report import _build_tag as _bt

    return _bt(state)


def _setup_cjk_font() -> None:
    """Configure matplotlib for CJK text rendering.  Idempotent; cheap to
    call at the top of every plot-generation function."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as _plt
    for _font in ("PingFang HK", "Heiti TC", "STHeiti", "Arial Unicode MS",
                  "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
        try:
            fm.findfont(_font, fallback_to_default=False)
            _plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
            _plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue


def _gen_group_returns_plot(group_rets: dict, plots_dir: Path) -> None:
    """Generate group-returns bar chart from step4 monotonicity metrics."""
    try:
        _setup_cjk_font()
        import matplotlib.pyplot as plt

        if not group_rets:
            return
        groups = sorted(int(g) for g in group_rets)
        values = [group_rets.get(str(g), group_rets.get(g, 0)) for g in groups]
        if not groups:
            return

        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["#d73027" if v < 0 else "#4575b4" for v in values]
        ax.bar(groups, values, color=colors, alpha=0.85)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("分位组")
        ax.set_ylabel("平均前瞻收益")
        ax.set_title("各分位组平均收益（h=1）")
        ax.set_xticks(groups)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / "eval_group_returns.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def _gen_ic_decay_plot(all_ic: dict, plots_dir: Path) -> None:
    """Generate IC decay overview chart from pre-computed IC metrics."""
    try:
        _setup_cjk_font()
        import matplotlib.pyplot as plt
        import numpy as np

        horizons = sorted(int(h) if isinstance(h, int) else int(h) for h in all_ic)

        def _get(h, key):
            ic = all_ic.get(h, {}) or all_ic.get(str(h), {})
            return ic.get(key, np.nan)

        ic_means = [_get(h, "ic_mean") for h in horizons]
        ic_stds = [_get(h, "ic_std") for h in horizons]
        ic_icirs = [_get(h, "icir") for h in horizons]
        ric_means = [_get(h, "rank_ic_mean") for h in horizons]
        has_rankic = any(not np.isnan(v) for v in ric_means)

        n_cols = 4 if has_rankic else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
        if n_cols == 2:
            axes = [axes, None, None, None]

        ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]
        ax1.errorbar(horizons, ic_means, yerr=ic_stds, marker="o",
                     color="steelblue", capsize=4, linewidth=1.5)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("预测周期（天）")
        ax1.set_ylabel("IC")
        ax1.set_title("IC 均值 ± 标准差（Pearson）")
        ax1.grid(True, alpha=0.3)
        ax2.bar(horizons, ic_icirs, color="darkorange", alpha=0.8)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("预测周期（天）")
        ax2.set_ylabel("ICIR")
        ax2.set_title("ICIR（Pearson）")
        ax2.grid(True, alpha=0.3, axis="y")

        if has_rankic:
            ric_stds = [_get(h, "rank_ic_std") for h in horizons]
            ric_icirs = [_get(h, "rank_icir") for h in horizons]
            ax3.errorbar(horizons, ric_means, yerr=ric_stds, marker="o",
                         color="seagreen", capsize=4, linewidth=1.5)
            ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax3.set_xlabel("预测周期（天）")
            ax3.set_ylabel("RankIC")
            ax3.set_title("RankIC 均值 ± 标准差（Spearman）")
            ax3.grid(True, alpha=0.3)
            ax4.bar(horizons, ric_icirs, color="mediumpurple", alpha=0.8)
            ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax4.set_xlabel("预测周期（天）")
            ax4.set_ylabel("RankICIR")
            ax4.set_title("RankICIR（Spearman）")
            ax4.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(plots_dir / "eval_ic_decay.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def _gen_backtest_nav_plot(
    state: PipelineState, sub_dir: str, out_dir: Path,
    nav_df: "pd.DataFrame",
) -> None:
    """Generate NAV + drawdown chart from in-memory backtest result.

    Uses the caller's live ``nav_df`` directly rather than re-reading the
    just-written ``nav.parquet`` from disk.
    """
    try:
        _setup_cjk_font()
        import matplotlib.pyplot as plt

        if nav_df is None or nav_df.empty or "nav" not in nav_df.columns:
            return

        nav_df = nav_df.copy()
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_series = nav_df.set_index("date")["nav"].astype(float)
        nav_norm = nav_series / nav_series.iloc[0]
        drawdown = nav_series / nav_series.expanding().max() - 1.0

        title_map = {"simple": "简单回测", "detailed": "详细回测"}
        title = title_map.get(sub_dir, sub_dir)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
        ax1.plot(nav_norm.index, nav_norm.values, color="steelblue",
                 linewidth=1.4, label="策略")
        # Overlay benchmark if available
        bench_code = getattr(state.config, "benchmark", None)
        if bench_code:
            try:
                from backtest.evaluation.benchmark import align_benchmark, load_benchmark
                bench_nav = load_benchmark(
                    bench_code,
                    start=nav_df["date"].min().strftime("%Y%m%d"),
                    end=nav_df["date"].max().strftime("%Y%m%d"),
                )
                bench_aligned = align_benchmark(nav_df, bench_nav)
                ax1.plot(bench_aligned.index, bench_aligned.values,
                         color="darkorange", linewidth=1.2,
                         label=f"基准 ({bench_code})")
            except Exception:
                pass
        ax1.legend(loc="upper left")
        ax1.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
        ret_type = state.config.ret_type
        ret_label = "o2o" if ret_type == "open" else "c2c"
        ax1.set_ylabel("净值")
        ax1.set_title(f"{title} — 净值曲线 ({ret_label})")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color="red", alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=1.0)
        ax2.set_ylabel("回撤")
        ax2.set_xlabel("日期")
        ax2.set_title(f"{title} — 回撤曲线 ({ret_label})")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out_png = out_dir / f"nav_{sub_dir}.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass  # plot failure is non-fatal



# ---------------------------------------------------------------------------
# Step 8: Ridge R2 classification
# ---------------------------------------------------------------------------


def step8_ridge_r2(state: PipelineState) -> PipelineState:
    """Per-date Ridge R² classification against ALL admitted factors.

    When R² exceeds the smart_beta threshold, the factor is NOT rejected
    outright — instead it is marked ``needs_residual`` and step9 decides
    whether the residual has predictive power worth admitting.
    """
    config = state.config

    try:
        ridge_result = ridge_r2_check(
            config.factor_id,
            start=config.start_date,
            end=config.end_date,
        )
    except Exception as exc:
        return _reject(state, "step8", f"Ridge check failed: {exc}")

    state.ridge_result = ridge_result

    th = get_section("thresholds", "admission", "ridge_r2")
    needs_residual = ridge_result.r2 >= th["smart_beta_max"]

    metrics = {
        "r2": float(ridge_result.r2),
        "tier": ridge_result.tier if not needs_residual else TIER_REJECT,
        "n_obs": ridge_result.n_obs,
        "needs_residual": needs_residual,
        "r2_stats": ridge_result.r2_stats,
    }

    if needs_residual:
        return _pass(
            state, "step8", metrics,
        )

    return _pass(state, "step8", metrics)


# ---------------------------------------------------------------------------
# Step 9: Residual ICIR incremental-information check
# ---------------------------------------------------------------------------


def step9_residual_icir(state: PipelineState) -> PipelineState:
    """Per-date Ridge regression against ALL admitted factors, residual RankICIR.

    Two admission paths:
    - **Normal**: step8 R² < smart_beta_max → admit raw factor values.
    - **Residual**: step8 R² ≥ smart_beta_max but residual ICIR passes →
      admit the *residualised* factor (orthogonal to existing factors).

    If residual ICIR fails in either case → reject.
    """
    config = state.config

    # Reuse step8's precomputed per-date residuals to avoid a second Ridge fit
    precomputed = getattr(state.ridge_result, "residuals_df", None) if state.ridge_result else None

    try:
        th = get_section("thresholds", "admission", "residual_icir")
        result = residual_icir_check(
            config.factor_id,
            horizons=th.get("horizons", [1, 5, 20]),
            threshold=float(th.get("min_annual_icir", 0.05)),
            ic_mean_threshold=float(th.get("min_abs_ic_mean", 0.001)),
            alpha=float(th.get("ridge_alpha", 1.0)),
            ret_type=config.ret_type,
            start=config.start_date,
            end=config.end_date,
            precomputed_residuals=precomputed,
            precomputed_n_regressors=(
                state.ridge_result.n_regressors
                if (precomputed is not None and state.ridge_result)
                else -1
            ),
        )
    except (ValueError, KeyError, InsufficientOverlapError,
            CandidateNotBackfilledError) as exc:
        return _reject(state, "step9", f"Residual ICIR check failed: {exc}")

    state.residual_icir_result = result

    sr8 = state.step_results.get("step8")
    needs_residual = sr8.metrics.get("needs_residual", False) if sr8 else False

    admission_mode = "residual" if (needs_residual and result.passed) else (
        "raw" if result.passed else "reject"
    )

    # When using precomputed residuals, n_regressors from step8
    n_reg = (
        state.ridge_result.n_regressors
        if (precomputed is not None and state.ridge_result)
        else result.n_regressors
    )

    metrics = {
        "residual_rank_icirs": result.residual_rank_icirs,
        "annual_icirs": result.annual_icirs,
        "residual_rank_ic_means": result.residual_rank_ic_means,
        "residual_rank_ic_stds": result.residual_rank_ic_stds,
        "n_regressors": n_reg,
        "n_dates": result.n_dates,
        "n_obs_total": result.n_obs_total,
        "threshold": result.threshold,
        "ic_mean_threshold": result.ic_mean_threshold,
        "passed": result.passed,
        "admission_mode": admission_mode,
    }

    if result.passed:
        return _pass(state, "step9", metrics)

    max_annual = max(
        (v for v in result.annual_icirs.values() if not math.isnan(v)),
        default=float("-inf"),
    )
    return _reject(
        state, "step9",
        f"Residual ICIR: max annualised={max_annual:.4f}, "
        f"threshold={result.threshold}. No horizon adds incremental "
        f"information beyond {result.n_regressors} existing factors.",
        metrics,
    )


# ---------------------------------------------------------------------------
# Step 10: Report + admission
# ---------------------------------------------------------------------------


def step10_report_and_admit(state: PipelineState) -> PipelineState:
    """Mark pipeline as ready for human review.

    Does NOT auto-admit. The human reviews the report and manually runs
    ``python -m backtest.factor.admission admit <fid>``.

    The report itself is generated by ``run_pipeline()`` after the step
    loop, so it covers both pass and rejection cases exactly once.
    """
    config = state.config

    state.status = "ready_for_review"
    return _pass(state, "step10", {
        "action": "ready_for_review",
        "next_step": f"python -m backtest.factor.admission admit {config.factor_id}",
    })


# ---------------------------------------------------------------------------
# Shared pipeline runner — called by both manual CLI and agent
# ---------------------------------------------------------------------------

_STEP_ORDER = [
    ("step1", step1_coverage_check),
    ("step2", step2_neutralization_check),
    ("step3", step3_icir_check),
    ("step4", step4_monotonicity_check),
    ("step5", step5_build_strategy),
    ("step6", step6_simple_backtest),
    ("step7", step7_detailed_backtest),
    ("step8", step8_ridge_r2),
    ("step9", step9_residual_icir),
    ("step10", step10_report_and_admit),
]

def run_pipeline(
    factor_id: str,
    *,
    frequency: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    results_root: str = "results",
    results_subdir: str | None = None,
    state_subdir: str | None = None,
    ret_type: str | None = None,
    benchmark: str | None = None,
    from_step: int = 1,
    to_step: int | None = None,
    skip_report: bool = False,
    skip_mark_rejected: bool = False,
    # Strategy kwargs forwarded to step5_build_strategy
    top_k: int | None = None,
    top_pct: float | None = None,
    decay: int | None = None,
    universe: str | None = None,
    rebalance: str | None = None,
) -> PipelineState:
    """Execute step1~step10 and generate report.

    Shared between the manual CLI (``python -m backtest.pipeline run-all``)
    and the agent runner (``agents/runner.py``).  Both paths get identical
    pipeline behavior, state persistence, and artifacts.

    Parameters
    ----------
    from_step : int
        1-based step index to start from (default 1 = step1).
    to_step : int | None
        If set, stop after this step (inclusive).  None = run all 10 steps.
    skip_mark_rejected : bool
        If True, do not call ``mark_rejected()`` on failure.  The agent
        runner sets this to True because it manages rejection differently
        (cleanup_work_db + experiment error field).
    top_k / top_pct / decay / universe / rebalance : optional
        Strategy params forwarded to step5.  When provided, they take
        priority over config.yaml defaults.  Useful for re-running from
        step5 with adjusted params after a backtest_fail.
    results_subdir / state_subdir : optional
        Nest results and state under results_root/<factor_id>/<subdir>/
        instead of the flat results_root/<factor_id>/ layout.  Used by
        the agent sweep to isolate per-combo artifacts.
    """
    # Validate from_step range.  Values >5 can only work when PipelineState
    # is loaded from a prior run (per-step CLI), not with a fresh state.
    if not 1 <= from_step <= 10:
        raise ValueError(f"from_step must be 1-10, got {from_step}")
    strategy_kwargs = {
        k: v for k, v in (("top_k", top_k), ("top_pct", top_pct),
                           ("decay", decay), ("universe", universe),
                           ("rebalance", rebalance))
        if v is not None
    }
    if strategy_kwargs and from_step > 5:
        warnings.warn(
            f"Strategy kwargs {list(strategy_kwargs.keys())} provided but "
            f"from_step={from_step} > 5 — step5 will be skipped and these "
            f"kwargs ignored. Use --from-step 5 to apply strategy overrides.",
            stacklevel=2,
        )

    overrides: dict[str, Any] = {"results_root": results_root}
    if results_subdir is not None:
        overrides["results_subdir"] = results_subdir
    if state_subdir is not None:
        overrides["state_subdir"] = state_subdir
    if start_date is not None:
        overrides["start_date"] = start_date
    if end_date is not None:
        overrides["end_date"] = end_date
    if ret_type is not None:
        overrides["ret_type"] = ret_type
    if benchmark is not None:
        overrides["benchmark"] = benchmark

    config = PipelineConfig.from_factor_config(
        factor_id=factor_id,
        frequency=frequency,
        **overrides,
    )

    state_path = config.state_path()
    if from_step == 1 or not state_path.exists():
        state = PipelineState(factor_id=factor_id, config=config)
    else:
        state = PipelineState.load(state_path)
        # Update config with any new overrides (e.g. changed start/end dates).
        state.config = config
        state.clear_from_step(f"step{from_step}")
        if from_step > 5:
            # When resuming from step6+, step5 may have been skipped.
            # Warn if strategy kwargs were provided but step5 won't be re-run.
            pass  # warning already emitted above
    state.save(state_path)

    # Run all steps linearly (one attempt each).  Strategy param tuning is
    # the caller's responsibility: re-run with from_step=5 + new kwargs when
    # step6 or step7 fails.
    for step_name, step_fn in _STEP_ORDER:
        step_idx = int(step_name[4:])
        if step_idx < from_step:
            continue
        if state.is_rejected():
            break
        if step_name == "step5":
            state = step_fn(state, top_k=top_k, top_pct=top_pct,
                            decay=decay, universe=universe, rebalance=rebalance)
        else:
            state = step_fn(state)
        state.save(config.state_path())
        if to_step is not None and step_idx >= to_step:
            break

    # Generate a diagnostic report unless the caller explicitly skips it
    # (e.g. quick sweep workers that only need numeric metrics).
    if not skip_report:
        from backtest.pipeline._report import generate_pipeline_report

        report_path = generate_pipeline_report(state)
        state.artifacts["report"] = str(report_path)

    if state.is_rejected() and not skip_mark_rejected:
        from backtest.factor.admission import mark_rejected

        last_step = state.last_step()
        reason = (
            state.step_results[last_step].reason
            if last_step and last_step in state.step_results
            else "unknown"
        )
        mark_rejected(
            factor_id,
            notes=f"Pipeline rejected at {last_step}: {reason}",
        )
        state.status = "rejected"
        state.save(config.state_path())

    return state
