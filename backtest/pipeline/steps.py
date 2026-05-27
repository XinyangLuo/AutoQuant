"""Pipeline step functions (step1~step10).

Each function is a pure transform: takes PipelineState, returns updated
PipelineState.  They are called by the CLI dispatcher in __main__.py.
"""

from __future__ import annotations

import json
import math
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


def _build_universe(index_members: str | None = None) -> UniverseConfig:
    """Build UniverseConfig from ``config.yaml`` → ``strategy.universe``.

    Falls back to safe defaults when config.yaml is missing a key.
    """
    from backtest.config_loader import get_section_or

    return UniverseConfig(
        exclude_st=get_section_or(True, "strategy", "universe", "exclude_st"),
        exclude_new_ipo_days=get_section_or(252, "strategy", "universe", "exclude_new_ipo_days"),
        include_cyb=get_section_or(True, "strategy", "universe", "include_cyb"),
        include_kcb=get_section_or(False, "strategy", "universe", "include_kcb"),
        include_bse=get_section_or(False, "strategy", "universe", "include_bse"),
        index_members=index_members,
        min_market_cap=get_section_or(500000000, "strategy", "universe", "min_market_cap"),
        min_avg_amount=get_section_or(10000000, "strategy", "universe", "min_avg_amount"),
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

    with FactorStorage() as fs:
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

    with FactorStorage() as fs:
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
    """Max abs Pearson corr between factor value and any industry dummy."""
    max_corr = 0.0
    for date, group in merged.groupby("date"):
        dummies = pd.get_dummies(group["industry_code"], prefix="ind")
        for col in dummies.columns:
            corr = _ic_series(group["value"], dummies[col])
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

    # Reuse eval_result from step3 if available (may be dict after JSON round-trip)
    eval_result = state.eval_result
    if eval_result is None or isinstance(eval_result, dict):
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
    2. state.retry_params from a previous failed attempt
    3. PipelineConfig defaults

    ``top_k`` and ``top_pct`` are mutually exclusive — exactly one must be
    specified.  ``top_k`` takes priority if both are provided.
    """
    config = state.config

    # Resolve params with priority: CLI > retry_params > defaults
    rp = state.retry_params
    _top_k = top_k if top_k is not None else rp.get("top_k", config.default_top_k)
    _top_pct = top_pct if top_pct is not None else rp.get("top_pct", config.default_top_pct)
    _decay = decay if decay is not None else rp.get("decay", config.default_decay)
    _rebalance = rebalance if rebalance is not None else rp.get("rebalance", config.default_rebalance)
    _universe = universe if universe is not None else rp.get("universe", config.default_universe)

    # Determine selection: top_k takes priority if specified
    if _top_k is not None:
        selection = SelectionConfig(method="topk", top_k=_top_k)
    elif _top_pct is not None:
        selection = SelectionConfig(method="topk", top_pct=_top_pct)
    else:
        raise ValueError(
            "Neither top_k nor top_pct is specified. "
            "Set exactly one of default_top_k or default_top_pct in config.yaml, "
            "or pass it via CLI / retry_params."
        )

    strategy_config = StrategyConfig(
        name=f"{config.factor_id}_pipeline",
        strategy_type="single_factor_topk",
        rebalance_freq=_rebalance,
        delay=1,
        universe=_build_universe(_universe),
        factors=[FactorConfig(id=config.factor_id, direction="desc")],
        selection=selection,
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
    artifact: dict = {
        "name": strategy_config.name,
        "rebalance_freq": strategy_config.rebalance_freq,
        "decay": strategy_config.decay,
        "universe": _universe,
    }
    if _top_k is not None:
        artifact["top_k"] = _top_k
    else:
        artifact["top_pct"] = _top_pct
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2, default=str)
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


# ---------------------------------------------------------------------------
# Step 6: Simple backtest gate
# ---------------------------------------------------------------------------


def _load_simulation_config(price_type: str = "o2o") -> SimulationConfig:
    """Build SimulationConfig from ``config.yaml simulation`` section.

    Falls back to ``SimulationConfig`` defaults when config.yaml is missing
    or a key is absent.
    """
    from backtest.config_loader import get_section

    defaults = SimulationConfig()

    def _get(key: str):
        try:
            return get_section("simulation", key)
        except (KeyError, FileNotFoundError):
            return getattr(defaults, key)

    return SimulationConfig(
        initial_cash=float(_get("initial_cash")),
        commission_rate=float(_get("commission_rate")),
        stamp_duty_rate=float(_get("stamp_duty_rate")),
        transfer_fee_rate=float(_get("transfer_fee_rate")),
        allow_short=_get("allow_short"),
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
    signals = strategy.run(config.start_date, config.end_date)
    state.signals = signals

    if signals.empty:
        return _reject(state, "step6", "Strategy produced no signals.")

    market_data = _load_market_data(config, signals)
    sim_cfg = _load_simulation_config()
    sim = SimpleSimulator(sim_cfg)
    result = sim.run(signals, market_data)

    return _backtest_gate(state, result, "step6", "simple", {
        "sharpe": "min_sharpe_simple",
        "annual_return": "min_annual_return_simple",
        "max_drawdown": "max_max_drawdown",
        "calmar": "min_calmar_simple",
    })


# ---------------------------------------------------------------------------
# Step 7: Detailed backtest gate
# ---------------------------------------------------------------------------


def step7_detailed_backtest(state: PipelineState) -> PipelineState:
    """Event-driven detailed backtest with threshold gates."""
    config = state.config

    if state.strategy_config is None or state.signals is None:
        return _reject(state, "step7", "No strategy/signals. Run step5-6 first.")

    market_data, dividends = _load_market_data(config, state.signals, with_dividends=True)
    price_type = "o2o" if config.ret_type == "open" else "c2c"
    sim_cfg = _load_simulation_config(price_type=price_type)
    sim = DetailedSimulator(sim_cfg)
    result = sim.run(state.signals, market_data, dividends)

    return _backtest_gate(state, result, "step7", "detailed", {
        "sharpe": "min_sharpe_detailed",
        "annual_return": "min_annual_return_detailed",
        "max_drawdown": "max_max_drawdown",
        "calmar": "min_calmar_detailed",
        "annual_turnover": "max_annual_turnover",
    })


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
    with MarketStorage(read_only=True) as ms:
        market_data = ms.get_bars(
            symbols=symbols, start=config.start_date, end=market_end,
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
    """Persist result, check thresholds, record pass/reject."""
    config = state.config
    tag = _build_tag(state)
    out_dir = Path(config.results_root) / config.factor_id / tag / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = _load_simulation_config()
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

    th = config.thresholds
    checks: dict[str, bool] = {}
    for metric_key, th_key in threshold_map.items():
        threshold = getattr(th, th_key)
        val = metrics.get(metric_key)
        # NaN means the engine doesn't compute this metric (e.g. SimpleSimulator
        # doesn't track turnover).  Skip the check rather than failing.
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        if metric_key in ("max_drawdown",):
            checks[metric_key] = val > -threshold
        elif metric_key in ("annual_turnover",):
            checks[metric_key] = val < threshold
        else:
            checks[metric_key] = val > threshold

    if all(checks.values()):
        return _pass(state, step, metrics)

    violations = []
    for metric_key, passed in checks.items():
        if not passed:
            val = metrics.get(metric_key, 0)
            label = metric_key.replace("_", " ").title()
            violations.append(f"{label}={val:.3f} <= threshold")
    return _reject(state, step, "; ".join(violations), metrics)


def _build_tag(state: PipelineState) -> str:
    """Mirror run_factor_pipeline.py tag format."""
    cfg = state.strategy_config
    if cfg is None:
        return "default"
    sel = cfg.selection
    if sel.top_pct is not None:
        tag = f"top{int(round(sel.top_pct * 100))}pct"
    else:
        tag = f"top{sel.top_k}"
    decay = cfg.decay or 0
    return f"{tag}_{cfg.rebalance_freq.lower()}_d{decay}"



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
        "n_obs": ridge_result.n_obs,
    }

    if passed:
        return _pass(state, "step8", metrics)

    th = get_section("thresholds", "admission", "ridge_r2")
    reject_at = th["smart_beta_max"]
    return _reject(
        state, "step8",
        f"R2={ridge_result.r2:.3f} >= {reject_at}, "
        f"tier={TIER_REJECT} (style clone)",
        metrics,
    )


# ---------------------------------------------------------------------------
# Step 9: Residual ICIR incremental-information check
# ---------------------------------------------------------------------------


def step9_residual_icir(state: PipelineState) -> PipelineState:
    """Regress candidate against ALL admitted factors, check residual RankICIR.

    Passes if the annualised residual RankICIR exceeds the configured
    threshold for at least one forward-return horizon (1D / 5D / 20D).
    """
    config = state.config

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
        )
    except (ValueError, KeyError, InsufficientOverlapError,
            CandidateNotBackfilledError) as exc:
        return _reject(state, "step9", f"Residual ICIR check failed: {exc}")

    state.residual_icir_result = result

    passed = result.passed
    metrics = {
        "residual_rank_icirs": result.residual_rank_icirs,
        "annual_icirs": result.annual_icirs,
        "residual_rank_ic_means": result.residual_rank_ic_means,
        "residual_rank_ic_stds": result.residual_rank_ic_stds,
        "n_regressors": result.n_regressors,
        "n_dates": result.n_dates,
        "n_obs_total": result.n_obs_total,
        "threshold": result.threshold,
        "ic_mean_threshold": result.ic_mean_threshold,
        "passed": result.passed,
    }

    if passed:
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
    """Generate markdown report and mark pipeline as ready for human review.

    Does NOT auto-admit. The human reviews the report and manually runs
    ``python -m backtest.factor.admission admit <fid>``.
    """
    config = state.config

    # Generate report
    from backtest.pipeline._report import generate_pipeline_report

    report_path = generate_pipeline_report(state)
    state.artifacts["report"] = str(report_path)

    state.status = "ready_for_review"
    return _pass(state, "step10", {
        "report_path": str(report_path),
        "action": "ready_for_review",
        "next_step": f"python -m backtest.factor.admission admit {config.factor_id}",
    })
