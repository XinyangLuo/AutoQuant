"""Factor compute engine: fetch raw data, enforce PIT isolation, call registered functions."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from functools import partial

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.factor.builtin.barra._common import apply_l3_pipeline
from backtest.factor.registry import get_factor_function, get_factor_meta
from backtest.factor.storage import FactorLibrary, FactorStorage
from backtest.factor.transforms import cs_ols_residualize, cs_zscore, industry_median_fill
from backtest.factor.variants import (
    BARRA_IND_SIZE_VARIANT,
    BARRA_L3_VARIANT,
    NONE_VARIANT,
    SIZE_L1_ID,
)


def compute_factor(
    factor_id: str,
    start_date: str,
    end_date: str,
    *,
    market_storage: MarketStorage | None = None,
    factor_storage: FactorStorage | None = None,
) -> pd.DataFrame:
    """Compute a single factor for a date range and return raw values.

    Steps:
      1. Load factor metadata from registry.
      2. Fetch required raw data with PIT isolation (no future data).
      3. Call the registered compute function.
      4. Return a DataFrame with columns [date, symbol, factor_id, value].

    The caller is responsible for writing the result to FactorStorage.
    Only the raw factor is computed here. To apply the registry-declared
    neutralization, pass the result to :func:`apply_variant_pipeline`.

    Fina-heavy factors (Growth / Quality / Value / ...) read PIT snapshots
    via :meth:`MarketStorage.get_fina_snapshot_range` — one range-join SQL
    per fina table replaces the legacy per-trade-date QUALIFY loop, so a
    35-year backfill no longer needs calendar-year chunking. Two registry
    knobs control panel size:

    - ``fina_columns``: whitelist of prefixed columns the factor actually
      reads (avoids hauling all ~330 fina columns through pandas).
    - ``last_n_quarters``: keep only the most recent N ``end_date`` rows per
      ``(date, symbol)`` — slope / TTM helpers only need ~20.
    """
    meta = get_factor_meta(factor_id)
    compute_fn = get_factor_function(factor_id)
    data_sources = meta["data_sources"]
    params = meta.get("parameters", {})

    needs_market = any(src == "market_daily" for src in data_sources)
    needs_fina = any(
        src in ("income_q", "balancesheet_q", "cashflow_q", "financial_statements_q")
        for src in data_sources
    )
    needs_factor_store = any(src == "factors_daily" for src in data_sources)

    # ``event_driven=True`` in registry parameters lets a fina factor pull
    # its own per-(symbol, f_ann_date) panel via
    # ``MarketStorage.get_fina_event_panel`` and skip the per-(date, symbol)
    # injection + chunking machinery entirely.
    event_driven = bool(params.get("event_driven", False))

    own_market = market_storage is None
    own_factor = factor_storage is None

    try:
        if market_storage is None and (needs_market or needs_fina):
            market_storage = MarketStorage(read_only=True)
        if factor_storage is None and needs_factor_store:
            factor_storage = FactorStorage()

        input_panel = pd.DataFrame()

        if needs_market:
            window = params.get("window", 252)
            # Convert trading-day window to calendar days (~1.5x is a safe over-estimate).
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            lookback_start = (start_dt - timedelta(days=int(window * 1.5))).strftime("%Y%m%d")

            bars = market_storage.get_bars(
                symbols=None,  # full universe
                start=lookback_start,
                end=end_date,
            )
            if bars.empty and not needs_factor_store:
                return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])
            input_panel = bars

        if needs_fina and not event_driven:
            # PIT snapshots for every trade date in [start, end]. Use the
            # range function — one range-join SQL per fina table replaces
            # N_dates × 3 independent QUALIFY scans. ``fina_columns`` (from
            # registry) keeps the SELECT narrow; ``last_n_quarters`` (also
            # from registry, optional) caps the per-(date, symbol) history
            # depth — slope / TTM factors only need the most recent ~20.
            fina_cols = params.get("fina_columns")
            last_n_quarters = params.get("last_n_quarters")
            fina_all = market_storage.get_fina_snapshot_range(
                start=start_date,
                end=end_date,
                columns=fina_cols,
                last_n_quarters=last_n_quarters,
                delay=1,
            )
            if not fina_all.empty:
                # Range function returns DATE; align with market_daily merge key.
                fina_all["date"] = pd.to_datetime(fina_all["date"])
                if input_panel.empty:
                    input_panel = fina_all
                else:
                    input_panel = input_panel.merge(
                        fina_all, on=["date", "symbol"], how="left"
                    )
            else:
                # No fina snapshot is visible in this date range (e.g. a
                # pre-1990s window before any company listed). The factor
                # function would crash looking up missing fina columns —
                # short-circuit to an empty result frame.
                return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])

        # Composites (e.g. Barra L1) read L3 values directly from factor_storage
        # and event-driven fina factors pull their own data from market_storage
        # — both fall through with an empty input_panel; skip the empty-panel
        # guard so the factor function still runs.
        if input_panel.empty and not needs_factor_store and not event_driven:
            return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])

        # Bind parameters and call the registered compute function. Pass
        # market_storage / factor_storage / start_date / end_date as kwargs
        # only if the function declares them — plain bar-only factors stay
        # unaware of the storage layer. ``params`` (from registry) is the
        # union of "lookback hint" (read above to size the bars fetch) and
        # "actual function kwargs". Filter against the signature so registry-
        # only hints like ``window`` don't get passed to composites that
        # don't take them.
        sig_params = inspect.signature(compute_fn).parameters
        param_kwargs = {k: v for k, v in params.items() if k in sig_params}
        bound_fn = partial(compute_fn, **param_kwargs) if param_kwargs else compute_fn
        candidates = (
            ("market_storage", market_storage),
            ("factor_storage", factor_storage),
            ("start_date", start_date),
            ("end_date", end_date),
        )
        extra_kwargs = {k: v for k, v in candidates if k in sig_params}
        result_series = bound_fn(input_panel, **extra_kwargs)

        if not isinstance(result_series, pd.Series):
            raise TypeError(
                f"Factor {factor_id} compute function must return a pandas Series, "
                f"got {type(result_series)}"
            )

        # Convert MultiIndex Series to DataFrame
        df = result_series.reset_index()
        df.columns = list(df.columns[:-1]) + ["value"]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        df["factor_id"] = factor_id
        return df[["date", "symbol", "factor_id", "value"]]

    finally:
        if own_market and market_storage is not None:
            market_storage.close()
        if own_factor and factor_storage is not None:
            factor_storage.close()


def apply_variant_pipeline(
    raw_df: pd.DataFrame,
    factor_id: str,
    *,
    market_storage: MarketStorage | None = None,
    factor_storage: FactorStorage | None = None,
) -> pd.DataFrame:
    """Apply the factor's declared neutralization pipeline.

    The factor's ``variant`` (from registry) selects the pipeline:

    * ``"none"`` — pass-through, factor values untouched. Used by Barra L1
      composites (which apply the L3 pipeline to each component internally)
      and by any factor that opts out of post-processing.
    * ``"barra_l3"`` — CNE6 L3 style-exposure pipeline: MAD winsorize →
      SW-L1 industry median fill → cs_zscore. Output is a z-scored style
      exposure, **not** a residual against other styles. Used internally by
      Barra L1 composites and available as a variant label for ad-hoc style
      factors that want the same treatment.
    * ``"barra_ind_size"`` — full PLAN.md §2.2 alpha-neutralization pipeline:
      MAD winsorize → SW-L1 industry median fill → cs_zscore → cross-section
      OLS residual against industry dummies + ``f_barra_size`` Size_z (read
      from :class:`FactorLibrary`) → re-cs_zscore. Strips industry and Size
      exposure so what remains is pure alpha.

    Returns
    -------
    pd.DataFrame
        ``[date, symbol, factor_id, value]``, one row per non-null (date, symbol).
    """
    if raw_df.empty:
        return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])

    meta = get_factor_meta(factor_id)
    variant = meta.get("variant", BARRA_IND_SIZE_VARIANT)

    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_series = raw_df.set_index(["date", "symbol"])["value"]

    if variant == NONE_VARIANT:
        series = raw_series
    elif variant in (BARRA_L3_VARIANT, BARRA_IND_SIZE_VARIANT):
        series = raw_series
    else:
        raise ValueError(f"Unknown variant {variant!r} for {factor_id}")

    own_market = market_storage is None
    try:
        if market_storage is None:
            market_storage = MarketStorage(read_only=True)
        start = raw_df["date"].min().strftime("%Y%m%d")
        end = raw_df["date"].max().strftime("%Y%m%d")

        if variant == NONE_VARIANT:
            industry_panel = market_storage.get_industry_panel_range(
                start=start, end=end, level="L1",
            )
            series = industry_median_fill(series, industry_panel)

        elif variant in (BARRA_L3_VARIANT, BARRA_IND_SIZE_VARIANT):
            series = apply_l3_pipeline(raw_series, market_storage, start=start, end=end)

            if variant == BARRA_IND_SIZE_VARIANT:
                with FactorLibrary() as lib:
                    size_df = lib.get_factor(SIZE_L1_ID, start=start, end=end)
                if size_df.empty:
                    raise RuntimeError(
                        f"barra_ind_size pipeline requires {SIZE_L1_ID} to be "
                        f"admitted into the factor library first; got empty Size_z panel."
                    )
                size_df = size_df.rename(columns={"value": "size_z"})
                industry_panel = market_storage.get_industry_panel_range(
                    start=start, end=end, level="L1",
                )
                design = industry_panel.merge(
                    size_df, on=["date", "symbol"], how="outer",
                )
                series = cs_ols_residualize(
                    series,
                    design,
                    dummy_col="industry_code",
                    numeric_cols=("size_z",),
                )
                series = cs_zscore(series)
    finally:
        if own_market and market_storage is not None:
            market_storage.close()

    sub = series.reset_index()
    sub.columns = ["date", "symbol", "value"]
    sub["factor_id"] = factor_id
    sub = sub.dropna(subset=["value"])
    return sub[["date", "symbol", "factor_id", "value"]]


def compute_all(
    start_date: str,
    end_date: str,
    *,
    market_storage: MarketStorage | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute all registered factors for a date range."""
    from backtest.factor.registry import get_registry

    registry = get_registry()
    results = {}
    for factor_id in registry:
        try:
            df = compute_factor(factor_id, start_date, end_date, market_storage=market_storage)
            results[factor_id] = df
        except Exception as exc:
            print(f"WARN: failed to compute {factor_id}: {exc}")
            continue
    return results
