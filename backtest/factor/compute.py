"""Factor compute engine: fetch raw data, enforce PIT isolation, call registered functions."""

from __future__ import annotations

from functools import partial

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.registry import get_factor_function, get_factor_meta
from backtest.factor.storage import FactorStorage
from backtest.factor.transforms import cap_neutralize, industry_neutralize
from backtest.factor.variants import (
    RAW_VARIANT,
    expand_variant_names,
    normalize_neutralizations,
    parse_variant,
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
    Note: only the raw factor is computed here. To produce neutralization
    variants, pass the result to :func:`apply_neutralizations`.
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

    own_market = market_storage is None

    try:
        if market_storage is None:
            market_storage = MarketStorage()

        input_panel = pd.DataFrame()

        if needs_market:
            window = params.get("window", 252)
            # Convert trading-day window to calendar days (~1.5x is a safe over-estimate).
            from datetime import datetime, timedelta

            start_dt = datetime.strptime(start_date, "%Y%m%d")
            lookback_start = (start_dt - timedelta(days=int(window * 1.5))).strftime("%Y%m%d")

            bars = market_storage.get_bars(
                symbols=None,  # full universe
                start=lookback_start,
                end=end_date,
            )
            if bars.empty:
                return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])
            input_panel = bars

        if needs_fina:
            # Financial data requires per-date PIT snapshots.
            # We fetch the snapshot for each trade date in the range and
            # concatenate them into a single (date, symbol) panel.
            trade_dates = get_trade_dates(start_date, end_date)
            fina_dfs: list[pd.DataFrame] = []
            for date in trade_dates:
                snap = market_storage.get_fina_snapshot(as_of_date=date)
                if not snap.empty:
                    snap["date"] = pd.Timestamp(date)
                    fina_dfs.append(snap)
            if fina_dfs:
                fina_all = pd.concat(fina_dfs, ignore_index=True)
                if input_panel.empty:
                    input_panel = fina_all
                else:
                    input_panel = input_panel.merge(
                        fina_all, on=["date", "symbol"], how="left"
                    )

        if input_panel.empty:
            return pd.DataFrame(columns=["date", "symbol", "factor_id", "value"])

        # Bind parameters and call the registered compute function
        bound_fn = partial(compute_fn, **params) if params else compute_fn
        result_series = bound_fn(input_panel)

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


def apply_neutralizations(
    raw_df: pd.DataFrame,
    factor_id: str,
    *,
    market_storage: MarketStorage | None = None,
) -> pd.DataFrame:
    """对 raw 因子值应用 registry 声明的所有变体,返回带 ``variant`` 列的 long DF。

    Pipeline per variant: 行业中性化(可选) → 市值中性化(可选)。
    raw 变体直接拷贝 raw_df 的值,只追加 ``variant='raw'``。

    Parameters
    ----------
    raw_df : pd.DataFrame
        :func:`compute_factor` 的输出: ``[date, symbol, factor_id, value]``。
    factor_id : str
        用于从 registry 拿声明的 neutralizations。
    market_storage : MarketStorage, optional
        复用已打开的 storage 句柄;None 则自建。

    Returns
    -------
    pd.DataFrame
        ``[date, symbol, factor_id, variant, value]``。每个声明变体一份数据。
    """
    if raw_df.empty:
        return pd.DataFrame(
            columns=["date", "symbol", "factor_id", "variant", "value"]
        )

    meta = get_factor_meta(factor_id)
    neutralizations = normalize_neutralizations(meta.get("neutralizations"))
    variant_names = [
        v for v in expand_variant_names(meta.get("neutralizations"))
    ]

    # Determine what panels we'll need
    need_industry_levels: set[str] = set()
    need_cap_fields: set[str] = set()
    for spec in neutralizations:
        if spec["industry"]:
            need_industry_levels.add(
                "L1" if spec["industry"] == "SW-L1" else "L2"
            )
        if spec["cap"]:
            field, _, _ = spec["cap"].partition("-")
            need_cap_fields.add(field)

    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    start = raw_df["date"].min().strftime("%Y%m%d")
    end = raw_df["date"].max().strftime("%Y%m%d")

    own_market = market_storage is None
    try:
        if market_storage is None:
            market_storage = MarketStorage()

        # Pre-fetch panels once for each declared dimension
        industry_panels: dict[str, pd.DataFrame] = {}
        for lvl in need_industry_levels:
            industry_panels[lvl] = market_storage.get_industry_panel_range(
                start=start, end=end, level=lvl,
            )

        cap_panels: dict[str, pd.DataFrame] = {}
        if need_cap_fields:
            bars = market_storage.get_bars(
                start=start, end=end,
                columns=list(need_cap_fields),
            )
            for field in need_cap_fields:
                if field in bars.columns:
                    cap_panels[field] = bars[["date", "symbol", field]].copy()

        raw_series = raw_df.set_index(["date", "symbol"])["value"]

        out_frames: list[pd.DataFrame] = []
        for spec, variant in zip(neutralizations, variant_names):
            if variant == RAW_VARIANT:
                series = raw_series
            else:
                series = raw_series
                ind = spec["industry"]
                cap = spec["cap"]
                if ind:
                    lvl = "L1" if ind == "SW-L1" else "L2"
                    series = industry_neutralize(series, industry_panels[lvl])
                if cap:
                    field, _, qpart = cap.partition("-")
                    q = int(qpart[1:])  # "q5" → 5
                    series = cap_neutralize(
                        series, cap_panels[field],
                        cap_field=field, quantiles=q,
                    )

            sub = series.reset_index()
            sub.columns = ["date", "symbol", "value"]
            sub["factor_id"] = factor_id
            sub["variant"] = variant
            sub = sub.dropna(subset=["value"])
            out_frames.append(sub[["date", "symbol", "factor_id", "variant", "value"]])

        if not out_frames:
            return pd.DataFrame(
                columns=["date", "symbol", "factor_id", "variant", "value"]
            )
        return pd.concat(out_frames, ignore_index=True)

    finally:
        if own_market and market_storage is not None:
            market_storage.close()


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
