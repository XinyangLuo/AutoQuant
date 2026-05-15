"""Factor compute engine: fetch raw data, enforce PIT isolation, call registered functions."""

from __future__ import annotations

from functools import partial

import pandas as pd

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.registry import get_factor_function, get_factor_meta
from backtest.factor.storage import FactorStorage


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
