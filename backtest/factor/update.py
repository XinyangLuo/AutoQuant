#!/usr/bin/env python3
"""Incremental update for admitted factors — writes to the **library** DB.

The library (``factor_library.duckdb``) is the long-term home for stabilised
factors. ``admit()`` seeds it once with the factor's full history; this
script's job is to keep the tail in sync with new market_daily rows.

It does *not* touch the work DB (``factors.duckdb``). New factors live in
work via ``backfill``; only after ``admit`` do they show up here.

Usage:
    python -m backtest.factor.update
"""

from __future__ import annotations

from tqdm import tqdm

from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates
from backtest.factor.admission import get_admitted_factor_ids
from backtest.factor.admission_check import (
    _per_date_ridge_residuals,
    _residuals_to_insert_df,
)
from backtest.factor.compute import apply_variant_pipeline, compute_factor
from backtest.factor.dag import get_admission_mode, get_depends_on, topological_sort
from backtest.factor.registry import get_registry
from backtest.factor.storage import FactorLibrary


def main():
    with MarketStorage(read_only=True) as market_storage:
        latest_date = market_storage.get_max_date()
        if latest_date is None:
            print("market_daily is empty. Run cold_start first.")
            return

        factor_ids = get_admitted_factor_ids()
        if not factor_ids:
            print("No admitted factors. Admit one first with "
                  "`python -m backtest.factor.admission admit <factor_id>`.")
            return

        registry = get_registry()
        try:
            factor_ids = topological_sort(factor_ids, registry)
        except ValueError as e:
            print(f"WARNING: dependency cycle in admitted factors, "
                  f"falling back to flat order: {e}")

        with FactorLibrary() as lib:
            failed_ids: set[str] = set()
            for factor_id in tqdm(factor_ids, desc="update"):
                max_date = lib.get_max_date(factor_id)

                if max_date and max_date >= latest_date:
                    continue

                if max_date:
                    resume_dates = get_trade_dates(max_date, latest_date)
                    start = resume_dates[1] if len(resume_dates) > 1 else latest_date
                else:
                    # An admitted factor with no rows in the library is odd —
                    # likely a manual cleanup. Skip rather than mass-backfill.
                    print(f"  {factor_id}: no library rows yet, skip "
                          f"(re-backfill in work + re-admit if intended)")
                    continue

                if start > latest_date:
                    continue

                try:
                    df = compute_factor(
                        factor_id,
                        start,
                        latest_date,
                        market_storage=market_storage,
                        factor_storage=lib,
                    )
                    if df.empty:
                        continue

                    df = apply_variant_pipeline(
                        df, factor_id,
                        market_storage=market_storage,
                        factor_storage=lib,
                    )
                    if df.empty:
                        continue

                    mode = get_admission_mode(factor_id, registry)
                    if mode == "residual":
                        deps = get_depends_on(factor_id, registry)
                        stale = [d for d in deps if d in failed_ids]
                        if stale:
                            print(f"  WARNING {factor_id}: dependencies {stale} "
                                  f"failed to update, storing raw values "
                                  f"(residuals would use stale data)")
                        elif not deps:
                            print(f"  WARNING {factor_id}: admission_mode=residual "
                                  f"but depends_on is empty, storing raw values")
                        else:
                            candidate = df[["date", "symbol", "value"]].copy()
                            wide_parts = []
                            for dep_id in deps:
                                sub = lib.get_factor(dep_id, start=start, end=latest_date)
                                if sub.empty:
                                    raise ValueError(
                                        f"Dependency {dep_id} missing from library "
                                        f"for residual-mode factor {factor_id}"
                                    )
                                wide_parts.append(sub.rename(columns={"value": dep_id}))
                            reg_df = wide_parts[0]
                            for sub in wide_parts[1:]:
                                reg_df = reg_df.merge(sub, on=["date", "symbol"], how="outer")
                            residuals_df, _ = _per_date_ridge_residuals(
                                candidate, reg_df, alpha=1.0,
                            )
                            df = _residuals_to_insert_df(residuals_df, factor_id)

                    if not df.empty:
                        lib.insert_factors(df)
                        print(f"  {factor_id}: {len(df):,} rows ({start} ~ {latest_date})")
                except Exception as exc:
                    print(f"  ERROR {factor_id}: {exc}")
                    failed_ids.add(factor_id)
                    continue

        print("\nUpdate complete.")


if __name__ == "__main__":
    main()
