#!/usr/bin/env python3
"""
Consistency test for new columns: limit_up/down and daily_basic indicators.
Randomly samples rows from DuckDB and verifies against Tushare API.

Usage:
    python tests/test_new_columns.py
    python tests/test_new_columns.py --n 100
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data.tushare_client import api_call, pro


def compare_float(a, b, field: str, rel_tol: float = 1e-5, abs_tol: float = 1e-6) -> tuple[bool, str | None]:
    """Compare two numeric values with tolerance. Returns (ok, error_msg)."""
    if pd.isna(a) and pd.isna(b):
        return True, None
    if pd.isna(a) or pd.isna(b):
        return False, f"{field}: DB={a}, API={b} (one is NULL)"

    a, b = float(a), float(b)
    diff = abs(a - b)
    tol = max(rel_tol * max(abs(a), abs(b)), abs_tol)

    if diff <= tol:
        return True, None
    return False, f"{field}: DB={a:.4f}, API={b:.4f}, diff={diff:.6f}"


def check_limit_prices(db_row: pd.Series, limit_df: pd.DataFrame) -> list[str]:
    """Compare limit_up/down. Returns list of mismatch messages."""
    if limit_df.empty:
        return ["API stk_limit returned no data"]

    match = limit_df[limit_df["ts_code"] == db_row["symbol"]]
    if match.empty:
        return [f"symbol {db_row['symbol']} not found in API stk_limit"]

    api = match.iloc[0]
    errors = []

    for field in ("limit_up", "limit_down"):
        ok, msg = compare_float(db_row[field], api[field], field)
        if not ok:
            errors.append(msg)

    return errors


def check_daily_basic(db_row: pd.Series, basic_df: pd.DataFrame) -> list[str]:
    """Compare daily_basic indicators. Returns list of mismatch messages."""
    if basic_df.empty:
        return ["API daily_basic returned no data"]

    match = basic_df[basic_df["ts_code"] == db_row["symbol"]]
    if match.empty:
        return [f"symbol {db_row['symbol']} not found in API daily_basic"]

    api = match.iloc[0]
    errors = []

    fields = (
        "turnover_rate", "turnover_rate_f", "volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "dv_ttm",
        "total_share", "float_share", "free_share",
        "total_mv", "circ_mv",
    )

    for field in fields:
        if field not in api:
            continue
        ok, msg = compare_float(db_row[field], api[field], field, rel_tol=1e-9, abs_tol=1e-6)
        if not ok:
            errors.append(msg)

    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify new columns against Tushare API")
    parser.add_argument("--n", type=int, default=100, help="Number of random rows to sample")
    parser.add_argument("--db", type=str, default="data/duckdb/market.duckdb", help="DuckDB path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    # Sample random rows that have both limit and daily_basic data
    print(f"Sampling {args.n} random rows from DB...")
    sample = conn.execute("""
        SELECT date, symbol, limit_up, limit_down,
               turnover_rate, turnover_rate_f, volume_ratio,
               pe, pe_ttm, pb, ps, ps_ttm,
               dv_ratio, dv_ttm,
               total_share, float_share, free_share,
               total_mv, circ_mv
        FROM market_daily
        WHERE turnover_rate IS NOT NULL
        ORDER BY random()
        LIMIT ?
    """, [args.n]).fetchdf()

    conn.close()

    if sample.empty:
        print("No rows with daily_basic data found. Run backfill first.")
        sys.exit(1)

    print(f"Checking {len(sample)} rows against Tushare API...")
    passed = 0
    failed = 0
    failures = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Checking"):
        trade_date = row["date"].strftime("%Y%m%d")
        symbol = row["symbol"]

        # Fetch limit prices for the whole day (single API call per date)
        api_limit = api_call(pro.stk_limit, trade_date=trade_date)
        errors = check_limit_prices(row, api_limit)

        # Fetch daily_basic for the whole day (single API call per date)
        api_basic = api_call(pro.daily_basic, trade_date=trade_date)
        errors += check_daily_basic(row, api_basic)

        if errors:
            failed += 1
            failures.append({"date": trade_date, "symbol": symbol, "errors": errors})
        else:
            passed += 1

    # Report
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(sample)}")

    if failures:
        print(f"\nFailure details (first 5):")
        for f in failures[:5]:
            print(f"  {f['date']} {f['symbol']}")
            for e in f["errors"]:
                print(f"    - {e}")

    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
