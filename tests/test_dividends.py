#!/usr/bin/env python3
"""
Consistency test for dividend data.
Randomly samples rows from DuckDB and verifies against Tushare API.

Usage:
    python tests/test_dividends.py
    python tests/test_dividends.py --n 50
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


def check_dividend(db_row: pd.Series, api_df: pd.DataFrame) -> list[str]:
    """Compare dividend fields. Returns list of mismatch messages."""
    if api_df.empty:
        return ["API dividend returned no data"]

    match = api_df[
        (api_df["ts_code"] == db_row["symbol"]) &
        (api_df["end_date"] == db_row["end_date"])
    ]
    if match.empty:
        return [f"symbol {db_row['symbol']} end_date {db_row['end_date']} not found in API"]

    # Filter div_proc == '实施'
    match = match[match["div_proc"] == "实施"]
    if match.empty:
        return [f"No implemented dividend found for {db_row['symbol']} {db_row['end_date']}"]

    # Match by (ann_date, ex_date) to handle multiple dividends per end_date.
    # Tushare may return NULL ex_date; use the same fallback chain as the fetcher.
    match["ann_date"] = match["ann_date"].fillna(match["end_date"])
    match["ex_date"] = (
        match["ex_date"]
        .fillna(match["pay_date"])
        .fillna(match["ann_date"])
    )
    sub = match[
        (match["ann_date"] == db_row["ann_date"]) &
        (match["ex_date"] == db_row["ex_date"])
    ]
    if sub.empty:
        return [
            f"No API row matching ann_date={db_row['ann_date']} ex_date={db_row['ex_date']} "
            f"for {db_row['symbol']} {db_row['end_date']}"
        ]

    api = sub.iloc[0]
    errors = []

    for field in ("cash_div", "cash_div_tax", "stk_div", "stk_bo_rate"):
        if field not in api:
            continue
        ok, msg = compare_float(db_row[field], api[field], field, rel_tol=1e-9, abs_tol=1e-6)
        if not ok:
            errors.append(msg)

    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify dividend data against Tushare API")
    parser.add_argument("--n", type=int, default=50, help="Number of random rows to sample")
    parser.add_argument("--db", type=str, default="data/duckdb/market.duckdb", help="DuckDB path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    # TODO: Adjust schema query once dividends table is created
    try:
        sample = conn.execute("""
            SELECT symbol, end_date, ann_date, ex_date, cash_div, cash_div_tax, stk_div, stk_bo_rate
            FROM dividends
            WHERE div_proc = '实施'
            ORDER BY random()
            LIMIT ?
        """, [args.n]).fetchdf()
    except Exception as exc:
        print(f"dividends table not ready: {exc}")
        conn.close()
        sys.exit(0)

    conn.close()

    if sample.empty:
        print("No dividend data found. Run fetch script first.")
        sys.exit(1)

    print(f"Checking {len(sample)} rows against Tushare API...")
    passed = 0
    failed = 0
    failures = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Checking"):
        symbol = row["symbol"]
        end_date = row["end_date"]

        api_div = api_call(pro.dividend, ts_code=symbol, end_date=end_date)
        errors = check_dividend(row, api_div)

        if errors:
            failed += 1
            failures.append({"symbol": symbol, "end_date": end_date, "errors": errors})
        else:
            passed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(sample)}")

    if failures:
        print(f"\nFailure details (first 5):")
        for f in failures[:5]:
            print(f"  {f['symbol']} {f['end_date']}")
            for e in f["errors"]:
                print(f"    - {e}")

    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
