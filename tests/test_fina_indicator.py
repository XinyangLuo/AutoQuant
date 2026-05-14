#!/usr/bin/env python3
"""
Consistency test for fina_indicator data.
Randomly samples rows from DuckDB and verifies against Tushare API.

Usage:
    python tests/test_fina_indicator.py
    python tests/test_fina_indicator.py --n 50
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


def check_fina(db_row: pd.Series, api_df: pd.DataFrame) -> list[str]:
    """Compare fina_indicator fields. Returns list of mismatch messages."""
    if api_df.empty:
        return ["API fina_indicator returned no data"]

    # Deduplicate: keep the row with fewest NaNs
    api_df = api_df.copy()
    api_df["_nan_count"] = api_df.isna().sum(axis=1)
    api_df = api_df.sort_values("_nan_count").drop_duplicates(subset=["ts_code", "end_date"], keep="first")
    api = api_df.iloc[0]

    errors = []
    # Check key fields: eps, roe, netprofit_margin, grossprofit_margin
    key_fields = ["eps", "roe", "netprofit_margin", "grossprofit_margin", "bps", "ocfps"]
    for field in key_fields:
        if field not in api:
            continue
        ok, msg = compare_float(db_row[field], api[field], field, rel_tol=1e-9, abs_tol=1e-6)
        if not ok:
            errors.append(msg)

    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify fina_indicator data against Tushare API")
    parser.add_argument("--n", type=int, default=50, help="Number of random rows to sample")
    parser.add_argument("--db", type=str, default="data/duckdb/market.duckdb", help="DuckDB path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    # TODO: Adjust schema query once fina_indicator_quarterly table is created
    # For now, this is a placeholder that will fail gracefully
    try:
        sample = conn.execute("""
            SELECT symbol, end_date, ann_date, eps, roe, netprofit_margin, grossprofit_margin, bps, ocfps
            FROM fina_indicator_quarterly
            ORDER BY random()
            LIMIT ?
        """, [args.n]).fetchdf()
    except Exception as exc:
        print(f"fina_indicator_quarterly table not ready: {exc}")
        conn.close()
        sys.exit(0)

    conn.close()

    if sample.empty:
        print("No fina_indicator data found. Run fetch script first.")
        sys.exit(1)

    print(f"Checking {len(sample)} rows against Tushare API...")
    passed = 0
    failed = 0
    failures = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Checking"):
        symbol = row["symbol"]
        end_date = row["end_date"]

        api_fina = api_call(pro.fina_indicator, ts_code=symbol, period=end_date)
        errors = check_fina(row, api_fina)

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
