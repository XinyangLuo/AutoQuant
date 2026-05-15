#!/usr/bin/env python3
"""
Consistency test for income / balancesheet / cashflow data.
Randomly samples rows from DuckDB and verifies against Tushare API.

Usage:
    python tests/test_fundamentals.py
    python tests/test_fundamentals.py --n 30
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data.storage import BALANCESHEET_NUMERIC, CASHFLOW_NUMERIC, INCOME_NUMERIC
from backtest.data.tushare_client import api_call, pro


KEY_META = ["symbol", "end_date", "ann_date", "f_ann_date", "report_type", "comp_type", "end_type", "update_flag"]


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


def check_row(db_row: pd.Series, api_df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    """Compare a DB row against the corresponding API row matched by (f_ann_date, update_flag)."""
    if api_df.empty:
        return ["API returned no data"]

    # Keep only consolidated statements
    api_df = api_df[api_df.get("report_type", "1").astype(str) == "1"]
    if api_df.empty:
        return ["API returned no consolidated (report_type=1) rows"]

    match = api_df[
        (api_df["f_ann_date"] == db_row["f_ann_date"])
        & (api_df["update_flag"] == db_row["update_flag"])
    ]
    if match.empty:
        flags = api_df["update_flag"].unique().tolist()
        fdates = api_df["f_ann_date"].unique().tolist()
        return [f"no API row for f_ann_date={db_row['f_ann_date']} update_flag={db_row['update_flag']} (have f_dates={fdates}, flags={flags})"]

    api = match.iloc[0]
    errors = []
    for field in numeric_cols:
        if field not in db_row or field not in api:
            continue
        ok, msg = compare_float(db_row[field], api[field], field, rel_tol=1e-9, abs_tol=1e-6)
        if not ok:
            errors.append(msg)

    return errors


def sample_table(conn, table: str, n: int) -> pd.DataFrame:
    """Sample random rows from a fundamentals table."""
    try:
        return conn.execute(f"""
            SELECT *
            FROM {table}
            ORDER BY random()
            LIMIT ?
        """, [n]).fetchdf()
    except Exception:
        return pd.DataFrame()


def test_table(conn, table: str, label: str, api_func, numeric_cols: list[str], n: int) -> tuple[int, int, list]:
    """Sample *n* rows from *table* and verify against *api_func*."""
    sample = sample_table(conn, table, n)
    if sample.empty:
        print(f"{label}: no data in DB. Run backfill script first.")
        return 0, 0, []

    passed = 0
    failed = 0
    failures = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc=label):
        symbol = row["symbol"]
        period = row["end_date"]

        api_df = api_call(api_func, ts_code=symbol, period=period)
        errors = check_row(row, api_df, numeric_cols)

        if errors:
            failed += 1
            failures.append({"symbol": symbol, "end_date": period, "f_ann_date": row["f_ann_date"],
                             "update_flag": row["update_flag"], "errors": errors})
        else:
            passed += 1

    return passed, failed, failures


def main():
    parser = argparse.ArgumentParser(description="Verify fundamentals data against Tushare API")
    parser.add_argument("--n", type=int, default=15, help="Number of random rows per table")
    parser.add_argument("--db", type=str, default="data/duckdb/market.duckdb", help="DuckDB path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    configs = [
        ("income_q", "income", pro.income, INCOME_NUMERIC),
        ("balancesheet_q", "balancesheet", pro.balancesheet, BALANCESHEET_NUMERIC),
        ("cashflow_q", "cashflow", pro.cashflow, CASHFLOW_NUMERIC),
    ]

    total_passed = 0
    total_failed = 0
    all_failures = []

    for table, label, api_func, numeric_cols in configs:
        passed, failed, failures = test_table(conn, table, label, api_func, numeric_cols, args.n)
        total_passed += passed
        total_failed += failed
        all_failures.extend(failures)

    conn.close()

    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed out of {total_passed + total_failed}")

    if all_failures:
        print(f"\nFailure details (first 5):")
        for f in all_failures[:5]:
            print(f"  {f['symbol']} {f['end_date']} f_ann={f['f_ann_date']} flag={f['update_flag']}")
            for e in f["errors"][:3]:
                print(f"    - {e}")

    print("=" * 60)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
