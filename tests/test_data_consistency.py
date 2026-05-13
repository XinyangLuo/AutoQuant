#!/usr/bin/env python3
"""
Data consistency test: randomly sample rows from DuckDB and verify against Tushare API.

Usage:
    python tests/test_data_consistency.py
    python tests/test_data_consistency.py --n 200
    python tests/test_data_consistency.py --n 50 --check-adj
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


def check_daily(db_row: pd.Series, api_df: pd.DataFrame) -> list[str]:
    """Compare OHLCV+amount. Returns list of mismatch messages."""
    if api_df.empty:
        return ["API daily returned no data"]

    api = api_df.iloc[0]
    errors = []

    for field in ("open", "high", "low", "close", "pre_close"):
        ok, msg = compare_float(db_row[field], api[field], field)
        if not ok:
            errors.append(msg)

    # Volume: DB stores actual shares (×100), API returns hands
    api_vol = round(float(api["vol"]) * 100) if pd.notna(api["vol"]) else None
    ok, msg = compare_float(db_row["volume"], api_vol, "volume", rel_tol=1e-9, abs_tol=100)
    if not ok:
        errors.append(msg)

    ok, msg = compare_float(db_row["amount"], api["amount"], "amount")
    if not ok:
        errors.append(msg)

    return errors


def check_adj_factor(db_row: pd.Series, api_df: pd.DataFrame) -> list[str]:
    """Compare adj_factor. Returns list of mismatch messages."""
    if api_df.empty:
        return ["API adj_factor returned no data"]

    api = api_df.iloc[0]
    ok, msg = compare_float(db_row["adj_factor"], api["adj_factor"], "adj_factor")
    if not ok:
        return [msg]
    return []


def main():
    parser = argparse.ArgumentParser(description="Verify DB data consistency against Tushare API")
    parser.add_argument("--n", type=int, default=100, help="Number of random rows to sample")
    parser.add_argument("--db", type=str, default="data/duckdb/market.duckdb", help="DuckDB path")
    parser.add_argument("--check-adj", action="store_true", help="Also verify adj_factor")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    # Sample random rows
    fields = "date, symbol, open, high, low, close, pre_close, volume, amount"
    if args.check_adj:
        fields += ", adj_factor"

    print(f"Sampling {args.n} random rows from DB...")
    sample = conn.execute(f"""
        SELECT {fields}
        FROM market_daily
        ORDER BY random()
        LIMIT {args.n}
    """).fetchdf()

    conn.close()

    if sample.empty:
        print("DB is empty.")
        sys.exit(1)

    print(f"Checking {len(sample)} rows against Tushare API...")
    passed = 0
    failed = 0
    failures = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Checking"):
        trade_date = row["date"].strftime("%Y%m%d")
        symbol = row["symbol"]

        api_daily = api_call(pro.daily, ts_code=symbol, start_date=trade_date, end_date=trade_date)
        errors = check_daily(row, api_daily)

        if args.check_adj:
            api_adj = api_call(pro.adj_factor, ts_code=symbol, start_date=trade_date, end_date=trade_date)
            errors += check_adj_factor(row, api_adj)

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
