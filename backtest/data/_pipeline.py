"""Generic backfill / update loops shared between scripts.

These helpers capture two reusable patterns:
  * `backfill_by_symbol` — per-symbol resume loop (cold-start of fina, dividends, ...)
  * `update_by_ann_date` — per-trade-date incremental loop driven by an ann_date cursor

Plus shared CLI / print helpers used by the backfill scripts and cold_start.
"""

import argparse
from datetime import datetime
from typing import Callable

import pandas as pd
from tqdm import tqdm

from backtest.data.stock_list import fetch_stock_list
from backtest.data.storage import MarketStorage
from backtest.data.trade_calendar import get_trade_dates


def backfill_by_symbol(
    *,
    label: str,
    fetch_one: Callable[[str], pd.DataFrame],
    insert: Callable[[pd.DataFrame], None],
    get_done: Callable[[], set[str]],
    force: bool = False,
    stock_list: pd.DataFrame | None = None,
) -> None:
    """Loop the full universe of symbols, fetching and inserting one at a time.

    Already-fetched symbols are skipped via *get_done* unless *force=True*.
    Failures on individual symbols are logged but do not abort the run.
    """
    if stock_list is None:
        stock_list = fetch_stock_list()
    all_symbols = sorted(stock_list["ts_code"].tolist())

    if force:
        todo = all_symbols
    else:
        done = get_done()
        todo = [s for s in all_symbols if s not in done]
        print(f"{label}: {len(done)} symbols already have data, "
              f"{len(todo)} to fetch")

    if not todo:
        print(f"{label}: nothing to do.")
        return

    failed: list[tuple[str, str]] = []
    for symbol in tqdm(todo, desc=label):
        try:
            df = fetch_one(symbol)
            if not df.empty:
                insert(df)
        except Exception as exc:
            failed.append((symbol, str(exc)))
            print(f"\n  WARN: failed {symbol}: {exc}")
            continue

    if failed:
        head = [s for s, _ in failed][:20]
        tail = " ..." if len(failed) > 20 else ""
        print(f"\n  Failed symbols ({len(failed)}): {head}{tail}")


def update_by_ann_date(
    *,
    label: str,
    get_max_ann_date: Callable[[], str | None],
    fetch_by_ann_date: Callable[[str], pd.DataFrame],
    insert: Callable[[pd.DataFrame], None],
) -> None:
    """Scan trade dates from MAX(ann_date) to today, upserting per day.

    The boundary day is re-fetched so late-arriving rows are captured
    (UPSERT makes this idempotent).
    """
    today = datetime.now().strftime("%Y%m%d")
    max_ann = get_max_ann_date()

    if max_ann is None:
        print(f"{label}: table is empty. Run the backfill script first.")
        return

    trade_dates = get_trade_dates(max_ann, today)
    if not trade_dates:
        print(f"{label}: no trade dates to update (last ann_date {max_ann}).")
        return

    print(f"{label}: {len(trade_dates)} trade dates to scan "
          f"({trade_dates[0]} ~ {trade_dates[-1]})")

    failed: list[tuple[str, str]] = []
    rows_added = 0
    for ann_date in tqdm(trade_dates, desc=label):
        try:
            df = fetch_by_ann_date(ann_date)
            if not df.empty:
                insert(df)
                rows_added += len(df)
        except Exception as exc:
            failed.append((ann_date, str(exc)))
            print(f"\n  WARN: failed {ann_date}: {exc}")
            continue

    if failed:
        print(f"\n  Failed dates ({len(failed)}): {[d for d, _ in failed]}")

    print(f"{label}: scanned {len(trade_dates)} days, upserted {rows_added:,} rows.")


def print_stats(label: str, stats: dict, *, date_col: str = "ann_date", prefix: str = "") -> None:
    """Phase summary: 1-line stats sandwiched between dashed separators.

    *date_col* selects which `min_{date_col}` / `max_{date_col}` keys to read.
    """
    print("-" * 50)
    print(f"{label}: {stats['total_rows']:,} rows, "
          f"{stats['total_symbols']:,} symbols, "
          f"{prefix}{stats[f'min_{date_col}']} ~ {stats[f'max_{date_col}']}")
    print("-" * 50)


def run_symbol_backfill_cli(
    name: str,
    *,
    backfill_fn: Callable[..., None],
    get_stats: Callable[[MarketStorage], dict],
    date_col: str = "ann_date",
) -> None:
    """Standard CLI entry point for a per-symbol backfill script.

    Wires up `--force`, opens MarketStorage, calls *backfill_fn(storage, force=...)*,
    then prints a final stats banner. Used by `backfill_fina_indicator` /
    `backfill_dividends` (and any future sibling).
    """
    parser = argparse.ArgumentParser(description=f"Backfill {name}")
    parser.add_argument("--force", action="store_true",
                        help="Refetch even symbols already in DB")
    args = parser.parse_args()

    with MarketStorage() as storage:
        backfill_fn(storage, force=args.force)
        stats = get_stats(storage)

    print("\n" + "=" * 50)
    print(f"{name} backfill complete.")
    print(f"  Total rows    : {stats['total_rows']:,}")
    print(f"  Total symbols : {stats['total_symbols']:,}")
    print(f"  {date_col} range: "
          f"{stats[f'min_{date_col}']} ~ {stats[f'max_{date_col}']}")
    print("=" * 50)
